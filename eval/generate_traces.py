"""
Generate evaluation traces by running the actual agent against gold dataset queries.

This script replicates the agent setup from main.py (tools, system prompt, YAML
config) but runs headlessly — no Chainlit UI required.  Each gold query is sent
through the real agent and the response (text + tool calls) is written to the
traces JSONL file consumed by the Foundry eval runner.

Usage:
    uv run python eval/generate_traces.py                        # defaults
    uv run python eval/generate_traces.py --gold eval/datasets/sql_agent_gold_starter.jsonl
    uv run python eval/generate_traces.py --out eval/traces/agent_runs.jsonl
    uv run python eval/generate_traces.py --profile "Tactical Readiness AI"   # single profile
    uv run python eval/generate_traces.py --model gpt-4o-mini --temperature 0.2 --top-p 0.9
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from openai.lib.azure import AsyncAzureOpenAI
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework import RawAgent, FunctionTool, tool
from agent_framework._types import Message as ChatMessage, Content, ChatOptions

from tools import SqlDatabase
from rag_tools import semantic_search
from prompt_config import load_prompt_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("generate_traces")

# --------------------------------------------------------------------------- #
# Profile → tool mapping (mirrors main.py logic)
# --------------------------------------------------------------------------- #
AI_SEARCH_BOT_PROFILE = "Technical Maintenance AI"
SQL_BOT_PROFILE = "Tactical Readiness AI"
HYBRID_BOT_PROFILE = "Combat LogiGuard AI"

# Gold dataset uses short names; map to Chainlit profile names
_GOLD_PROFILE_MAP: dict[str, str] = {
    "tactical readiness ai": SQL_BOT_PROFILE,
    "technical maintenance ai": AI_SEARCH_BOT_PROFILE,
    "combat logiguard ai": HYBRID_BOT_PROFILE,
    "hybrid": HYBRID_BOT_PROFILE,
    "sql": SQL_BOT_PROFILE,
    "search": AI_SEARCH_BOT_PROFILE,
}


def _sanitize_endpoint(endpoint: str) -> str:
    if endpoint.startswith("https://https://"):
        return endpoint.replace("https://https://", "https://", 1)
    return endpoint


def _normalize_tools(
    raw_tools: list[Any],
    description_overrides: Optional[dict[str, str]] = None,
) -> list[Any]:
    overrides = description_overrides or {}
    normalized: list[Any] = []
    for raw_tool in raw_tools:
        if isinstance(raw_tool, FunctionTool):
            normalized.append(raw_tool)
        elif callable(raw_tool):
            tool_name = getattr(raw_tool, "__name__", None)
            description = overrides.get(str(tool_name), None) if tool_name else None
            normalized.append(tool(raw_tool, description=description))
        else:
            normalized.append(raw_tool)
    return normalized


def _resolve_profile(raw: Optional[str]) -> str:
    """Map gold dataset chat_profile value to Chainlit profile name."""
    if not raw:
        return HYBRID_BOT_PROFILE
    normalized = " ".join(raw.strip().lower().split())
    return _GOLD_PROFILE_MAP.get(normalized, HYBRID_BOT_PROFILE)


# --------------------------------------------------------------------------- #
# Agent + tool factory (cached per profile)
# --------------------------------------------------------------------------- #

class _AgentKit:
    """Bundles agent + tools + metadata for a single profile."""

    def __init__(
        self,
        agent: RawAgent,
        tools: list[Any],
        logical_profile: str,
        prompt_manifest: dict[str, str],
        tool_names: list[str],
    ):
        self.agent = agent
        self.tools = tools
        self.logical_profile = logical_profile
        self.prompt_manifest = prompt_manifest
        self.tool_names = tool_names


_agent_cache: dict[str, _AgentKit] = {}

# Module-level model parameter slots set by generate() before agent creation
_TEMPERATURE: Optional[float] = None
_TOP_P: Optional[float] = None


def _get_agent_kit(
    profile: str,
    llm: AzureOpenAIChatClient,
    db_tools_raw: list[Any],
    search_tools_raw: list[Any],
) -> _AgentKit:
    if profile in _agent_cache:
        return _agent_cache[profile]

    available_tool_names = {
        name
        for name in [
            *(getattr(t, "__name__", None) for t in db_tools_raw),
            *(getattr(t, "__name__", None) for t in search_tools_raw),
        ]
        if isinstance(name, str) and name
    }

    prompt_bundle = load_prompt_bundle(
        chat_profile=profile,
        available_tool_names=available_tool_names,
    )

    # Build description overrides from YAML
    overrides: dict[str, str] = {}
    for tool_def in prompt_bundle.tools_config.get("tools", []):
        if not isinstance(tool_def, dict):
            continue
        tn = tool_def.get("tool_name")
        desc = tool_def.get("description")
        enabled = tool_def.get("enabled_profiles") or []
        if not isinstance(tn, str) or not isinstance(desc, str):
            continue
        if enabled and prompt_bundle.logical_profile not in enabled:
            continue
        overrides[tn] = desc

    db_tools = _normalize_tools(db_tools_raw, overrides)
    search_tools = _normalize_tools(search_tools_raw, overrides)

    if profile == AI_SEARCH_BOT_PROFILE:
        tools = search_tools
    elif profile == SQL_BOT_PROFILE:
        tools = db_tools
    else:
        tools = db_tools + search_tools

    # model_params injected via module-level slots set by generate()
    model_options: ChatOptions = {}  # type: ignore[typeddict-item]
    if _TEMPERATURE is not None:
        model_options["temperature"] = _TEMPERATURE
    if _TOP_P is not None:
        model_options["top_p"] = _TOP_P

    agent = RawAgent(
        client=llm,
        name="eval_agent",
        instructions=(prompt_bundle.system_prompt or "").strip(),
        default_options=model_options if model_options else None,
    )

    tool_name_list = [getattr(t, "name", str(t)) for t in tools]

    kit = _AgentKit(
        agent=agent,
        tools=tools,
        logical_profile=prompt_bundle.logical_profile,
        prompt_manifest=prompt_bundle.prompt_manifest,
        tool_names=tool_name_list,
    )
    _agent_cache[profile] = kit
    return kit


# --------------------------------------------------------------------------- #
# Run a single query
# --------------------------------------------------------------------------- #

async def _run_query(
    kit: _AgentKit,
    query: str,
    retry_delay: float = 15.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Run a single query through the agent and return a trace dict."""

    messages = [ChatMessage(role="user", contents=[Content.from_text(query)])]

    for attempt in range(1, max_retries + 1):
        try:
            response = await kit.agent.run(
                messages=messages,
                tools=kit.tools,
                stream=False,
            )
            break
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "rate" in error_str.lower():
                wait = retry_delay * attempt
                logger.warning(
                    "Rate-limited on attempt %d/%d, waiting %.0fs: %s",
                    attempt, max_retries, wait, error_str[:120],
                )
                await asyncio.sleep(wait)
                if attempt == max_retries:
                    logger.error("Exhausted retries for query: %s", query[:80])
                    return {
                        "input": query,
                        "output": f"[ERROR after {max_retries} retries: {error_str[:200]}]",
                        "tool_events": [],
                        "error": True,
                    }
            else:
                logger.error("Non-retryable error for query '%s': %s", query[:80], exc)
                return {
                    "input": query,
                    "output": f"[ERROR: {error_str[:300]}]",
                    "tool_events": [],
                    "error": True,
                }

    # Extract tool events from response messages
    tool_events: list[dict[str, Any]] = []
    resp_dict = response.to_dict()
    for msg in resp_dict.get("messages", []):
        for content in msg.get("contents", []):
            ctype = content.get("type")
            if ctype == "function_call":
                call_id = content.get("call_id", "")
                name = content.get("name", "")
                arguments = content.get("arguments", "")
                if isinstance(arguments, (dict, list)):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_events.append({
                    "call_id": call_id,
                    "name": name,
                    "arguments": str(arguments),
                    "result": None,  # filled in below
                })
            elif ctype == "function_result":
                call_id = content.get("call_id", "")
                result = content.get("result", "")
                if isinstance(result, (dict, list)):
                    result = json.dumps(result, ensure_ascii=False)
                # Match to existing event by call_id
                matched = False
                for evt in tool_events:
                    if evt["call_id"] == call_id and evt["result"] is None:
                        evt["result"] = str(result)
                        matched = True
                        break
                if not matched:
                    tool_events.append({
                        "call_id": call_id,
                        "name": "",
                        "arguments": "",
                        "result": str(result),
                    })

    return {
        "input": query,
        "output": (response.text or "").strip(),
        "tool_events": tool_events,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def generate(
    gold_path: Path,
    output_path: Path,
    profile_filter: Optional[str] = None,
    inter_query_delay: float = 2.0,
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> None:
    # --- LLM setup (mirrors main.py) ---
    endpoint = _sanitize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    deployment = model_override or os.getenv("AZURE_OPENAI_MODEL", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([endpoint, deployment, api_key]):
        logger.error("Missing AZURE_OPENAI_ENDPOINT / MODEL / API_KEY in .env")
        sys.exit(1)

    logger.info(
        "LLM config: model=%s  temperature=%s  top_p=%s",
        deployment, temperature, top_p,
    )

    async_client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
        max_retries=2,
        timeout=120.0,
    )

    llm = AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment,
        api_key=api_key,
        api_version=api_version,
        async_client=async_client,
    )

    # --- Model params (module-level for agent factory) ---
    global _TEMPERATURE, _TOP_P  # noqa: PLW0603
    _TEMPERATURE = temperature
    _TOP_P = top_p

    # --- Tools setup (mirrors main.py) ---
    connection_string = os.getenv("AZURE_SQL_CONNECTIONSTRING")
    if not connection_string:
        logger.error("AZURE_SQL_CONNECTIONSTRING not set")
        sys.exit(1)

    db = SqlDatabase(connection_string)
    db_tools_raw = [db.list_tables, db.list_views, db.describe_table, db.read_query]
    search_tools_raw = [semantic_search]

    # --- Load gold dataset ---
    gold_rows: list[dict[str, Any]] = []
    with gold_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                gold_rows.append(json.loads(stripped))

    logger.info("Loaded %d gold rows from %s", len(gold_rows), gold_path)

    # --- Run each query ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(gold_rows, 1):
            case_id = row.get("case_id", f"row_{idx}")
            query = row.get("query", "")
            raw_profile = row.get("chat_profile", "")
            profile = _resolve_profile(raw_profile)

            if profile_filter and profile.lower() != profile_filter.lower():
                logger.info("  [%s] skipped (profile filter)", case_id)
                continue

            kit = _get_agent_kit(profile, llm, db_tools_raw, search_tools_raw)

            logger.info(
                "  [%d/%d] %s  profile=%s  query=%s",
                idx, len(gold_rows), case_id, profile, query[:60],
            )

            trace = await _run_query(kit, query)

            # Enrich trace with metadata
            trace["case_id"] = case_id
            trace["chat_profile"] = profile
            trace["prompt_logical_profile"] = kit.logical_profile
            trace["prompt_manifest"] = kit.prompt_manifest
            trace["tool_names_available"] = kit.tool_names
            trace["model"] = deployment
            trace["temperature"] = temperature
            trace["top_p"] = top_p
            trace["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

            out.write(json.dumps(trace, ensure_ascii=False) + "\n")

            if trace.get("error"):
                failed += 1
            else:
                succeeded += 1

            tool_names_used = [e["name"] for e in trace.get("tool_events", []) if e.get("name")]
            response_preview = (trace.get("output") or "")[:100]
            logger.info(
                "    -> tools=%s  response=%s%s",
                tool_names_used or "(none)",
                response_preview,
                "..." if len(trace.get("output", "")) > 100 else "",
            )

            # Rate-limit courtesy delay between queries
            if idx < len(gold_rows):
                await asyncio.sleep(inter_query_delay)

    logger.info(
        "Done. %d succeeded, %d failed. Traces written to %s",
        succeeded, failed, output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate eval traces by running the agent against gold queries.",
    )
    parser.add_argument(
        "--gold",
        default="eval/datasets/sql_agent_gold_starter.jsonl",
        help="Path to gold dataset JSONL.",
    )
    parser.add_argument(
        "--out",
        default="eval/traces/agent_runs.jsonl",
        help="Output traces JSONL path.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Only run queries for this chat profile (exact name match).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between queries (rate-limit courtesy).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override AZURE_OPENAI_MODEL deployment name (e.g. gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0.0-2.0). Omit to use model default.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (0.0-1.0). Omit to use model default.",
    )
    args = parser.parse_args()

    asyncio.run(generate(
        gold_path=Path(args.gold),
        output_path=Path(args.out),
        profile_filter=args.profile,
        inter_query_delay=args.delay,
        model_override=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
    ))


if __name__ == "__main__":
    main()
