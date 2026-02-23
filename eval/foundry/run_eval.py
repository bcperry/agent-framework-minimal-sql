import argparse
import contextlib
import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


load_dotenv()


@dataclass
class EvalRow:
    case_id: str
    query: str
    response: str
    actual_tool_calls: list[dict[str, Any]]
    expected_tools: list[str]
    required_tools: list[str]
    forbidden_tools: list[str]
    tool_definitions: list[dict[str, Any]]
    trace_metadata: dict[str, Any] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(item)
    return rows


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float))]


def _try_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _ordered_subsequence(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    position = 0
    for name in actual:
        if position < len(expected) and name == expected[position]:
            position += 1
    return position == len(expected)


def _precision_recall_f1(expected: list[str], actual: list[str]) -> tuple[float, float, float]:
    expected_set = set(expected)
    actual_set = set(actual)

    if not expected_set and not actual_set:
        return 1.0, 1.0, 1.0
    if not actual_set:
        return 0.0, 0.0, 0.0

    true_positive = len(expected_set & actual_set)
    precision = true_positive / len(actual_set)
    recall = true_positive / len(expected_set) if expected_set else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _build_tool_definitions(tool_yaml: dict[str, Any]) -> list[dict[str, Any]]:
    tools = tool_yaml.get("tools") or []
    definitions: list[dict[str, Any]] = []

    for item in tools:
        if not isinstance(item, dict):
            continue
        name = item.get("tool_name")
        description = item.get("description")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(description, str):
            description = ""

        definitions.append(
            {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            }
        )

    return definitions


def _build_eval_rows(
    traces: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
    skip_unmatched: bool,
) -> tuple[list[EvalRow], list[str]]:
    gold_index: dict[str, dict[str, Any]] = {}
    for row in gold_rows:
        query = row.get("query")
        if isinstance(query, str) and query.strip():
            gold_index[_norm(query)] = row

    unmatched: list[str] = []
    eval_rows: list[EvalRow] = []

    for trace in traces:
        query = trace.get("input")
        if not isinstance(query, str) or not query.strip():
            continue

        gold = gold_index.get(_norm(query))
        if gold is None:
            unmatched.append(query)
            if skip_unmatched:
                continue
            gold = {}

        events = trace.get("tool_events") or []
        tool_calls: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            name = event.get("name")
            if not isinstance(name, str) or not name:
                continue
            call_id = event.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"call_{index}"

            parsed_args = _try_parse_json(event.get("arguments"))
            arguments = parsed_args if isinstance(parsed_args, dict) else {"raw": str(parsed_args)}

            tool_calls.append(
                {
                    "type": "tool_call",
                    "tool_call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )

        # Carry trace-level metadata through for portal tagging
        trace_meta = {
            "model": trace.get("model") or "",
            "temperature": trace.get("temperature"),
            "top_p": trace.get("top_p"),
            "chat_profile": trace.get("chat_profile") or "",
            "prompt_manifest": trace.get("prompt_manifest") or {},
        }

        eval_rows.append(
            EvalRow(
                case_id=str(gold.get("case_id") or f"trace_{len(eval_rows)+1}"),
                query=query,
                response=str(trace.get("output") or ""),
                actual_tool_calls=tool_calls,
                expected_tools=_as_list_of_str(gold.get("expected_tools")),
                required_tools=_as_list_of_str(gold.get("required_tools")),
                forbidden_tools=_as_list_of_str(gold.get("forbidden_tools")),
                tool_definitions=tool_definitions,
                trace_metadata=trace_meta,
            )
        )

    return eval_rows, unmatched


def _build_model_config(config: dict[str, Any]) -> dict[str, str] | None:
    azure_cfg = _as_dict(config.get("azure"))
    model_cfg = _as_dict(azure_cfg.get("model"))

    endpoint = os.getenv(str(model_cfg.get("endpoint_env") or "AZURE_OPENAI_ENDPOINT"))
    api_key = os.getenv(str(model_cfg.get("key_env") or "AZURE_OPENAI_API_KEY"))
    api_version = os.getenv(str(model_cfg.get("api_version_env") or "AZURE_OPENAI_API_VERSION"))

    deployment_env = str(model_cfg.get("deployment_env") or "AZURE_OPENAI_EVAL_DEPLOYMENT")
    fallback_deployment_env = str(model_cfg.get("deployment_fallback_env") or "AZURE_OPENAI_MODEL")
    deployment = os.getenv(deployment_env) or os.getenv(fallback_deployment_env)

    required = [endpoint, api_key, api_version, deployment]
    if any(not value for value in required):
        return None

    return {
        "azure_endpoint": endpoint or "",
        "api_key": api_key or "",
        "api_version": api_version or "",
        "azure_deployment": deployment or "",
    }


def _resolve_project(config: dict[str, Any]) -> str | dict[str, str] | None:
    """Return a project identifier for the evaluate() azure_ai_project param.

    Tries (in order):
      1. AZURE_AI_PROJECT_ENDPOINT  -> string  (Foundry-native projects)
      2. AZURE_AI_PROJECT_CONNECTION_STRING -> AzureAIProject dict
         (hub-based / GCC High where the endpoint audience is unavailable)
    """
    azure_cfg = _as_dict(config.get("azure"))
    project_endpoint_env = str(azure_cfg.get("project_endpoint_env") or "AZURE_AI_PROJECT_ENDPOINT")
    project_connection_env = str(
        azure_cfg.get("project_connection_string_env") or "AZURE_AI_PROJECT_CONNECTION_STRING"
    )

    # Path 1 – direct endpoint string (commercial / Foundry-native)
    project_endpoint = os.getenv(project_endpoint_env)
    if project_endpoint and project_endpoint.strip():
        return project_endpoint.strip()

    # Path 2 – connection string -> AzureAIProject dict
    #   Format: "<host>;<subscription_id>;<resource_group>;<project_name>"
    #   The SDK authenticates via ml.azure.us for this form, which works in GCC High.
    project_connection_string = os.getenv(project_connection_env)
    if project_connection_string and project_connection_string.strip():
        return _parse_connection_string(project_connection_string)

    return None


def _parse_connection_string(connection_string: str) -> dict[str, str] | None:
    """Parse a Foundry project connection string into an AzureAIProject dict.

    Accepted formats:
      - Semicolon-delimited: host;subscription_id;resource_group;project_name
      - Key=Value pairs:    endpoint=...;subscription_id=...;resource_group=...;project=...
    """
    raw = connection_string.strip()
    if not raw:
        return None

    # Try key=value format first (some portal versions use this)
    if "=" in raw:
        kv: dict[str, str] = {}
        for segment in raw.split(";"):
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            kv[key.strip().lower().replace(" ", "_")] = value.strip()

        sub = kv.get("subscription_id") or kv.get("subscription")
        rg = kv.get("resource_group") or kv.get("resource_group_name")
        proj = kv.get("project") or kv.get("project_name")
        if sub and rg and proj:
            return {"subscription_id": sub, "resource_group_name": rg, "project_name": proj}

    # Positional format: host;subscription_id;resource_group;project_name
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if len(parts) >= 4:
        return {
            "subscription_id": parts[1],
            "resource_group_name": parts[2],
            "project_name": parts[3],
        }

    return None


def _is_gcc_high(config: dict[str, Any]) -> bool:
    """Return True when the connection string points to a GCC High (.azure.us) host."""
    azure_cfg = _as_dict(config.get("azure"))
    env_name = str(azure_cfg.get("project_connection_string_env") or "AZURE_AI_PROJECT_CONNECTION_STRING")
    cs = os.getenv(env_name, "")
    if not cs:
        return False
    host = cs.split(";")[0].strip().lower()
    return ".azure.us" in host


@contextlib.contextmanager
def _gcc_high_patches(config: dict[str, Any]):
    """Monkey-patch azure-ai-evaluation SDK internals for GCC High.

    The SDK v1.15.x hardcodes ``management.azure.com`` for ARM calls and
    ``https://management.azure.com/.default`` as the token scope.  GCC High
    requires ``management.usgovcloudapi.net`` instead.

    Also sets ``AZURE_AUTHORITY_HOST`` so ``DefaultAzureCredential`` targets the
    government AAD endpoint.
    """
    from azure.ai.evaluation._evaluate._utils import LiteMLClient  # type: ignore[import-untyped]

    GOV_ARM = "management.usgovcloudapi.net"
    GOV_SCOPE = f"https://{GOV_ARM}/.default"
    GOV_AUTHORITY = "https://login.microsoftonline.us"

    # --- ARM base URL patch ---
    orig_init = LiteMLClient.__init__

    def _patched_init(self, subscription_id, resource_group, logger, credential=None, **kwargs):
        orig_init(self, subscription_id, resource_group, logger, credential=credential, **kwargs)
        self._base_url = self._base_url.replace("management.azure.com", GOV_ARM)

    # --- Token scope patch ---
    orig_get_tm = LiteMLClient._get_token_manager

    def _patched_get_token_manager(self):
        if self._token_manager is None:
            with self._lock:
                if self._token_manager is None:
                    from azure.ai.evaluation._azure._token_manager import AzureMLTokenManager  # type: ignore
                    self._token_manager = AzureMLTokenManager(
                        GOV_SCOPE, self._logger, credential=self._credential
                    )
                    self._credential = self._token_manager.credential
        return self._token_manager

    # --- Apply patches ---
    prev_authority = os.environ.get("AZURE_AUTHORITY_HOST")
    os.environ["AZURE_AUTHORITY_HOST"] = GOV_AUTHORITY
    LiteMLClient.__init__ = _patched_init  # type: ignore[assignment]
    LiteMLClient._get_token_manager = _patched_get_token_manager  # type: ignore[assignment]
    try:
        yield
    finally:
        LiteMLClient.__init__ = orig_init  # type: ignore[assignment]
        LiteMLClient._get_token_manager = orig_get_tm  # type: ignore[assignment]
        if prev_authority is None:
            os.environ.pop("AZURE_AUTHORITY_HOST", None)
        else:
            os.environ["AZURE_AUTHORITY_HOST"] = prev_authority


def _extract_run_metadata(rows: list[EvalRow]) -> dict[str, Any]:
    """Extract run-level metadata from the first trace that has it.

    All traces in a single generate_traces run share the same model/temperature/top_p,
    so the first row's metadata is representative of the whole run.
    Prompt versions are merged across all rows (profiles may differ).
    """
    meta: dict[str, Any] = {"model": "", "temperature": None, "top_p": None, "prompt_versions": {}}
    for row in rows:
        tm = row.trace_metadata or {}
        if not meta["model"] and tm.get("model"):
            meta["model"] = tm["model"]
        if meta["temperature"] is None and tm.get("temperature") is not None:
            meta["temperature"] = tm["temperature"]
        if meta["top_p"] is None and tm.get("top_p") is not None:
            meta["top_p"] = tm["top_p"]
        manifest = tm.get("prompt_manifest") or {}
        for k, v in manifest.items():
            if k not in meta["prompt_versions"]:
                meta["prompt_versions"][k] = v
    return meta


def _write_portal_input_jsonl(rows: list[EvalRow], portal_input_path: Path) -> Path:
    portal_input_path.parent.mkdir(parents=True, exist_ok=True)
    with portal_input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            meta = row.trace_metadata or {}
            manifest = meta.get("prompt_manifest") or {}
            # Flatten prompt versions into a readable string  e.g. "system=tactical_readiness_system@1.0.0"
            prompt_versions = "; ".join(
                f"{k}={v}" for k, v in manifest.items()
            ) if manifest else ""

            item = {
                "case_id": row.case_id,
                "query": row.query,
                "response": row.response,
                "tool_calls": row.actual_tool_calls,
                "tool_definitions": row.tool_definitions,
                # ── Config metadata (visible in portal Data tab) ──
                "model": meta.get("model", ""),
                "temperature": meta.get("temperature"),
                "top_p": meta.get("top_p"),
                "chat_profile": meta.get("chat_profile", ""),
                "prompt_versions": prompt_versions,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return portal_input_path


def _run_portal_logged_evaluation(
    rows: list[EvalRow],
    config: dict[str, Any],
    model_config: dict[str, str] | None,
) -> dict[str, Any]:
    if not rows:
        return {"enabled": False, "status": "skipped", "reason": "no_eval_rows"}

    if not model_config:
        return {"enabled": False, "status": "skipped", "reason": "missing_model_config"}

    project = _resolve_project(config)
    if not project:
        return {"enabled": False, "status": "skipped", "reason": "missing_project_scope"}

    paths = _as_dict(config.get("paths"))
    portal_input_path = Path(str(paths.get("portal_input_jsonl") or "eval/results/foundry_portal_input.jsonl"))
    portal_output_path = Path(str(paths.get("portal_output_json") or "eval/results/foundry_portal_eval.json"))

    _write_portal_input_jsonl(rows, portal_input_path)

    # Serialisable representation for output JSON
    project_repr = project if isinstance(project, str) else json.dumps(project)

    try:
        evaluation_module = importlib.import_module("azure.ai.evaluation")
        evaluate_fn = getattr(evaluation_module, "evaluate")

        evaluators: dict[str, Any] = {
            "intent_resolution": evaluation_module.IntentResolutionEvaluator(model_config),
            "task_adherence": evaluation_module.TaskAdherenceEvaluator(model_config),
            "tool_call_accuracy": evaluation_module.ToolCallAccuracyEvaluator(model_config),
        }

        evaluator_config: dict[str, Any] = {
            "intent_resolution": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "task_adherence": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "tool_call_accuracy": {
                "column_mapping": {
                    "query": "${data.query}",
                    "tool_calls": "${data.tool_calls}",
                    "tool_definitions": "${data.tool_definitions}",
                }
            },
        }

        # NOTE: ResponseCompletenessEvaluator requires 'ground_truth' + 'response'
        # (not 'query' + 'response').  Our gold dataset does not include ground-truth
        # expected answers, so this evaluator is intentionally omitted.

        # ── Build run-level metadata from traces for tags + naming ──
        run_meta = _extract_run_metadata(rows)
        model_name = run_meta.get("model", "unknown")
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        evaluation_name = f"{model_name}-eval-{timestamp}"

        tags: dict[str, str] = {
            "model": model_name,
        }
        if run_meta.get("temperature") is not None:
            tags["temperature"] = str(run_meta["temperature"])
        if run_meta.get("top_p") is not None:
            tags["top_p"] = str(run_meta["top_p"])
        for key, value in run_meta.get("prompt_versions", {}).items():
            tags[f"prompt.{key}"] = str(value)

        # Detect GCC High from connection string and apply sovereign-cloud patches
        gcc_ctx = _gcc_high_patches(config) if _is_gcc_high(config) else contextlib.nullcontext()
        with gcc_ctx:
            result = evaluate_fn(
                data=str(portal_input_path),
                evaluation_name=evaluation_name,
                evaluators=evaluators,
                evaluator_config=evaluator_config,
                azure_ai_project=project,
                output_path=str(portal_output_path),
                tags=tags,
            )

        return {
            "enabled": True,
            "status": "completed",
            "evaluation_name": evaluation_name,
            "project": project_repr,
            "studio_url": result.get("studio_url") if isinstance(result, dict) else None,
            "metrics": result.get("metrics") if isinstance(result, dict) else None,
            "portal_output_json": str(portal_output_path),
            "portal_input_jsonl": str(portal_input_path),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "status": "error",
            "project": project_repr,
            "portal_output_json": str(portal_output_path),
            "portal_input_jsonl": str(portal_input_path),
            "error": str(exc),
        }


def _run_deterministic_metrics(row: EvalRow) -> dict[str, Any]:
    actual_tool_names = [call.get("name", "") for call in row.actual_tool_calls if isinstance(call.get("name"), str)]

    required_missing = [name for name in row.required_tools if name not in actual_tool_names]
    forbidden_hit = [name for name in row.forbidden_tools if name in actual_tool_names]

    precision, recall, f1 = _precision_recall_f1(row.expected_tools, actual_tool_names)
    sequence_ok = _ordered_subsequence(row.expected_tools, actual_tool_names)

    return {
        "actual_tools": actual_tool_names,
        "required_missing": required_missing,
        "forbidden_hit": forbidden_hit,
        "required_tools_pass": len(required_missing) == 0,
        "forbidden_tools_pass": len(forbidden_hit) == 0,
        "expected_sequence_pass": sequence_ok,
        "tool_precision": round(precision, 4),
        "tool_recall": round(recall, 4),
        "tool_f1": round(f1, 4),
    }


def _run_ai_judge_metrics(rows: list[EvalRow], model_config: dict[str, str] | None) -> list[dict[str, Any]]:
    if not model_config:
        return [{"ai_judge_skipped": True, "reason": "missing_model_config"} for _ in rows]

    evaluation_module = importlib.import_module("azure.ai.evaluation")
    intent_eval = evaluation_module.IntentResolutionEvaluator(model_config)
    task_eval = evaluation_module.TaskAdherenceEvaluator(model_config)
    tool_eval = evaluation_module.ToolCallAccuracyEvaluator(model_config)
    # NOTE: ResponseCompletenessEvaluator omitted — requires ground_truth labels.

    results: list[dict[str, Any]] = []
    for row in rows:
        row_result: dict[str, Any] = {}

        try:
            row_result["intent_resolution"] = intent_eval(query=row.query, response=row.response)
        except Exception as exc:
            row_result["intent_resolution_error"] = str(exc)

        try:
            row_result["task_adherence"] = task_eval(query=row.query, response=row.response)
        except Exception as exc:
            row_result["task_adherence_error"] = str(exc)

        try:
            row_result["tool_call_accuracy"] = tool_eval(
                query=row.query,
                tool_calls=row.actual_tool_calls,
                tool_definitions=row.tool_definitions,
            )
        except Exception as exc:
            row_result["tool_call_accuracy_error"] = str(exc)

        results.append(row_result)

    return results


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "required_tools_pass_rate": 0.0,
            "forbidden_tools_pass_rate": 0.0,
            "expected_sequence_pass_rate": 0.0,
            "avg_tool_f1": 0.0,
        }

    required_pass = sum(1 for row in rows if row.get("required_tools_pass") is True)
    forbidden_pass = sum(1 for row in rows if row.get("forbidden_tools_pass") is True)
    sequence_pass = sum(1 for row in rows if row.get("expected_sequence_pass") is True)

    avg_tool_f1 = sum(float(row.get("tool_f1") or 0.0) for row in rows) / len(rows)

    return {
        "row_count": len(rows),
        "required_tools_pass_rate": round(required_pass / len(rows), 4),
        "forbidden_tools_pass_rate": round(forbidden_pass / len(rows), 4),
        "expected_sequence_pass_rate": round(sequence_pass / len(rows), 4),
        "avg_tool_f1": round(avg_tool_f1, 4),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)

    paths = _as_dict(config.get("paths"))
    traces_path = Path(str(paths.get("traces_jsonl") or "eval/traces/agent_runs.jsonl"))
    gold_path = Path(str(paths.get("gold_jsonl") or "eval/datasets/sql_agent_gold_starter.jsonl"))
    tools_yaml_path = Path(str(paths.get("tool_config_yaml") or "config/prompts/tools.yaml"))
    output_path = Path(str(paths.get("output_json") or "eval/results/foundry_eval_latest.json"))

    matching = _as_dict(config.get("matching"))
    skip_unmatched = bool(matching.get("skip_unmatched_traces", True))

    metrics = _as_dict(config.get("metrics"))
    include_ai_judge = bool(metrics.get("include_ai_judge_metrics", True))
    include_deterministic = bool(metrics.get("include_deterministic_metrics", True))
    include_portal_logging = bool(metrics.get("log_to_foundry_portal", True))

    traces = _load_jsonl(traces_path)
    gold_rows = _load_jsonl(gold_path)
    tools_yaml = _load_yaml(tools_yaml_path)

    tool_definitions = _build_tool_definitions(tools_yaml)
    eval_rows, unmatched = _build_eval_rows(traces, gold_rows, tool_definitions, skip_unmatched)

    deterministic_rows: list[dict[str, Any]] = []
    if include_deterministic:
        deterministic_rows = [_run_deterministic_metrics(row) for row in eval_rows]

    ai_rows: list[dict[str, Any]] = []
    model_config: dict[str, str] | None = None
    if include_ai_judge:
        model_config = _build_model_config(config)
        ai_rows = _run_ai_judge_metrics(eval_rows, model_config)

    portal_eval: dict[str, Any] = {"enabled": False, "status": "skipped", "reason": "disabled"}
    if include_portal_logging:
        if model_config is None:
            model_config = _build_model_config(config)
        portal_eval = _run_portal_logged_evaluation(eval_rows, config, model_config)

    row_results: list[dict[str, Any]] = []
    for index, row in enumerate(eval_rows):
        payload: dict[str, Any] = {
            "case_id": row.case_id,
            "query": row.query,
            "response": row.response,
            "actual_tool_calls": row.actual_tool_calls,
            "expected_tools": row.expected_tools,
            "required_tools": row.required_tools,
            "forbidden_tools": row.forbidden_tools,
        }
        if deterministic_rows:
            payload["deterministic_metrics"] = deterministic_rows[index]
        if ai_rows:
            payload["ai_judge_metrics"] = ai_rows[index]
        row_results.append(payload)

    summary = _summarize(deterministic_rows) if deterministic_rows else {"row_count": len(row_results)}

    output = {
        "name": str(config.get("name") or "foundry-eval"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "traces_jsonl": str(traces_path),
            "gold_jsonl": str(gold_path),
            "tool_config_yaml": str(tools_yaml_path),
            "output_json": str(output_path),
        },
        "counts": {
            "trace_rows": len(traces),
            "gold_rows": len(gold_rows),
            "evaluated_rows": len(row_results),
            "unmatched_traces": len(unmatched),
        },
        "summary": summary,
        "portal_evaluation": portal_eval,
        "unmatched_trace_queries": unmatched,
        "rows": row_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Azure AI Foundry-oriented evaluation over trace and gold datasets.")
    parser.add_argument(
        "--config",
        default="eval/foundry/config.yaml",
        help="Path to runner config YAML.",
    )
    args = parser.parse_args()

    output = run(Path(args.config))
    print(json.dumps({"summary": output.get("summary"), "counts": output.get("counts")}, indent=2))
    print("Output written to:", _as_dict(output.get("paths")).get("output_json", "configured output path"))


if __name__ == "__main__":
    main()
