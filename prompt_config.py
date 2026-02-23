import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_PROMPT_ENV = "local"

_PROFILE_ALIAS = {
    "tactical readiness ai": "sql",
    "technical maintenance ai": "search",
    "combat logiguard ai": "hybrid",
}


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    logical_profile: str
    prompt_manifest: dict[str, str]
    tools_config: dict[str, Any]
    eval_config: dict[str, Any]


def _normalize_profile_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return " ".join(name.strip().lower().split())


def _resolve_logical_profile(chat_profile: Optional[str]) -> str:
    normalized = _normalize_profile_name(chat_profile)
    if normalized in _PROFILE_ALIAS:
        return _PROFILE_ALIAS[normalized]
    return "hybrid"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Prompt config file must contain a YAML object: {path}")
    return loaded


def _merge_manifest(
    system_doc: dict[str, Any],
    tools_doc: dict[str, Any],
    eval_doc: dict[str, Any],
    overlay_doc: dict[str, Any],
    logical_profile: str,
) -> dict[str, str]:
    overlay_manifest = overlay_doc.get("manifest", {}) if isinstance(overlay_doc, dict) else {}
    system_manifest = overlay_manifest.get("system", {}) if isinstance(overlay_manifest, dict) else {}
    system_item = system_manifest.get(logical_profile, {}) if isinstance(system_manifest, dict) else {}

    profile_entry = (system_doc.get("profiles") or {}).get(logical_profile, {})

    system_id = system_item.get("id") or profile_entry.get("id") or f"{logical_profile}_system"
    system_version = system_item.get("version") or profile_entry.get("version") or "0.0.0"

    tools_version = (
        (overlay_manifest.get("tools") or {}).get("version")
        if isinstance(overlay_manifest, dict)
        else None
    ) or "0.0.0"
    eval_version = (
        (overlay_manifest.get("eval") or {}).get("version")
        if isinstance(overlay_manifest, dict)
        else None
    ) or "0.0.0"

    if tools_version == "0.0.0":
        tool_items = tools_doc.get("tools") or []
        if tool_items and isinstance(tool_items, list):
            first_tool = tool_items[0] or {}
            tools_version = str(first_tool.get("version") or "0.0.0")

    if eval_version == "0.0.0":
        eval_templates = eval_doc.get("eval_templates") or {}
        if isinstance(eval_templates, dict):
            first_template = next(iter(eval_templates.values()), {})
            if isinstance(first_template, dict):
                eval_version = str(first_template.get("version") or "0.0.0")

    return {
        "system": f"{system_id}@{system_version}",
        "tools": str(tools_version),
        "eval": str(eval_version),
    }


def _validate_tools(
    tools_doc: dict[str, Any],
    available_tool_names: Optional[set[str]],
) -> None:
    configured_tools = tools_doc.get("tools") or []
    if not isinstance(configured_tools, list):
        raise ValueError("tools.yaml must define tools as a list")

    configured_names: list[str] = []
    for item in configured_tools:
        if not isinstance(item, dict):
            raise ValueError("Each tools.yaml item must be an object")
        tool_name = item.get("tool_name")
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError("Each tools.yaml item requires a string tool_name")
        configured_names.append(tool_name)

    if available_tool_names is None:
        return

    missing = sorted(set(configured_names) - set(available_tool_names))
    if missing:
        raise ValueError(
            "Prompt tool configuration references unregistered runtime tools: "
            + ", ".join(missing)
        )


def load_prompt_bundle(
    chat_profile: Optional[str],
    available_tool_names: Optional[set[str]] = None,
    prompt_env: Optional[str] = None,
    workspace_root: Optional[Path] = None,
) -> PromptBundle:
    root = workspace_root or Path(__file__).resolve().parent
    config_root = root / "config" / "prompts"

    active_env = (prompt_env or os.getenv("PROMPT_ENV") or DEFAULT_PROMPT_ENV).strip().lower()
    logical_profile = _resolve_logical_profile(chat_profile)

    system_doc = _read_yaml(config_root / "system.yaml")
    tools_doc = _read_yaml(config_root / "tools.yaml")
    eval_doc = _read_yaml(config_root / "eval.yaml")
    overlay_doc = _read_yaml(config_root / "overlays" / f"{active_env}.yaml")

    profiles = system_doc.get("profiles") or {}
    if logical_profile not in profiles:
        raise ValueError(f"No system prompt profile found for logical profile '{logical_profile}'")

    profile_entry = profiles[logical_profile]
    if not isinstance(profile_entry, dict) or not profile_entry.get("text"):
        raise ValueError(f"Invalid system prompt entry for logical profile '{logical_profile}'")

    _validate_tools(tools_doc, available_tool_names)

    manifest = _merge_manifest(system_doc, tools_doc, eval_doc, overlay_doc, logical_profile)

    return PromptBundle(
        system_prompt=str(profile_entry.get("text", "")).strip(),
        logical_profile=logical_profile,
        prompt_manifest=manifest,
        tools_config=tools_doc,
        eval_config=eval_doc,
    )
