"""Validation tests for prompt tool configuration schema."""

from pathlib import Path

import yaml


TOOLS_YAML_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts" / "tools.yaml"
REQUIRED_TOOL_FIELDS = {"tool_name", "version", "enabled_profiles", "description", "usage_rules", "examples"}
ALLOWED_PROFILES = {"sql", "search", "hybrid"}


def _load_tools_yaml() -> dict:
    with TOOLS_YAML_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise AssertionError("tools.yaml must parse to a mapping")
    return data


def test_tools_yaml_exists() -> None:
    assert TOOLS_YAML_PATH.exists(), f"Expected tools config at {TOOLS_YAML_PATH}"


def test_tools_yaml_top_level_shape() -> None:
    payload = _load_tools_yaml()
    assert isinstance(payload.get("schema_version"), int), "schema_version must be an integer"
    assert isinstance(payload.get("description"), str) and payload["description"].strip(), "description must be non-empty"
    assert isinstance(payload.get("tools"), list) and payload["tools"], "tools must be a non-empty list"


def test_each_tool_has_required_fields_and_types() -> None:
    payload = _load_tools_yaml()
    tools = payload["tools"]

    tool_names: list[str] = []
    for index, item in enumerate(tools):
        assert isinstance(item, dict), f"tools[{index}] must be an object"
        missing = sorted(REQUIRED_TOOL_FIELDS - set(item.keys()))
        assert not missing, f"tools[{index}] is missing required fields: {', '.join(missing)}"

        tool_name = item["tool_name"]
        assert isinstance(tool_name, str) and tool_name.strip(), f"tools[{index}].tool_name must be non-empty"
        tool_names.append(tool_name)

        version = item["version"]
        assert isinstance(version, str) and version.strip(), f"tools[{index}].version must be non-empty"

        description = item["description"]
        assert isinstance(description, str) and description.strip(), f"tools[{index}].description must be non-empty"

        enabled_profiles = item["enabled_profiles"]
        assert isinstance(enabled_profiles, list) and enabled_profiles, f"tools[{index}].enabled_profiles must be non-empty list"
        for profile in enabled_profiles:
            assert profile in ALLOWED_PROFILES, (
                f"tools[{index}].enabled_profiles contains unsupported profile '{profile}'"
            )

        usage_rules = item["usage_rules"]
        assert isinstance(usage_rules, list) and usage_rules, f"tools[{index}].usage_rules must be non-empty list"
        assert all(isinstance(rule, str) and rule.strip() for rule in usage_rules), (
            f"tools[{index}].usage_rules entries must be non-empty strings"
        )

        examples = item["examples"]
        assert isinstance(examples, list) and examples, f"tools[{index}].examples must be non-empty list"
        for example_index, example in enumerate(examples):
            assert isinstance(example, dict), f"tools[{index}].examples[{example_index}] must be an object"
            assert isinstance(example.get("user_prompt"), str) and example["user_prompt"].strip(), (
                f"tools[{index}].examples[{example_index}].user_prompt must be non-empty"
            )
            assert isinstance(example.get("expected"), str) and example["expected"].strip(), (
                f"tools[{index}].examples[{example_index}].expected must be non-empty"
            )

    assert len(tool_names) == len(set(tool_names)), "tool_name values must be unique"
