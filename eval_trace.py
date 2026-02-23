import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _is_enabled(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class EvalTraceLogger:
    def __init__(
        self,
        enabled: bool,
        output_path: Path,
    ):
        self.enabled = enabled
        self.output_path = output_path

    @classmethod
    def from_env(cls, workspace_root: Optional[Path] = None) -> "EvalTraceLogger":
        root = workspace_root or Path(__file__).resolve().parent
        enabled = _is_enabled(os.getenv("ENABLE_EVAL_TRACE_LOGGING"))
        configured_path = os.getenv("EVAL_TRACE_OUTPUT_PATH", "eval/traces/agent_runs.jsonl")
        output_path = (root / configured_path).resolve()
        return cls(enabled=enabled, output_path=output_path)

    def log(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
