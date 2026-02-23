# Foundry Eval Runner

This scaffold evaluates agent traces against your gold dataset and writes a result report to `eval/results/`.

## What it does

- Reads trace rows from `eval/traces/agent_runs.jsonl`
- Matches each trace query to gold rows in `eval/datasets/sql_agent_gold_starter.jsonl`
- Computes deterministic tool-call metrics:
  - required/forbidden tool pass
  - expected sequence pass
  - precision/recall/F1 against expected tool set
- Optionally runs Azure AI evaluator judges (when model credentials are set):
  - `IntentResolutionEvaluator`
  - `ResponseCompletenessEvaluator`
  - `TaskAdherenceEvaluator`
  - `ToolCallAccuracyEvaluator`

## Files

- Config: `eval/foundry/config.yaml`
- Runner: `eval/foundry/run_eval.py`
- Output: `eval/results/foundry_eval_latest.json`

## Run

```powershell
uv run python eval/foundry/run_eval.py --config eval/foundry/config.yaml
```

The runner loads environment variables from the workspace `.env` file.

## Required input files

- `eval/traces/agent_runs.jsonl` (from runtime trace logging)
- `eval/datasets/sql_agent_gold_starter.jsonl` (gold labels)
- `config/prompts/tools.yaml` (tool definitions source of truth)

## Azure AI judge metrics (optional)

Set these env vars to enable AI-assisted evaluator metrics:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_EVAL_DEPLOYMENT` (or fallback `AZURE_OPENAI_MODEL`)

If these are not present, AI judge metrics are skipped and deterministic metrics still run.

## Foundry portal logging

To log the evaluation in the Foundry portal (and get a `studio_url` in output), set exactly one of:

- `AZURE_AI_PROJECT_ENDPOINT` (preferred)
- `AZURE_AI_PROJECT_CONNECTION_STRING` (fallback when endpoint is not exposed in your UI)

When enabled (`log_to_foundry_portal: true` in config), the runner submits a portal-backed
evaluation and writes:

- `eval/results/foundry_portal_input.jsonl`
- `eval/results/foundry_portal_eval.json`

The consolidated local output `eval/results/foundry_eval_latest.json` includes a
`portal_evaluation` section with status, metrics summary, and `studio_url` when available.

## Notes

- Query matching is exact case-insensitive by default.
- Unmatched traces are listed in output under `unmatched_trace_queries`.
- This scaffold is intended to be CI-friendly and easy to extend for Foundry cloud eval APIs later.
