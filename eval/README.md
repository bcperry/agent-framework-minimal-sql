# Evaluation Pipeline — End-to-End Guide

This guide walks you through the full evaluate → iterate → track loop.

---

## Overview

The pipeline has **two steps** you run in sequence:

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `eval/generate_traces.py` | Runs the real agent against gold queries → writes traces JSONL |
| 2 | `eval/foundry/run_eval.py` | Scores those traces (deterministic + AI judge) → uploads to Foundry portal |

Each run in the portal gets a unique name and URL, so you can compare across prompt/model/parameter changes over time.

---

## Prerequisites

Your `.env` must have these set:

```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_MODEL=gpt-4o
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_SQL_CONNECTIONSTRING=...
AZURE_AI_PROJECT_CONNECTION_STRING=...        # from Foundry Overview → "Project connection string"
```

Optionally set `AZURE_OPENAI_EVAL_DEPLOYMENT` if you want the AI judge evaluators to use a different model than the agent.

---

## Step 1 — Generate Traces

This runs the **actual agent** (same tools, same system prompt, same YAML config) against every query in the gold dataset.

### Basic run (uses `.env` defaults)

```powershell
uv run python eval/generate_traces.py --delay 3
```

### Override model deployment

```powershell
uv run python eval/generate_traces.py --model gpt-4o-mini --delay 3
```

### Override sampling parameters

```powershell
uv run python eval/generate_traces.py --temperature 0.1 --top-p 0.95 --delay 3
```

### Combine everything

```powershell
uv run python eval/generate_traces.py --model gpt-4o-mini --temperature 0.0 --top-p 1.0 --delay 3
```

### Run only one profile

```powershell
uv run python eval/generate_traces.py --profile "Tactical Readiness AI" --delay 3
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--gold` | `eval/datasets/sql_agent_gold_starter.jsonl` | Gold dataset path |
| `--out` | `eval/traces/agent_runs.jsonl` | Output traces path |
| `--profile` | *(all)* | Only run queries for this chat profile |
| `--delay` | `2.0` | Seconds between queries (rate-limit courtesy) |
| `--model` | `$AZURE_OPENAI_MODEL` | Override deployment name |
| `--temperature` | *(model default)* | Sampling temperature (0.0–2.0) |
| `--top-p` | *(model default)* | Nucleus sampling cutoff (0.0–1.0) |

**Output:**
- `eval/traces/agent_runs.jsonl` (legacy/latest path kept for compatibility)
- `eval/results/traces-<timestamp>__model-...__temp-...__top-p-.../agent_runs.jsonl`
- `eval/results/traces-<...>/run_metadata.json` (model, temp, top_p, prompt versions, timestamp)

---

## Step 2 — Run Evaluation & Upload to Portal

This scores every trace against the gold labels and uploads results to Azure AI Foundry.

```powershell
uv run python eval/foundry/run_eval.py --config eval/foundry/config.yaml
```

By default, the runner now auto-selects the newest run-scoped traces file from
`eval/results/traces-*/agent_runs.jsonl` when `paths.traces_jsonl` is left at its
default value. If none exist, it falls back to `eval/traces/agent_runs.jsonl`.

**What happens:**

1. **Deterministic metrics** (local, instant):
   - Required tools pass rate — did the agent call the must-have tools?
   - Forbidden tools pass rate — did it avoid tools it shouldn't use?
   - Expected sequence pass rate — did tool calls follow the right order?
   - Tool F1 — precision/recall of actual vs. expected tools

2. **AI judge metrics** (calls your Azure OpenAI model):
   - Intent Resolution — did the agent understand what the user wanted?
   - Task Adherence — did the response actually answer the question?
   - Tool Call Accuracy — were tool calls appropriate and correct?

3. **Portal upload** — all metrics + per-row details are pushed to Foundry.

**Output:**
- `eval/results/foundry_eval_latest.json` (legacy/latest path)
- run-scoped eval artifacts are written into the same trace folder that provided `agent_runs.jsonl` (for example `eval/results/traces-<...>/`):
   - `foundry_eval_latest.json`
   - `foundry_portal_input.jsonl`
   - `foundry_portal_eval.json`
   - `eval_run_metadata.json` (model, temp, top_p, prompt versions, timestamp)
- Console prints a **studio_url** you can click to view in the portal

---

## How to Iterate

### Change a system prompt

1. Edit `config/prompts/system.yaml` — modify the `text` field under the profile you want to change
2. Bump the `version` (e.g. `1.0.0` → `1.1.0`)
3. Re-run Step 1 + Step 2

```powershell
uv run python eval/generate_traces.py --delay 3
uv run python eval/foundry/run_eval.py --config eval/foundry/config.yaml
```

### Change tool descriptions / usage rules

1. Edit `config/prompts/tools.yaml` — modify `description`, `usage_rules`, or `examples` for the tool
2. Bump the tool's `version`
3. Re-run Step 1 + Step 2

### Change the model

```powershell
# Try gpt-4o-mini instead of gpt-4o
uv run python eval/generate_traces.py --model gpt-4o-mini --delay 3
uv run python eval/foundry/run_eval.py --config eval/foundry/config.yaml
```

### Change temperature / sampling

```powershell
# More deterministic (lower temperature)
uv run python eval/generate_traces.py --temperature 0.0 --delay 3
uv run python eval/foundry/run_eval.py --config eval/foundry/config.yaml

# More creative (higher temperature)
uv run python eval/generate_traces.py --temperature 1.0 --delay 3
uv run python eval/foundry/run_eval.py --config eval/foundry/config.yaml
```

### Add new test cases

1. Add rows to `eval/datasets/sql_agent_gold_starter.jsonl` — one JSON line per test case
2. Each row needs:
   ```json
   {
     "case_id": "SQL-011",
     "chat_profile": "Tactical Readiness AI",
     "query": "Your test question here",
     "expected_tools": ["list_views", "describe_table", "read_query"],
     "required_tools": ["read_query"],
     "forbidden_tools": ["semantic_search"]
   }
   ```
3. Re-run Step 1 + Step 2

---

## Comparing Runs in the Portal

Every Step 2 execution creates a **new evaluation run** in Foundry with a unique timestamp name like `foundry-eval-20260223-211910`. To compare:

1. Open the **studio_url** printed in the console (or from `eval/results/foundry_eval_latest.json` → `portal_evaluation.studio_url`)
2. In the Foundry portal, navigate to **Build → Evaluation**
3. You'll see each run listed with its metrics — select multiple to compare side-by-side

### Tracking what changed

Each trace in `agent_runs.jsonl` records:
- `model` — which deployment was used
- `temperature` / `top_p` — sampling parameters
- `prompt_manifest` — which prompt versions were active (e.g. `tactical_readiness_system@1.0.0`)

This lets you correlate portal metrics with exactly what config produced them.

---

## Quick-Reference: Full Experiment Run

```powershell
# 1. Edit prompts/tools/model as needed, then:

# 2. Generate fresh traces
uv run python eval/generate_traces.py --delay 3

# 3. Evaluate and upload
uv run python eval/foundry/run_eval.py --config eval/foundry/config.yaml

# 4. Click the studio_url in the output to view results
```

---

## File Map

| File | Purpose |
|------|---------|
| `config/prompts/system.yaml` | System prompts per profile (edit to change agent behavior) |
| `config/prompts/tools.yaml` | Tool descriptions & usage rules (edit to change tool guidance) |
| `eval/datasets/sql_agent_gold_starter.jsonl` | Gold test cases with expected tools |
| `eval/generate_traces.py` | Step 1: runs agent against gold queries |
| `eval/foundry/run_eval.py` | Step 2: scores traces + uploads to portal |
| `eval/foundry/config.yaml` | Eval runner config (paths, metrics toggles, Azure config) |
| `eval/traces/agent_runs.jsonl` | Generated traces (output of Step 1, input to Step 2) |
| `eval/results/foundry_eval_latest.json` | Full local results (output of Step 2) |
