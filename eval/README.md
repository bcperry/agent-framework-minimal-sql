# Evaluation Pipeline — End-to-End Guide

This guide walks you through the full evaluate → iterate → track loop using the **single all-in-one command**.

---

## Overview

Use one script:

| Script | What it does |
|--------|---------------|
| `eval/generate_and_eval.py` | Runs agent traces, scores them (deterministic + AI judge), and uploads to Foundry |

Each run gets a unique run folder plus a Foundry evaluation URL so you can compare prompt/model/parameter changes over time.

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

Optionally set `AZURE_OPENAI_EVAL_DEPLOYMENT` if you want AI judges to use a different model than the agent.

---

## Run Everything (Single Command)

### Basic run (uses `.env` defaults)

```powershell
uv run python eval/generate_and_eval.py --delay 3
```

### Override model deployment

```powershell
uv run python eval/generate_and_eval.py --model gpt-4o-mini --delay 3
```

### Override sampling parameters

```powershell
uv run python eval/generate_and_eval.py --temperature 0.1 --top-p 0.95 --delay 3
```

### Combine everything

```powershell
uv run python eval/generate_and_eval.py --model gpt-4o-mini --temperature 0.0 --top-p 1.0 --delay 3
```

### Run only one profile

```powershell
uv run python eval/generate_and_eval.py --profile "Tactical Readiness AI" --delay 3
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--gold` | `eval/datasets/sql_agent_gold_starter.jsonl` | Gold dataset path |
| `--out` | `eval/traces/agent_runs.jsonl` | Output traces path (legacy/latest compatibility path) |
| `--profile` | *(all)* | Only run queries for this chat profile |
| `--delay` | `2.0` | Seconds between queries (rate-limit courtesy) |
| `--model` | `$AZURE_OPENAI_MODEL` | Override deployment name |
| `--temperature` | *(model default)* | Sampling temperature (0.0–2.0) |
| `--top-p` | *(model default)* | Nucleus sampling cutoff (0.0–1.0) |

---

## What the Command Produces

The all-in-one command performs:

1. Trace generation against your gold dataset
2. Deterministic scoring:
   - Required tools pass rate
   - Forbidden tools pass rate
   - Expected sequence pass rate
   - Tool F1
3. AI judge scoring:
   - Intent Resolution
   - Task Adherence
   - Tool Call Accuracy
4. Foundry upload with `studio_url` in output

**Output artifacts:**
- `eval/traces/agent_runs.jsonl` (legacy/latest path)
- `eval/results/traces-<timestamp>__model-...__temp-...__top-p-.../agent_runs.jsonl`
- `eval/results/traces-<...>/run_metadata.json`
- `eval/results/traces-<...>/foundry_eval_latest.json`
- `eval/results/traces-<...>/foundry_portal_input.jsonl`
- `eval/results/traces-<...>/foundry_portal_eval.json`
- `eval/results/traces-<...>/eval_run_metadata.json`
- `eval/results/foundry_eval_latest.json` (legacy/latest summary path)

---

## How to Iterate

### Change a system prompt

1. Edit `config/prompts/system.yaml`
2. Bump the profile `version`
3. Re-run all-in-one command

```powershell
uv run python eval/generate_and_eval.py --delay 3
```

### Change tool guidance

1. Edit `config/prompts/tools.yaml` (`description`, `usage_rules`, `examples`)
2. Bump tool `version`
3. Re-run all-in-one command

### Change model

```powershell
uv run python eval/generate_and_eval.py --model gpt-4o-mini --delay 3
```

### Change temperature / sampling

```powershell
uv run python eval/generate_and_eval.py --temperature 0.0 --delay 3
uv run python eval/generate_and_eval.py --temperature 1.0 --delay 3
```

### Add new test cases

1. Add rows to `eval/datasets/sql_agent_gold_starter.jsonl`
2. Re-run all-in-one command

Example row:

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

---

## Comparing Runs in the Portal

Every all-in-one execution creates a new Foundry evaluation run.

1. Open the printed `studio_url` (or read it from `foundry_eval_latest.json`)
2. In Foundry portal, go to **Build → Evaluation**
3. Compare runs side-by-side

Each trace row captures model/sampling/prompt manifest so you can attribute metric changes to config changes.

---

## Quick Reference

```powershell
# Edit prompts/tools/model as needed, then run everything:
uv run python eval/generate_and_eval.py --delay 3

# Open the printed studio_url to review results
```

---

## File Map

| File | Purpose |
|------|---------|
| `config/prompts/system.yaml` | System prompts per profile |
| `config/prompts/tools.yaml` | Tool descriptions & usage rules |
| `eval/datasets/sql_agent_gold_starter.jsonl` | Gold test cases |
| `eval/generate_and_eval.py` | All-in-one runner (generate traces + eval + portal upload) |
| `eval/foundry/config.yaml` | Eval config used by all-in-one runner |
| `eval/results/traces-*/` | Run-scoped artifacts and metadata |
| `eval/results/foundry_eval_latest.json` | Latest summary JSON |
