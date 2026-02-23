# SQL Agent Gold Dataset

This folder contains starter gold data for evaluating the SQL-focused agent in Azure AI Foundry.

## Files
- `sql_agent_gold_starter.csv`: Spreadsheet-friendly authoring format.
- `sql_agent_gold_starter.jsonl`: Foundry-friendly line-delimited format.

## Field Contract
- `case_id`: Unique test id.
- `chat_profile`: Expected profile context (`Tactical Readiness AI`, `Technical Maintenance AI`, or `hybrid`).
- `user_prompt`: User input under test.
- `history`: Optional prior-turn context (JSON string in CSV; object in JSONL).
- `intent_class`: Task taxonomy label.
- `expected_tools`: Ordered preferred tool trajectory.
- `required_tools`: Must appear in actual tool calls.
- `forbidden_tools`: Must not appear.
- `arg_expectations`: Soft argument constraints (filters, objects, etc.).
- `prompt_manifest`: Prompt config versions used when the row was authored.
- `notes`: Human notes for adjudication.

## Taxonomy (starter)
- `readiness_rollup`
- `lin_specific`
- `schema_discovery`
- `procedure_lookup`
- `mixed_sql_search`
- `no_tool_smalltalk`
