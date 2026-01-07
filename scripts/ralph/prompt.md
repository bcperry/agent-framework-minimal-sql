# Ralph Agent Instructions

## Your Task

1. Read `scripts/ralph/prd.json`
2. Read `scripts/ralph/progress.txt`
   (check Codebase Patterns first)
3. Check you're on the correct branch
4. Pick highest priority story 
   where `passes: false`
5. Implement that ONE story
6. Run linting, type checking, and tests
7. Update AGENTS.md files with learnings
8. Commit: `feat: [ID] - [Title]`
9. Update prd.json: `passes: true`
10. Append learnings to progress.txt

## Progress Format

APPEND to progress.txt:

```
## [Date] - [Story ID]
- What was implemented
- Files changed
- **Learnings:**
  - Patterns discovered
  - Gotchas encountered
---
```

## Codebase Patterns

Add reusable patterns to the TOP of progress.txt:

```
## Codebase Patterns
- Package manager: Prefer `uv` over pip (e.g., `uv add`, `uv run`)
- Migrations: Use alembic with `op.create_table` idempotent checks
- FastAPI: Use `Depends()` for dependency injection
- Pydantic: Inherit from BaseModel, use Field() for validation
- Async: Use `async def` with `await` for I/O operations
- Testing: Use pytest fixtures, mock external services
- Type hints: Always annotate function signatures
```

## Stop Condition

If ALL stories pass, reply:
<promise>COMPLETE</promise>

Otherwise end normally.
