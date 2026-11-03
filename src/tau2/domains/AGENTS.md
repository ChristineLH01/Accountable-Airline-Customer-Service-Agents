# AGENTS.md — src/tau2/domains/

> See `README.md` for the domain layout, the airline data files and how to add
> a domain.

## Rules for Working in This Directory

### Shipped Domains

`airline` is the only domain in `src/tau2/domains/` and the only one
registered in `src/tau2/registry.py`. The `mock` domain used by tests lives in
`tests/mock_domain/` (data in `tests/mock_domain/data/`) and is registered at
test time by `tests/conftest.py`; do not import it from `src/`.

### Required Files per Domain

Every domain in `src/tau2/domains/<name>/` MUST have:

- `data_model.py`: `DB` subclass (e.g. `FlightDB(DB)`)
- `tools.py`: `ToolKitBase` subclass with `@is_tool(ToolType.READ|WRITE|GENERIC|THINK)` decorated methods
- `environment.py`: exports `get_environment()`, `get_tasks()` and `get_tasks_split()`
- `utils.py`: data paths pointing to `data/tau2-<name>/`
- `__init__.py`

Optional: `user_tools.py`, `user_data_model.py`.

### Registration Is Mandatory

After creating or renaming a domain, register it in `src/tau2/registry.py`:

```python
registry.register_domain(get_environment, "domain_name")
registry.register_tasks(get_tasks, "domain_name", get_task_splits=get_tasks_split)
```

Without this the domain is not usable from the CLI (`--domain` choices come
from the registry). Tests and scripts may instead register on the global
`tau2.registry.registry` at runtime, as `tests/conftest.py` does.

### Task Split Rules

- The split file MUST include a `base` split; it is the default for evaluation.
- Split file naming follows `split_<tasks_file_stem>.json` (e.g. `split_tasks.json` for `tasks.json`).
- Airline also ships `train` (30 tasks) and `test` (20 tasks) splits.

### Tool Decorator Types

Use the correct `ToolType`; it drives metrics and `partial_action_reward`:

- `ToolType.READ`: read-only queries (e.g. look up a reservation)
- `ToolType.WRITE`: state-mutating operations (e.g. cancel a reservation)
- `ToolType.GENERIC`: utilities (e.g. `calculate`, `transfer_to_human_agents`)
- `ToolType.THINK`: internal reasoning, no side effects

Evaluation replay re-executes only tools with `mutates_state=True`, which
`is_tool` infers as `True` for WRITE and `False` otherwise. Pass
`mutates_state` explicitly for exceptions. A non-mutating tool that actually
changes the DB would be skipped on replay and break the DB check.

### Data Files and Prompts

Domain data lives in `data/tau2-<name>/`. Be careful modifying
`db.json`, `policy.md`, `tasks.json` and `user_personas.json`: the framework
reads them at runtime, and the airline policy, tasks and personas are pinned
by `tests/test_domains/test_airline/test_prompts_airline.py` to reproduce the
τ-bench 2.0 English prompts. Run `tau2 check-data` and the airline tests after
changing data files.

### Tool Arguments

Some tools accept `dict` arguments and convert them internally to Pydantic
models. When adding new tools, prefer typed Pydantic model arguments. If
accepting `dict`, convert explicitly (e.g. `FlightInfo(**flight)`).

### Tests

Tests go in `tests/test_domains/test_<domain_name>/`:
- `test_tools_<domain_name>.py`: test tools via `environment.get_response(ToolCall(...))`
- `test_user_tools_<domain_name>.py`: if user tools exist

Run: `uv run pytest tests/test_domains/test_<domain_name>/`
