# Domains

This folder contains the domains: a database, the tools that act on it, a
policy for the agent and a set of tasks.

## Available domains

| Domain | Where | Description |
|--------|-------|-------------|
| `airline` | `src/tau2/domains/airline/`, `data/tau2-airline/` | Flight booking, changes, cancellations, baggage and compensation. The only shipped domain: 50 tasks. |
| `mock` | `tests/mock_domain/` (test-only) | Small task-management domain with agent and user tools, used by the test suite. Registered by `tests/conftest.py`; not available from the `tau2` CLI. |

## Domain structure

Each domain package has:

- `data_model.py`: the domain's data models, including a subclass of `DB`
  (`tau2/environment/db.py`), e.g. `FlightDB(DB)`.
- `tools.py`: a subclass of `ToolKitBase` (`tau2/environment/toolkit.py`),
  e.g. `AirlineTools(ToolKitBase)`, whose methods decorated with `@is_tool(...)`
  become the agent's tools.
- `environment.py`:
  - `get_environment(...)` returns an `Environment` with the domain's name,
    policy and toolkit(s). Airline's accepts optional `db` and `policy`
    overrides.
  - `get_tasks(task_split_name)` returns the domain's `Task`s, optionally for
    one split.
  - `get_tasks_split()` returns a dict mapping split names to task IDs.
- `utils.py`: data paths (e.g. `AIRLINE_DB_PATH`).
- Optional: `user_data_model.py` and `user_tools.py` for a user-side database
  and tools the user simulator can call (the mock domain has them; airline
  does not).

## The airline domain

| File | Contents |
|------|----------|
| `data/tau2-airline/db.json` | Users, reservations and flights (`FlightDB`) |
| `data/tau2-airline/policy.md` | The agent policy |
| `data/tau2-airline/tasks.json` | 50 tasks, all with `reward_basis` `[DB, COMMUNICATE]` |
| `data/tau2-airline/split_tasks.json` | Splits `base` (all 50, the default), `train` (30) and `test` (20) |
| `data/tau2-airline/user_personas.json` | The two caller personas and each task's caller gender (see `tau2/user/personas.py`) |

`tau2 domain airline` prints the policy and a table of the 14 agent tools.

## Task schema and evaluation

Each entry in `tasks.json` is a `Task` (see
[`src/tau2/data_model/tasks.py`](../data_model/tasks.py)) with an
`evaluation_criteria` block. Note: `evaluation_criteria.actions` is *one*
reference trajectory used to derive the target DB end state, not a per-call
requirement on the agent (unless `RewardType.ACTION` is in `reward_basis`,
which no airline task uses). See
[`docs/evaluation.md`](../../../docs/evaluation.md) for the full breakdown.

## Adding a domain

1. Create `src/tau2/domains/<name>/` with the files above, and put its data in
   `data/tau2-<name>/`: `tasks.json`, `split_tasks.json` (at least a
   `base` split; the file name is `split_<tasks file stem>.json`),
   `policy.md` and the database file.
2. Register it in `src/tau2/registry.py`:

   ```python
   from tau2.domains.my_domain.environment import (
       get_environment as my_domain_get_environment,
       get_tasks as my_domain_get_tasks,
       get_tasks_split as my_domain_get_tasks_split,
   )

   registry.register_domain(my_domain_get_environment, "my_domain")
   registry.register_tasks(
       my_domain_get_tasks, "my_domain", get_task_splits=my_domain_get_tasks_split
   )
   ```

   The domain name is what `tau2 run --domain` accepts; the task-set name is
   what `--task-set-name` accepts (it defaults to the domain name).
   `tests/conftest.py` shows the same calls for the mock domain, made at
   runtime on the global `tau2.registry.registry`.
3. Add tests under `tests/test_domains/test_<name>/`.

The agent's policy is the domain policy followed by
`AGENT_DECISION_EXPLANATION_CLAUSE` and `AGENT_LANGUAGE_CLAUSE`
(`tau2/runner/build.py`). A domain without a
`data/tau2-<name>/user_personas.json` runs its user simulator without a
caller persona.

## Tests

Domain tests live in `tests/test_domains/test_<domain_name>/`:

- `test_tools_airline.py`: the airline tools, called through
  `environment.get_response(ToolCall(...))`.
- `test_prompts_airline.py`: pins the agent and user-simulator prompts to
  those of τ-bench 2.0.

```sh
uv run pytest tests/test_domains/test_airline
```
