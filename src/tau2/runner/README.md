# tau2.runner

Simulation execution framework with a layered architecture for running
evaluations at different levels of control. Usage guide with examples:
[`docs/running_simulations.md`](../../../docs/running_simulations.md).

## Module structure

```
runner/
├── __init__.py        # Package exports
├── simulation.py      # Layer 1: run_simulation()
├── build.py           # Layer 2: build_* functions, agent_domain_policy()
├── batch.py           # Layer 3: run_domain(), run_tasks(), run_single_task()
├── helpers.py         # Task loading, run metadata, utilities
├── checkpoint.py      # Save/resume logic for batch runs
├── progress.py        # Retry logic and status monitoring
└── README.md          # This file
```

## Layers

### Layer 1: `simulation.py`, execute

`run_simulation(orchestrator, *, evaluation_type=EvaluationType.ALL, env_kwargs=None, communicate_scoring=None)`
runs a fully constructed `Orchestrator`, records the policy and caller
persona on the result, evaluates it and returns the `SimulationRun` with
`reward_info` attached.

- **No registry dependency**: everything is in the orchestrator.
- **No config**: no `RunConfig` needed.
- **No side effects**: no logging setup, file saving or review.

```python
from tau2.runner import run_simulation

result = run_simulation(orchestrator)
```

### Layer 2: `build.py`, build

Turns names and configuration into live instances, resolving names through
the registry.

Low-level builders:
- `build_environment(domain, *, env_kwargs=None)`
- `build_agent(agent_name, environment, *, llm=None, llm_args=None, task=None)`:
  calls the registered factory with the environment's tools and
  `agent_domain_policy(environment)` (the policy plus
  `AGENT_DECISION_EXPLANATION_CLAUSE` and `AGENT_LANGUAGE_CLAUSE`).
- `build_user(user_name, environment, task, *, llm=None, llm_args=None, persona_config=None)`

High-level builder:
- `build_orchestrator(config, task, *, seed=None, simulation_id=None)`:
  builds all three from a `RunConfig`, assigns the caller persona with
  `assign_user_persona(config.domain, task.id, config.seed)` and returns an
  `Orchestrator`.

```python
from tau2 import RunConfig
from tau2.runner import build_orchestrator, run_simulation

config = RunConfig(llm_agent="openai/gpt-6.1-sol")
orchestrator = build_orchestrator(config, task, seed=42)
result = run_simulation(orchestrator)
```

### Layer 3: `batch.py`, batch

Batch execution with the operational concerns:

- **Concurrency**: a thread pool of `config.max_concurrency` workers.
- **Seeds**: one seed per trial, drawn from `config.seed`.
- **Checkpointing**: results are saved after every simulation; an existing
  results file is resumed (with a prompt unless `auto_resume`). Changing the
  communicate scorer between runs is refused.
- **Retries**: failed simulations are retried `max_retries` times
  `retry_delay` seconds apart (`progress.run_with_retry`); after that a
  placeholder with `INFRASTRUCTURE_ERROR` is recorded.
- **Status monitoring**: progress printed every 30 s (`progress.StatusMonitor`).
- **Side effects**: per-task logs and LLM debug logs with `verbose_logs`
  (under `<save_dir>/artifacts/`), and optional auto-review.
- **Ctrl+C**: cancels pending simulations; completed ones are already on disk.

Entry points:
- `run_domain(config)`: load and filter tasks, save to
  `data/simulations/<save_to or make_run_name(config)>/results.json`, run,
  display metrics. Used by `tau2 run`.
- `run_tasks(config, tasks, *, save_path=None, save_dir=None, evaluation_type=EvaluationType.ALL, console_display=True, results_format="json")`
- `run_single_task(config, task, *, seed=None, evaluation_type=EvaluationType.ALL, save_dir=None, verbose_logs=False, auto_review=False, review_mode="full", review_model=None)`

```python
from tau2 import RunConfig
from tau2.runner import run_domain

results = run_domain(RunConfig(llm_agent="openai/gpt-6.1-sol", num_tasks=5))
```

## Supporting modules

- **`helpers.py`**: task loading (`get_tasks`, `load_tasks`,
  `load_task_splits`, `filter_tasks`, `select_run_tasks`), run metadata
  (`get_info`, `make_run_name`) and registry queries (`get_options`,
  `get_environment_info`).
- **`checkpoint.py`**: `try_resume()` to resume from existing results and
  `create_checkpoint_fns()` for the thread-safe save/replace functions (`json` and
  `dir` results formats).
- **`progress.py`**: `run_with_retry()` and `StatusMonitor`.

## Relationship to `tau2.run`

`tau2.run` is a thin facade that re-exports this package's API (plus
`RunConfig` and `EvaluationType`); the CLI imports `get_options` and
`run_domain` from it. New code can import from either.
