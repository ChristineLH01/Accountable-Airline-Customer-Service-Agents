# Running Simulations

This guide covers the `tau2.runner` API for running airline simulations at
different levels of control, from a single CLI command to wiring every
component yourself.

## Architecture overview

The runner is organized into three layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| **Layer 1** | `tau2.runner.simulation` | Execute a pre-built `Orchestrator` and evaluate it. No registry, no config. |
| **Layer 2** | `tau2.runner.build` | Build the environment, agent, user and orchestrator from registered names or a `RunConfig`. |
| **Layer 3** | `tau2.runner.batch` | Batch execution with concurrency, checkpointing, retries, logging and optional auto-review. |

Each layer builds on the one below it; you can enter at any level. Everything
is also importable from `tau2.run` and `tau2.runner`.

---

## Level 1: CLI

```bash
# All 50 airline tasks with the default models
tau2 run

# Specific tasks, three trials each
tau2 run --agent-llm openai/gpt-6.1-sol --user-llm openai/gpt-6.1-sol --task-ids 0 1 --num-trials 3

# More concurrency, named output, resume without prompting
tau2 run --max-concurrency 8 --save-to my_run --auto-resume
```

See `tau2 run --help` or the [CLI reference](cli-reference.md) for all
options.

---

## Level 2: Config-based Python API

Describe a run with `RunConfig` (`tau2.data_model.simulation.RunConfig`, also
exported as `tau2.RunConfig`) and pass it to `run_domain()`:

```python
from tau2 import RunConfig
from tau2.metrics.agent_metrics import compute_metrics
from tau2.runner import run_domain

config = RunConfig(
    domain="airline",
    agent="llm_agent",
    llm_agent="openai/gpt-6.1-sol",
    llm_user="openai/gpt-6.1-sol",
    num_tasks=5,
    num_trials=3,
    max_concurrency=4,
    seed=42,
)

results = run_domain(config)  # also prints the metrics table
metrics = compute_metrics(results)
print(f"Average reward: {metrics.avg_reward}")
print(f"pass^k: {metrics.pass_hat_ks}")
```

`run_domain()` loads and filters the tasks, saves to
`data/simulations/<save_to or auto name>/results.json`, runs the batch and
displays the metrics.

`RunConfig` fields (defaults from [`src/tau2/config.py`](../src/tau2/config.py)):

| Group | Fields |
|-------|--------|
| Tasks | `domain="airline"`, `task_set_name=None` (the domain's task set), `task_split_name="base"`, `task_ids=None`, `num_tasks=None` |
| Agent | `agent="llm_agent"`, `llm_agent="openai/gpt-6.1-sol"`, `llm_args_agent={"reasoning_effort": "medium"}` |
| User simulator | `user="user_simulator"`, `llm_user="openai/gpt-6.1-sol"`, `llm_args_user={"reasoning_effort": "low"}` |
| Scoring | `communicate_scoring=CommunicateScoring()` (semantic judge; see [evaluation](evaluation.md#communication-checks)) |
| Execution | `num_trials=1`, `max_steps=200`, `max_errors=10`, `timeout=None`, `enforce_communication_protocol=False`, `max_concurrency=3`, `seed=300`, `max_retries=3`, `retry_delay=1.0`, `auto_resume=False` |
| Logging | `log_level="ERROR"`, `verbose_logs=False` |
| Review | `auto_review=False`, `review_mode="full"`, `review_model="openai/gpt-6-luna"` |
| Output | `save_to=None` (a directory name under `data/simulations/`) |

To score communicate-info with substring matching instead of the LLM judge:

```python
from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring

config = RunConfig(
    communicate_scoring=CommunicateScoring(scorer=CommunicateScorer.SUBSTRING),
)
```

### Running specific tasks

```python
from pathlib import Path

from tau2 import RunConfig
from tau2.runner import get_tasks, run_tasks

config = RunConfig(llm_agent="openai/gpt-6.1-sol")

tasks = get_tasks("airline", task_ids=["0", "1"])

results = run_tasks(
    config,
    tasks,
    save_path=Path("my_results/results.json"),  # None: do not persist
    save_dir=Path("my_results/"),  # per-task logs with verbose_logs=True
)
```

### Running a single task

```python
from tau2 import RunConfig
from tau2.runner import get_tasks, run_single_task

config = RunConfig(llm_agent="openai/gpt-6.1-sol")
task = get_tasks("airline", task_ids=["0"])[0]

result = run_single_task(config, task, seed=42)
print(f"Task {result.task_id}: reward={result.reward_info.reward}")
```

`run_single_task()` does not retry or write a results file; `run_tasks()`
adds both.

---

## Level 3: Instance-based Python API

Build instances yourself and call `run_simulation()`. Use this for agents
that are not registered, fine-grained control over construction, or
embedding simulations in your own pipeline.

### Using the build helpers

```python
from tau2 import RunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner import (
    build_agent,
    build_environment,
    build_orchestrator,
    build_user,
    get_tasks,
    run_simulation,
)

task = get_tasks("airline", task_ids=["0"])[0]

# Option A: build the orchestrator from a config (uses the registry)
config = RunConfig(llm_agent="openai/gpt-6.1-sol")
orchestrator = build_orchestrator(config, task, seed=42)
result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)

# Option B: build the components individually
env = build_environment("airline")
agent = build_agent("llm_agent", env, llm="openai/gpt-6.1-sol", task=task)
user = build_user("user_simulator", env, task, llm="openai/gpt-6.1-sol")
```

`build_orchestrator()` gives the user simulator the task's caller persona;
`build_user()` does so only when you pass `persona_config` (see below).

### Fully custom instances

```python
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, get_tasks, run_simulation
from tau2.runner.build import agent_domain_policy
from tau2.user.personas import assign_user_persona
from tau2.user.user_simulator import UserSimulator

task = get_tasks("airline", task_ids=["0"])[0]
env = build_environment("airline")

# Your agent: any HalfDuplexAgent (see src/tau2/agent/README.md)
agent = MyAgent(tools=env.get_tools(), domain_policy=agent_domain_policy(env))

user = UserSimulator(
    llm="openai/gpt-6.1-sol",
    instructions=str(task.user_scenario),
    tools=None,  # airline has no user tools
    persona_config=assign_user_persona("airline", task.id, seed=300),
)

orchestrator = Orchestrator(
    domain="airline",
    agent=agent,
    user=user,
    environment=env,
    task=task,
    max_steps=200,
    max_errors=10,
    seed=42,
)

result = run_simulation(orchestrator)
print(f"Reward: {result.reward_info.reward}")
```

### The prompts and the τ-bench 2.0 baseline

The registered builders reproduce the English airline prompts of τ-bench 2.0
plus one agent instruction, the decision-explanation clause;
`tests/test_domains/test_airline/test_prompts_airline.py` pins them with a
digest and checks that they match τ-bench 2.0 once the clause is removed. Two
pieces are added on top of the raw data, so keep them when you build instances
yourself:

- **Agent policy**: `agent_domain_policy(env)` returns
  `data/tau2-airline/policy.md`, `AGENT_DECISION_EXPLANATION_CLAUSE` and
  `AGENT_LANGUAGE_CLAUSE` (all in `tau2.runner.build`), joined by `"\n\n"`.
  Passing `env.get_policy()` alone omits both clauses.
- **Caller persona**: the user simulator's system prompt is the scaffold in
  `tau2.user.guidelines` with a persona from
  `data/tau2-airline/user_personas.json` inserted. There are two
  personas, `mamadou_diallo` (male) and `priya_patil` (female), chosen by
  each task's caller gender; `assign_user_persona(domain, task_id, seed)`
  picks it. Each `SimulationRun` records the persona ID in `user_persona`.

### Evaluating a simulation separately

`run_simulation()` already attaches `reward_info`. To re-score a
`SimulationRun` (for example with a different `EvaluationType`):

```python
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

reward_info = evaluate_simulation(
    simulation=result,
    task=task,
    evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
    domain="airline",
    communicate_scoring=CommunicateScoring(),
)
```

For saved runs, `tau2 evaluate-trajs` does this for every simulation in a
results file.

---

## API reference

### Layer 1: execution

- **`run_simulation(orchestrator, *, evaluation_type=EvaluationType.ALL, env_kwargs=None, communicate_scoring=None)`**:
  run the orchestrator, record the policy and caller persona, evaluate, and
  return the `SimulationRun` with `reward_info` attached.
  `communicate_scoring=None` means the default semantic judge.

### Layer 2: build

- **`build_environment(domain, *, env_kwargs=None)`**: environment for a
  registered domain; `env_kwargs` go to its constructor (for airline, `db`
  and `policy`).
- **`build_agent(agent_name, environment, *, llm=None, llm_args=None, task=None)`**:
  agent from a registered factory, given the environment's tools and
  `agent_domain_policy(environment)`. `llm_agent_gt` needs `task`.
- **`build_user(user_name, environment, task, *, llm=None, llm_args=None, persona_config=None)`**:
  user simulator from a registered name, with the task's user tools and
  `str(task.user_scenario)` as instructions.
- **`build_orchestrator(config, task, *, seed=None, simulation_id=None)`**:
  `Orchestrator` from a `RunConfig`, with the caller persona assigned from
  `config.seed` and `task.id`. `seed=None` uses `config.seed`.

### Layer 3: batch execution

- **`run_domain(config)`**: load and filter tasks (applying the agent's
  registered task filter), save to
  `data/simulations/<run_name>/results.json`, run, display metrics.
- **`run_tasks(config, tasks, *, save_path=None, save_dir=None, evaluation_type=EvaluationType.ALL, console_display=True, results_format="json")`**:
  run every task for `config.num_trials` trials with a thread pool of
  `config.max_concurrency`, retries, checkpointing to `save_path` and
  resume. `results_format` (`"json"` or `"dir"`) applies to new results
  files. Returns `Results`.
- **`run_single_task(config, task, *, seed=None, evaluation_type=EvaluationType.ALL, save_dir=None, verbose_logs=False, auto_review=False, review_mode="full", review_model=None)`**:
  one simulation with per-task logging and optional auto-review.

### Helpers

- **`get_options()`**: registered domains, agents, users and task sets.
- **`get_tasks(task_set_name, task_split_name=None, task_ids=None, num_tasks=None, task_kwargs=None)`**:
  load tasks, then filter by IDs (all must exist) and count.
  `task_split_name=None` loads every task in the set.
- **`load_tasks(task_set_name, task_split_name=None, task_kwargs=None)`**:
  load tasks from the registry without filtering.
- **`load_task_splits(task_set_name)`**: the split dictionary
  (`base`, `train`, `test` for airline), or `None`.
- **`get_environment_info(domain_name, include_tool_info=False, env_kwargs=None)`**:
  `EnvironmentInfo` for a domain.
- **`get_info(config)`**: run metadata (`Info`) stored in `results.json`.
- **`make_run_name(config)`**: the timestamped default run name.

---

## Choosing the right level

| Scenario | Level |
|----------|-------|
| Quick evaluation from the command line | Level 1 (CLI) |
| Evaluations or sweeps from a Python script | Level 2 (`RunConfig`) |
| Developing and debugging a custom agent | Level 3 (instances) |
| Embedding simulations in a larger pipeline | Level 3 (instances) |
