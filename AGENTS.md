# AGENTS.md

> Instructions for AI coding agents working on this codebase.

## Project Overview

A course project, "Accountable Airline Customer Service Agents", built on
τ-bench (tau2-bench 2.0 / τ³-bench). The codebase is trimmed to one setting:
the **airline domain, text (turn-based) conversations, English**. An LLM agent
helps an LLM user simulator through a written airline policy and a set of
tools; each simulation is scored against the task's expected end state and the
information the agent had to communicate.

Keep changes within this scope (airline, text, English) unless asked
otherwise.

## Setup

```bash
uv sync --extra dev            # core + pytest, ruff, pre-commit
cp .env.example .env           # OPENROUTER_API_KEY
uv run tau2 check-data         # verify installation
```

All models are called through [OpenRouter](https://openrouter.ai): one
`OPENROUTER_API_KEY` covers every model, and model names are OpenRouter model
IDs (e.g. `openai/gpt-6.1-sol`). `tau2.utils.llm_utils.generate()` is the single
call boundary; it uses LiteLLM's OpenRouter provider as the HTTP client and
records OpenRouter's reported cost. Defaults (`config.py`): agent
`openai/gpt-6.1-sol` at `reasoning_effort: medium`, user simulator
`openai/gpt-6.1-sol` at `low`, judges and review `openai/gpt-6-luna` at
`medium`. These reasoning models do not take `temperature`. `langfuse` and
`redis` are not declared dependencies; install them only if you enable
`USE_LANGFUSE` or `LLM_CACHE_ENABLED` with the redis cache in `config.py`.

## Common Commands

| Command | What it does |
|---------|-------------|
| `make test` | Run the test suite (no API keys needed) |
| `make test-live` | Run tests that call real models (`--run-live-api -m live_api`; costs money) |
| `make lint` / `make format` / `make lint-fix` | Ruff |
| `make check-all` | Lint + format (same as the pre-commit hook) |
| `make env-cli` | Interactive environment CLI for testing domain tools |
| `make clean` | Remove venv, caches, build artifacts |

Always run `make check-all` and `make test` before committing.

## Running Evaluations

```bash
tau2 run --agent-llm openai/gpt-6.1-sol --user-llm openai/gpt-6.1-sol --num-trials 1 --num-tasks 5
tau2 run --task-split-name test --num-trials 4 --max-concurrency 10 --save-to my-run
tau2 run --save-to my-run --auto-resume          # resume an interrupted run
tau2 run --dry-run                               # print the resolved RunConfig
tau2 view --file data/simulations/my-run         # browse results
tau2 evaluate-trajs data/simulations/my-run      # re-score saved runs
tau2 review data/simulations/my-run              # LLM review of conversations
tau2 domain airline                              # policy + tool reference
```

Results go to `data/simulations/<save_to>/results.json` (git-ignored).

## Architecture

```
src/tau2/
├── agent/           # HalfDuplexAgent base, LLMAgent, LLMGTAgent
├── user/            # UserSimulator, guideline scaffold (guidelines.py), caller personas (personas.py)
├── orchestrator/    # Orchestrator: turn-based agent/user/environment loop
├── environment/     # Environment, ToolKitBase, Tool, DB base classes
├── domains/airline/ # data model, tools, environment, task loading
├── evaluator/       # env (DB), action, communicate, decision-reason, NL-assertion evaluators; reviewers
├── metrics/         # pass^k and run metrics
├── runner/          # simulation.py (layer 1), build.py (layer 2), batch.py (layer 3), checkpoint, progress
├── data_model/      # messages, tasks, RunConfig, SimulationRun, Results
├── scripts/         # view, evaluate-trajs, review, domain doc, check-data
├── utils/           # llm_utils (OpenRouter calls), display, io helpers
├── cli.py           # `tau2` command
├── config.py        # central defaults
└── registry.py      # agents, users, domains, task sets
```

Other top-level directories:
- `data/tau2-airline/` — `db.json`, `policy.md`, `tasks.json` (50 tasks),
  `split_tasks.json` (`base`/`train`/`test`/`scored`), `user_personas.json`,
  `mannual_label.json` (answer key of golden decisions and reasons; see
  Evaluation)
- `tests/` — pytest suite; `tests/mock_domain/` is a tiny test-only domain
  registered as `mock` by `tests/conftest.py`
- `examples/agents/` — custom agent examples
- `docs/` — getting started, evaluation, running simulations, CLI reference

## Key Patterns

### Registry

Agents, users, domains and task sets are registered in `src/tau2/registry.py`:

```python
registry.register_agent_factory(create_my_agent, "my_agent")
registry.register_domain(get_environment, "my_domain")
registry.register_tasks(get_tasks, "my_domain", get_task_splits=get_tasks_split)
```

### Agents

Agents subclass `HalfDuplexAgent` (`src/tau2/agent/base_agent.py`) and implement
`get_init_state()` and `generate_next_message()`. The constructor signature is
`__init__(self, tools: list[Tool], domain_policy: str)`; LLM agents mix in
`LLMConfigMixin` for `llm` / `llm_args`. Factories receive
`tools, domain_policy, llm, llm_args, task`.

### Run configuration

`tau2.data_model.simulation.RunConfig` is the single run configuration (CLI
flags map onto it in `cli.config_from_args`). Defaults live in `config.py`;
import constants from there rather than redefining them.

### Prompts are pinned

The agent sees `policy.md`, then `AGENT_DECISION_EXPLANATION_CLAUSE`, then
`AGENT_LANGUAGE_CLAUSE` (`runner/build.py`, `agent_domain_policy`). The
explanation clause is the project's addition: it asks the agent to tell the
customer the decision and its policy-based reason whenever it approves, denies
or partly approves a request. The user simulator sees the English text scaffold
(`user/guidelines.py`) with a caller persona (`user/personas.py`, chosen by the
task's caller gender from `user_personas.json`). Without the explanation
clause these are tau-bench 2.0's English airline prompts byte for byte.
`tests/test_domains/test_airline/test_prompts_airline.py` pins both facts with
two digests. If you intentionally change the policy, tasks, scaffold, personas
or clauses, update the digests and call it out: results are only comparable
across runs with identical prompts.

### Evaluation

`evaluate_simulation()` (`evaluator/evaluator.py`) multiplies the components in
the task's `reward_basis` (airline: `DB` and `COMMUNICATE`). `COMMUNICATE` uses
a semantic LLM judge by default (`--communicate-scorer substring` for the
original substring match). The run then adds `REASON` (`evaluator_reason.py`):
an LLM judge grades the decision and reason the agent told the customer against
the golden labels below. `--reason-scoring gate` (default) multiplies it into
the reward; `report` records it only; `off` skips it. `reward_info.task_reward`
keeps the reward without `REASON`, so runs stay comparable with tau-bench. The
judge prompt is pinned by a digest in `tests/test_evaluator/test_reason_judge.py`;
change it only with a new `ReasonJudgeSettings.prompt_version`. Reviews (`--auto-review`, `tau2 review`) and
authentication classification are diagnostics and never change the reward.

The answer key for the agent's decision explanations is
`data/tau2-airline/mannual_label.json`: one entry per task with the final
`golden_decision` and `golden_reason`, set for all 50 tasks from the three
annotators' labels (`golden_decision_1/2/3`, `golden_reason_1/2/3`, still kept
in each entry). `docs/golden_labels.md` records how they were set. Tasks 9, 37
and 39 carry `excluded_from_scoring: true` with an `exclusion_reason`; leave them
out of scoring (the `scored` split already does). Load the final labels with
`tau2.domains.airline.golden_reasons.load_golden_reasons()`, which ignores the
annotator fields and returns `None` for labels that are still empty.

## Testing

```bash
make test                                   # whole suite
uv run pytest tests/test_domains/test_airline
uv run pytest tests/test_orchestrator.py
uv run pytest --run-live-api -m live_api    # real model calls; needs keys
```

Tests marked `live_api` are skipped unless `--run-live-api` is passed. See
`tests/AGENTS.md` for the layout.

## Code Style

- **Formatter/linter**: Ruff (configured in `pyproject.toml`), line length 88
- **Python**: >=3.12, <3.14
- **Type hints**: encouraged, especially for public APIs
- **Docstrings**: required for public APIs and complex functions
- **Models**: use Pydantic `BaseModel` for data classes

Ruff rules: `E4`, `E7`, `E9`, `F`, `I` (with `E501` and `F541` ignored).

## Commit Conventions

```
feat: add an accountability check to the agent
fix: resolve environment tool timeout issues
docs: update evaluation guide
test: add airline policy tests
```

## Things to Watch Out For

- **`.env`**: never commit it. Use `.env.example` as the reference.
- **`data/`**: the airline data defines the benchmark. Changing `tasks.json`,
  `db.json` or `policy.md` changes results; do it deliberately and document it.
- **`config.py`**: single source of truth for defaults.
- **`registry.py`**: new agents, users and domains must be registered to be
  usable from the CLI.
- **Task splits**: `base` (all 50 tasks) is the default for evaluation;
  `train` (30) / `test` (20) are for experiments that need held-out tasks.
  `scored` (47) leaves out tasks 9, 37 and 39, whose labels or answer keys are
  disputed (`excluded_from_scoring` in `mannual_label.json`; see
  `docs/golden_labels.md`). Use it for evaluation runs until those are settled.
- **Pre-commit hook**: runs `make check-all`. Fix any issues before committing.
- **No CI**: there are no GitHub Actions workflows; run tests locally.
