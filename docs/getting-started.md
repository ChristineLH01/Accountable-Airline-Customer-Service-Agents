# Getting Started

This guide installs the framework, sets up API keys and runs a first
evaluation. The framework covers one benchmark: an LLM customer-service
agent and an LLM user simulator talk turn by turn (text, English) on the
**airline** domain's 50 tasks.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python
  package and project manager
- Python 3.12 or 3.13 (uv downloads the pinned 3.12 automatically if needed)

## Install

From the repository root:

```bash
UV_FROZEN=1 uv sync --extra dev
```

This creates `.venv`, installs the locked dependencies plus the dev tools
(pytest, ruff, pre-commit), and provides the `tau2` command. Set
`UV_FROZEN=1` (or pass `--frozen`) on every `uv` command so uv never
rewrites `uv.lock`; the examples below assume it is exported:

```bash
export UV_FROZEN=1
```

Check that the data directory is found:

```bash
uv run tau2 check-data
```

If you install the package somewhere other than this checkout (for example
with `uv pip install .`), point it at the data with
`export TAU2_DATA_DIR=/path/to/repo/data`.

## API keys

All models are called through [OpenRouter](https://openrouter.ai), so a
single key gives access to every model. Create one at
[openrouter.ai/keys](https://openrouter.ai/keys), then copy the example file
and fill it in:

```bash
cp .env.example .env    # set OPENROUTER_API_KEY
```

Model names on the command line (`--agent-llm`, `--user-llm`,
`--communicate-judge`, `--review-model`) are OpenRouter model IDs such as
`openai/gpt-6.1-sol` or `anthropic/claude-opus-4.5`; the list is at
[openrouter.ai/models](https://openrouter.ai/models).

## First run

```bash
uv run tau2 run --num-tasks 5
```

This runs the first 5 tasks of the `base` split with the default agent and
user simulator. Choose models and tasks explicitly with, for example:

```bash
uv run tau2 run --agent-llm openai/gpt-6.1-sol --user-llm openai/gpt-6.1-sol --task-ids 0 1 2 --num-trials 3
```

Add `--dry-run` to print the resolved configuration without running
anything. The released τ-bench 2.0 text benchmark used `openai/gpt-5.5`
as the user simulator; to match it, pass
`--user-llm openai/gpt-5.5 --user-llm-args '{"reasoning_effort": "low"}'`.
See the [CLI reference](cli-reference.md) for every option.

## Where results go

Each run writes one directory under `data/simulations/`:

```
data/simulations/<run_name>/
├── results.json        # run info, task definitions and every simulation
└── artifacts/          # only with --verbose-logs
    └── task_<task_id>/sim_<simulation_id>/
        ├── task.log
        ├── sim_status.json
        └── llm_debug/  # LLM request/response logs
```

`<run_name>` is `--save-to` if given, otherwise
`<timestamp>_airline_<agent>_<agent-model>_<user>_<user-model>`.
`results.json` is rewritten after every finished simulation, so an
interrupted run can be continued by rerunning the same command with the
same `--save-to` (add `--auto-resume` to skip the prompt).
`tau2 convert-results` switches a results file between this single-file
`json` format and a `dir` format with one file per simulation.

## Viewing results

```bash
uv run tau2 view
```

Browse runs in `data/simulations/`, see metrics (average reward, pass^k,
cost), and inspect each conversation, its reward breakdown and its task.
`tau2 view --file data/simulations/<run_name>` opens one run directly.

## Configuration defaults

Defaults live in [`src/tau2/config.py`](../src/tau2/config.py); most can be
overridden on the command line.

| Setting | Default | CLI flag |
|---------|---------|----------|
| Agent implementation | `llm_agent` | `--agent` |
| Agent model and args | `openai/gpt-6.1-sol`, `{"reasoning_effort": "medium"}` | `--agent-llm`, `--agent-llm-args` |
| User simulator model and args | `openai/gpt-6.1-sol`, `{"reasoning_effort": "low"}` | `--user-llm`, `--user-llm-args` |
| Communicate-info judge | `openai/gpt-6-luna`, `{"reasoning_effort": "medium"}` | `--communicate-scorer`, `--communicate-judge` |
| NL-assertion judge (diagnostic) | `openai/gpt-6-luna`, `{"reasoning_effort": "medium"}` | none |
| Conversation review model | `openai/gpt-6-luna`, `{"reasoning_effort": "medium"}` | `--review-model` |
| Max steps / max tool errors | `200` / `10` | `--max-steps`, `--max-errors` |
| Trials / concurrency / seed | `1` / `3` / `300` | `--num-trials`, `--max-concurrency`, `--seed` |
| Retries per failed simulation | `3`, 1 s apart | `--max-retries`, `--retry-delay` |
| Log level | `ERROR` | `--log-level` |

LLM call caching (`LLM_CACHE_ENABLED`, Redis) and Langfuse tracing
(`USE_LANGFUSE`) are off by default. To use them, install `redis` or
`langfuse` (`uv pip install redis langfuse`) and switch the flag in
`config.py`.

## Other tools

- `uv run tau2 domain airline` prints the agent policy and the airline tools.
- `make env-cli` opens an interactive prompt where an LLM answers questions
  by calling the airline tools directly, without a simulation
  (`:q` quits, `:d` changes domain, `:n` starts a new session).
- `make test` runs the test suite; `make check-all` runs ruff lint and
  format.
- `make clean` removes the virtual environment and build caches.

## Next steps

- [Running simulations](running_simulations.md): the Python API
- [CLI reference](cli-reference.md): every `tau2` command and option
- [Task schema and evaluation](evaluation.md): how a simulation is scored
- [Agent developer guide](../src/tau2/agent/README.md): build and evaluate your own agent
- [Domains](../src/tau2/domains/README.md): the airline domain and how to add one
