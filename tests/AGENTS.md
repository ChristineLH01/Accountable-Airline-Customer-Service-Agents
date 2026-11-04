# AGENTS.md — tests/

## Test Layout

```
tests/
├── conftest.py                    # Registers the mock domain; shared fixtures (mock tasks)
├── mock_domain/                   # Tiny test-only domain (code + data), registered as "mock"
├── test_agent.py                  # LLMAgent
├── test_user.py                   # UserSimulator
├── test_orchestrator.py           # Orchestrator initialization, steps, protocol checks
├── test_environment.py            # Environment base class, state replay
├── test_run.py                    # Run API, registry options, offline scripted run
├── test_tasks.py                  # Task loading/validation
├── test_checkpoint.py             # Save/resume of batch runs
├── test_results_format.py         # json/dir results formats
├── test_evaluate_trajectories.py  # Re-scoring saved runs
├── test_review_model.py           # Conversation review wiring
├── test_cli_output.py             # --help output, --dry-run config
├── test_llm_utils*.py, test_utils.py
├── test_evaluator/                # Communicate-info scoring (semantic judge, substring)
├── test_runner/test_batch.py      # Batch loop: trials, checkpointing, resume, fatal errors
└── test_domains/
    ├── test_airline/              # Airline tools; pinned agent/user prompts; golden-reason answer key
    └── test_mock/                 # Mock domain tools
```

## Running Tests

```bash
make test                                   # whole suite, no API keys needed
uv run pytest tests/test_domains/test_airline/
uv run pytest tests/test_agent.py           # single file
uv run pytest --run-live-api -m live_api    # tests that call real models
```

Tests marked `@pytest.mark.live_api` call paid model APIs. They are skipped
unless `--run-live-api` is passed (`make test-live`).

## The mock domain

`tests/mock_domain/` exercises framework features the airline tasks do not use:
user tools, environment assertions, initial message histories, and
initialization data and actions. `conftest.py` puts `tests/` on `sys.path` and
registers it as domain `mock`. Fixtures such as `base_task`,
`task_with_message_history` and `task_with_user_tools` load its tasks.

## Writing Tests

- Prefer offline tests: monkeypatch `tau2.utils.llm_utils.completion`, or the
  `generate` imported by the module under test, instead of calling a model.
- Scripted replies must be created at call time: trajectories are ordered by
  message timestamp.
- Use the mock domain for framework behavior and the airline domain for
  airline-specific behavior (tools, policy, prompts).
- If you change the airline policy, tasks, user scaffold, personas or the
  agent's explanation or language clause on purpose, update the digests in
  `test_domains/test_airline/test_prompts_airline.py` and say why.
- `test_domains/test_airline/test_golden_reasons.py` checks that the answer
  key, `data/tau2-airline/mannual_label.json`, has an entry for every task.
