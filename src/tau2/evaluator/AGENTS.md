# AGENTS.md — src/tau2/evaluator/

## Overview

The evaluator module scores simulation runs. Each evaluator is a
classmethod-only class that takes a task and a message trajectory
(`list[Message]`) and returns a `RewardInfo`. User-facing documentation:
`docs/evaluation.md`.

## Architecture

### Evaluators

| File | Evaluator | What it checks | `RewardType` |
|------|-----------|---------------|-------------|
| `evaluator_env.py` | `EnvironmentEvaluator` | DB state after replaying the trajectory matches the DB after replaying the reference `actions`; `env_assertions` hold | `DB`, `ENV_ASSERTION` |
| `evaluator_action.py` | `ActionEvaluator` | Each reference action matches some agent or user tool call | `ACTION` |
| `evaluator_communicate.py` | `CommunicateEvaluator` | The agent conveyed each `communicate_info` fact | `COMMUNICATE` |
| `evaluator_nl_assertions.py` | `NLAssertionsEvaluator` | LLM-judged natural-language assertions (experimental) | `NL_ASSERTION` |

All subclass `EvaluatorBase` (`evaluator_base.py`).

### Composition via `evaluate_simulation()`

`evaluate_simulation(simulation, task, evaluation_type, domain, env_kwargs=None, strict_replay=True, *, communicate_scoring)`
in `evaluator.py` is the entry point. It:
1. Returns reward 0 if the simulation did not end with `AGENT_STOP` or `USER_STOP`.
2. Returns reward 1 if the task has no `evaluation_criteria`.
3. Runs the evaluators selected by `EvaluationType`.
4. Multiplies the component rewards (only those in the task's `reward_basis`,
   unless an `*_IGNORE_BASIS` type is used).

Callers: `tau2.runner.simulation.run_simulation()` (live runs, `strict_replay=True`)
and `tau2.scripts.evaluate_trajectories` (`tau2 evaluate-trajs`,
`strict_replay=False`, with the scorer recorded in the results file).

### `EvaluationType` Controls What Gets Evaluated

- `ALL` (the runner's and CLI's default): runs ENV + ACTION, plus COMMUNICATE
  and NL_ASSERTIONS when listed in the task's `reward_basis`; only listed
  components affect the final reward
- `ALL_WITH_NL_ASSERTIONS`: like `ALL` but always runs NL assertions
- `ALL_IGNORE_BASIS`: runs ENV, ACTION and COMMUNICATE and always multiplies them
- `ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS`: like `ALL_IGNORE_BASIS` plus NL assertions
- `ENV`, `ACTION`, `COMMUNICATE`, `NL_ASSERTIONS`: a single evaluator

## Rules When Modifying This Module

1. **`calculate_reward` is a classmethod**: no instance state.

2. **Reward is multiplicative**: the combined reward is the product of the
   component rewards. A 0 in any gating component zeroes the total.

3. **Handle missing criteria gracefully**: if `task.evaluation_criteria` is
   `None` or a criterion is absent or empty, return `reward=1.0` (not 0).
   Missing criteria mean "not evaluated", not "failed".

4. **Premature termination = 0 reward**: if `simulation.termination_reason`
   is not `AGENT_STOP` or `USER_STOP`, the simulation gets reward 0 before any
   evaluator runs.

5. **`evaluation_criteria.actions` is *one* reference trajectory, not a
   per-action requirement**: `EnvironmentEvaluator` replays `actions` on a
   fresh env to derive the target DB hash and compares it with the predicted
   DB hash. Other trajectories producing an equivalent end state also pass.
   The agent must match those specific calls only when `RewardType.ACTION` is
   in `reward_basis` (no airline task uses it). See `docs/evaluation.md`.

6. **The environment evaluator rebuilds environments**:
   `EnvironmentEvaluator.calculate_reward()` takes an `environment_constructor`
   and `env_kwargs` and builds two fresh environments (predicted and gold).
   The predicted one replays only state-mutating tool calls from the
   trajectory (`Environment.set_state`); with `strict_replay=True` a replayed
   call whose output differs from the recorded one raises. Always forward
   `env_kwargs` so custom constructor arguments (airline: `db`, `policy`)
   apply to both environments.

7. **Communication matching**: `CommunicateScoring` selects the scorer. The
   semantic LLM judge (default `openai/gpt-6-luna`, `reasoning_effort: medium`, prompt
   version `communicate-semantic-v1`) makes one `communicate_info_eval` call
   per expected fact over the user and agent text messages, and the scoring
   settings are recorded in `RewardInfo.info`. Substring matching
   (case-insensitive, commas stripped from agent messages) runs only when
   `scorer="substring"` is explicit. Missing or empty `communicate_info` makes
   no LLM calls. The judge prompts are calibrated; a wording change should
   come with a new `SemanticCommunicateSettings.prompt_version`.

8. **NL assertions require LLM calls**: `NLAssertionsEvaluator` calls
   `DEFAULT_LLM_NL_ASSERTIONS` from `config.py`. Experimental.

9. **Conversation review never affects reward**: `reviewer.py`
   (`review_simulation`, `ReviewMode`), `review_llm_judge.py`
   (`ConversationReviewer`), `review_llm_judge_user_only.py`
   (`UserOnlyReviewer`) and `auth_classifier.py` (`AuthenticationClassifier`)
   produce qualitative diagnostics (`SimulationRun.review`,
   `user_only_review`, `auth_classification`) for `tau2 run --auto-review`
   and `tau2 review`. They use `DEFAULT_LLM_REVIEW`
   (`openai/gpt-6-luna`, with `DEFAULT_LLM_REVIEW_ARGS`) unless `--review-model`
   is given. Keep them out of
   `evaluate_simulation()`.

## Tests

`tests/test_evaluator/` and `tests/test_evaluate_trajectories.py`.
