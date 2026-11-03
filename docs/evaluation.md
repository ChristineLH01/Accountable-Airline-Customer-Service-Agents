# Task Schema and Evaluation

This page explains how a simulation is scored and, most importantly,
**what `evaluation_criteria.actions` actually does**. If you have looked at
`data/tau2-airline/tasks.json` and assumed the listed actions are
required of the agent, this page is for you.

## TL;DR

- A task's final reward is the **product** of the components listed in
  `evaluation_criteria.reward_basis`.
- The schema defaults `reward_basis` to `["DB", "COMMUNICATE"]`, and all 50
  airline tasks use exactly that basis.
- `evaluation_criteria.actions` is **one reference trajectory** that
  solves the task, not the only correct one. It is replayed on a fresh
  "gold" environment to derive a target DB end state, which is then
  compared (by hash) with the DB end state the agent produced. The agent
  is **not** required to take this path; any sequence of tool calls that
  produces an equivalent DB end state passes the DB check.
- The agent must match `actions` call by call *only* when `RewardType.ACTION`
  is in `reward_basis`. No airline task does this.
- A simulation that does not end with the agent or user stopping (for
  example it hits `--max-steps`, `--timeout` or `--max-errors`) gets
  reward `0.0` before any criterion is checked.
- This was a deliberate design choice (see [issue #224][issue-224] and
  [RFC #129][issue-129] for the discussion).

[issue-224]: https://github.com/sierra-research/tau2-bench/issues/224
[issue-129]: https://github.com/sierra-research/tau2-bench/issues/129

## The task schema, in plain English

A task's `evaluation_criteria` (see
[`src/tau2/data_model/tasks.py`](../src/tau2/data_model/tasks.py)) has five
fields:

| Field | What it is | When it gates the reward |
|-------|-----------|--------------------------|
| `actions` | A list of `Action`s recording **one** reference trajectory of tool calls that solves the task. Always replayed on a fresh "gold" environment to derive the target DB end state. Other trajectories may produce an equivalent end state and also pass. | Only when `RewardType.ACTION` is in `reward_basis` (then each entry must also appear among the agent's tool calls, i.e. the list is treated as the only acceptable trajectory). |
| `env_assertions` | Assertions run against the predicted environment after the simulation. | Only when `RewardType.ENV_ASSERTION` is in `reward_basis`. |
| `communicate_info` | Facts the agent must convey to the user. | Only when `RewardType.COMMUNICATE` is in `reward_basis`. |
| `nl_assertions` | Natural-language assertions judged by an LLM. | Only when `RewardType.NL_ASSERTION` is in `reward_basis` (experimental). |
| `reward_basis` | The list of `RewardType`s that gate the final reward. | Always. Default is `[DB, COMMUNICATE]`. |

The reward components are evaluated by the classes in
[`src/tau2/evaluator/`](../src/tau2/evaluator/):

| `RewardType` | Evaluator | What it checks |
|--------------|-----------|----------------|
| `DB` | `EnvironmentEvaluator` | Does the predicted DB hash match the target DB hash? Predicted = fresh env + the task's initial state + a replay of the simulation's state-mutating tool calls (`mutates_state` in `is_tool`; WRITE tools by default). Target = fresh env + the task's initial state + the reference `evaluation_criteria.actions`. |
| `ENV_ASSERTION` | `EnvironmentEvaluator` | Do all `env_assertions` pass on the predicted environment? |
| `COMMUNICATE` | `CommunicateEvaluator` | Was every `communicate_info` fact conveyed? See [Communication checks](#communication-checks). |
| `NL_ASSERTION` | `NLAssertionsEvaluator` | Does an LLM judge (`DEFAULT_LLM_NL_ASSERTIONS` in `config.py`, default `openai/gpt-6-luna` at medium reasoning effort) return true for every entry in `nl_assertions`? |
| `ACTION` | `ActionEvaluator` | For every entry in `actions`, did the agent (or user) make a matching tool call (per `Action.compare_with_tool_call`, which compares only `compare_args` when set)? |

The final reward is the product. With `reward_basis = [DB, COMMUNICATE]`
the reward is `db_reward * communicate_reward`. `actions` is consumed by
`EnvironmentEvaluator` to set up the target environment, but the agent's
tool-call trajectory is never directly compared to it.

## Evaluation types

`evaluate_simulation()` in
[`src/tau2/evaluator/evaluator.py`](../src/tau2/evaluator/evaluator.py)
takes an `EvaluationType`:

| `EvaluationType` | What runs | What gates the reward |
|------------------|-----------|-----------------------|
| `ALL` (used by `tau2 run` and `tau2 evaluate-trajs`) | ENV and ACTION always; COMMUNICATE and NL assertions only when in `reward_basis` | Only the components in `reward_basis` |
| `ALL_WITH_NL_ASSERTIONS` | Like `ALL`, but NL assertions always run | Only the components in `reward_basis` |
| `ALL_IGNORE_BASIS` | ENV, ACTION and COMMUNICATE | All of them, whatever `reward_basis` says |
| `ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS` | ENV, ACTION, COMMUNICATE and NL assertions | All of them |
| `ENV`, `COMMUNICATE`, `ACTION`, `NL_ASSERTIONS` | One evaluator | That evaluator |

The CLI always uses `ALL`; the other types are available from Python, for
example `run_tasks(..., evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS)`.
A missing `evaluation_criteria` gives reward `1.0`, and a missing or empty
criterion is neutral (reward `1.0` for that component).

## Communication checks

`CommunicateEvaluator` supports two scorers, selected by
`CommunicateScoring` (`RunConfig.communicate_scoring`, or
`--communicate-scorer` on the CLI):

- **`semantic` (default).** An LLM judge decides, once per expected fact,
  whether the agent conveyed its meaning anywhere in the conversation;
  paraphrases and different number or date formats count. The judge reads
  the text messages of both the agent and the user. The default judge is
  `openai/gpt-6-luna` with `reasoning_effort: medium` (`DEFAULT_SEMANTIC_COMMUNICATE_MODEL`
  in `config.py`); override it with
  `--communicate-judge '{"model": "...", "model_args": {...}}'`.
- **`substring`.** The original τ-bench check: each fact must appear,
  case-insensitively, in some agent message with commas removed.

The scorer used is recorded in `results.json` (`info.communicate_scoring`),
and `tau2 evaluate-trajs` re-grades with that recorded scorer.

Under `ALL`, communication is checked only when `COMMUNICATE` is in the
task's `reward_basis`; otherwise `communicate_checks` is `null` and no
judge calls are made. A missing, `null` or empty `communicate_info` is
neutral and makes no judge calls. Communication success cannot compensate
for a failed DB check when both gate the reward.

## Why `actions` looks like a requirement (and isn't)

`evaluation_criteria.actions` records *one* sequence of tool calls that
solves the task, typically how a task author solved it. There is generally
no claim that this is the only correct sequence; often several
trajectories produce an equivalent DB end state (for example looking up
the user before or after the reservation, or skipping a read-only lookup
the agent does not need). The reference trajectory is kept because the
target DB state is easier to express as "play these actions on a fresh
env" than to spell out by hand.

Because the list reads like a checklist, it is easy to misread as a list
of *required* agent actions. For scoring, the only things that matter are
whether the predicted DB end state matches the target and whether the
required facts were communicated. An agent that takes a different correct
path through the tools, or makes no tool calls at all when the right
answer is to refuse, can still receive full reward.

This is **intended**: the original τ-bench paper scores outcomes
(`DB + COMMUNICATE`), not adherence to a script. Gating the reward on
`actions` would break parity with those numbers and penalize
correct-but-different solutions.

## A worked example: airline task `1`

From [`data/tau2-airline/tasks.json`](../data/tau2-airline/tasks.json):

```json
{
  "id": "1",
  "evaluation_criteria": {
    "actions": [
      {
        "action_id": "1_0",
        "name": "get_user_details",
        "arguments": { "user_id": "raj_sanchez_7340" }
      },
      {
        "action_id": "1_1",
        "name": "get_reservation_details",
        "arguments": { "reservation_id": "Q69X3R" }
      }
    ],
    "communicate_info": [],
    "nl_assertions": ["Agent should not approve the cancellation."],
    "reward_basis": ["DB", "COMMUNICATE"]
  }
}
```

What this means in practice:

- `reward_basis = ["DB", "COMMUNICATE"]`, so reward = `db_reward * communicate_reward`.
- Both `actions` are **read-only** tools (`ToolType.READ` in
  `src/tau2/domains/airline/tools.py`), so replaying them leaves the DB
  unchanged and the target DB hash equals the *initial* DB hash.
- `db_reward = 1.0` iff the agent made no DB writes. The correct behavior
  is to refuse the cancellation, so a correct agent passes.
- `communicate_info = []`, so `communicate_reward = 1.0` (nothing must be
  said, and no judge call is made).
- `NL_ASSERTION` is not in `reward_basis`, so under `ALL` the assertion is
  not evaluated at all. Under `ALL_WITH_NL_ASSERTIONS` it runs as a
  diagnostic only.

Net effect: an agent that politely refuses receives full reward `1.0`
even if it never calls `get_user_details` or `get_reservation_details`.
The listed actions document one valid information-gathering path; they
are not a requirement.

## How to inspect action correctness anyway

Even though `ACTION` is not in the airline `reward_basis`, `ActionEvaluator`
runs under `ALL`, so every simulation's `reward_info.action_checks` is
populated:

- `RewardInfo.partial_action_reward` summarizes how many reference actions
  were matched, overall and split by `ToolType.READ` / `ToolType.WRITE`.
  `tau2 view` shows it as `Partial Action Reward: m/n`, and
  [`src/tau2/metrics/agent_metrics.py`](../src/tau2/metrics/agent_metrics.py)
  aggregates read/write action matches. This is a similarity signal against
  *one* reference trajectory, not a correctness verdict: an agent can score
  `0/n` and still be fully correct.
- For a single "did the agent reproduce this reference trajectory?" number,
  use `EvaluationType.ALL_IGNORE_BASIS` (or
  `ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS`), which multiplies ENV, ACTION and
  COMMUNICATE (and optionally NL) regardless of `reward_basis`. Same caveat
  applies.

The reported reward always uses the task's `reward_basis`; these
diagnostics do not change it.

## Using `RewardType.ACTION` in your own tasks

If you write tasks where the agent's tool calls should be a hard
requirement (not just a side effect on the DB), put `RewardType.ACTION`
in the task's `reward_basis`. This promotes `actions` from "one reference
trajectory" to "the only acceptable trajectory", so reserve it for tasks
where you have enumerated all valid solutions, or where there genuinely is
only one.

## Pointers

- Schema definitions: [`src/tau2/data_model/tasks.py`](../src/tau2/data_model/tasks.py)
- Evaluator source: [`src/tau2/evaluator/`](../src/tau2/evaluator/)
- Evaluator architecture overview: [`src/tau2/evaluator/AGENTS.md`](../src/tau2/evaluator/AGENTS.md)
- Reward shape (`RewardInfo`, `ActionCheck`, etc.):
  [`src/tau2/data_model/simulation.py`](../src/tau2/data_model/simulation.py)
- Discussion of the `actions`-vs-`reward_basis` confusion:
  [issue #224][issue-224] and [RFC #129][issue-129].
