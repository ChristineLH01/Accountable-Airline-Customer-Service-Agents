# Task Schema and Evaluation

This page explains how a simulation is scored and, most importantly,
**what `evaluation_criteria.actions` actually does**. If you have looked at
`data/tau2-airline/tasks.json` and assumed the listed actions are
required of the agent, this page is for you.

## TL;DR

- A task's final reward is the **product** of the components listed in
  `evaluation_criteria.reward_basis`, times **`REASON`** when the run scores
  decision explanations with `--reason-scoring gate` (the default). See
  [Decision-reason checks](#decision-reason-checks-reason).
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
| `REASON` | `ReasonEvaluator` | Did the agent tell the customer the task's golden decision and its golden policy-based reason? Added by the run, not by `reward_basis`. See [Decision-reason checks](#decision-reason-checks-reason). |

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

## Decision-reason checks (`REASON`)

The project's agent prompt asks the agent to tell the customer its decision
and the policy-based reason whenever it approves, denies or partly approves a
request. `ReasonEvaluator`
([`evaluator_reason.py`](../src/tau2/evaluator/evaluator_reason.py)) checks
that explanation against the task's final `golden_decision` and
`golden_reason` in
[`data/tau2-airline/mannual_label.json`](../data/tau2-airline/mannual_label.json)
(see [golden_labels.md](golden_labels.md)).

**Modes** (`RunConfig.reason_scoring`, `--reason-scoring` on the CLI):

| Mode | Judge runs | Final reward |
|---|---|---|
| `gate` (default) | yes | `task reward x REASON`, e.g. `DB x COMMUNICATE x REASON` |
| `report` | yes | unchanged; the verdict is only recorded |
| `off` | no | unchanged |

**The judge.** One LLM call per simulation (`call_name="reason_eval"`,
default `openai/gpt-6-luna` at medium reasoning effort, prompt version
`reason-judge-v1`; override with `--reason-judge '{"model": "...",
"model_args": {...}}'`). It sees the airline policy, the golden decision, the golden
reason and a transcript of the customer and agent messages plus the agent's
tool calls. Tool results are left out, so the judge grades against the golden
facts, not its own reading of the database. The policy is there only so the
judge can tell whether a rule the agent cites exists; decisions and reasons are
graded against the golden label. It splits the golden decision into the
customer's requests and, for each one, decides:

- `decision_correct`: the agent told the customer the golden outcome
  (approved, denied or partly approved, with the same scope).
- `reason_correct`: the agent's reason rests on the same deciding policy rule
  and facts as the golden reason. Paraphrase is fine and secondary details may
  be left out. When a request is denied because no allowing condition holds,
  the agent must make clear that none applies and give the facts behind the
  conditions that matter for this customer (the reservation's cabin, insurance
  and booking time, and any condition the customer claimed). The agent must not
  accept a false claim the golden reason corrects, and any number it gives must
  agree with the golden reason.

It also lists `incorrect_claims`: wrong facts or rules the agent told the
customer, even next to a correct reason.

**The pass rule is in code, not in the judge.** `REASON = 1` only if every
request's decision and reason are correct and there are no incorrect claims
(`ReasonCheck.from_reply`). `partial_score` is the share of requests that are
fully correct; it is recorded for analysis and does not affect the reward.

**Skipped tasks.** Tasks without a final label, and tasks marked
`excluded_from_scoring` (9, 37 and 39), are not judged. They get no `REASON`
component and the skip reason is recorded in `reward_info.info["reason"]`.
Abnormally ended simulations score 0 before any judging.

**What is recorded.**

- Per simulation, `reward_info` has:
  - `reason_check`: the golden decision and reason used, the per-request
    verdicts with evidence quotes and explanations, the incorrect claims, the
    judge model, the prompt version and the judge's cost.
  - `task_reward`: the reward from the task's `reward_basis` alone, i.e. the
    original τ-bench reward before `REASON`.
  - `reward_breakdown["REASON"]`, when gating.
- Per run, `results.json` `info` records `reason_scoring` and
  `golden_labels_sha256`, the digest of the label file. Resuming a run with a
  different scorer or changed labels is refused.
- With `--verbose-logs`, each judge call is saved under
  `artifacts/task_<id>/sim_<id>/llm_debug/` with `reason_eval` in its name.

The metrics shown after a run, in `tau2 view` and by `tau2 evaluate-trajs`
add a *Decision Explanations* section: decision-and-reason accuracy, decision
and reason accuracy separately, the average partial score, skipped tasks and
the judge cost. When `REASON` gates the reward, the average task reward is
shown next to the final reward.

**Re-scoring old runs.** Runs recorded before reason scoring existed read as
`off`. Re-score them without re-running the conversations:

```bash
uv run tau2 evaluate-trajs data/simulations/midway_a --reason-scoring gate \
    -o data/simulations/rescored_midway_a
```

**Summaries.** `uv run python summarize_reasons.py <run> [<run> ...]` prints
per-simulation and per-run tables. It saves
`<runs>_reason_summary.csv` (one row per simulation) and
`<runs>_reason_requests.csv` (one row per judged customer request, for error
analysis) in `data/simulations/`. `run_reason_eval.sh` runs agents with reason
scoring and prints this summary.

**Changing the judge.** The prompt is pinned by a digest in
`tests/test_evaluator/test_reason_judge.py`. Change it together with
`ReasonJudgeSettings.prompt_version`, because scores from different prompts are
not comparable. `tests/test_evaluator/test_reason_judge_live.py` runs the real
judge on eight hand-written replies for tasks 1 and 19, correct and incorrect
(`make test-live`). Run it after any prompt or judge-model change. The judge is
an LLM, so a single verdict can vary between calls; compare agents over several
trials (`--num-trials`) rather than one run.

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
