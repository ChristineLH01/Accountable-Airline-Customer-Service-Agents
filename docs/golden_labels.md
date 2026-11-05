# Final golden labels: how they were set

These labels are what the decision-reason judge (`REASON`, see
[evaluation.md](evaluation.md#decision-reason-checks-reason)) grades the agent's
explanations against. Every reason-scored run records the label file's SHA-256
in `info.golden_labels_sha256`. Changing a label therefore changes the scores,
and a run cannot be resumed across a label change.

`data/tau2-airline/mannual_label.json` now has a final `golden_decision` and
`golden_reason` for all 50 airline tasks. They were built from the three
annotators' labels (`golden_decision_1/2/3`, `golden_reason_1/2/3`, kept in the
file unchanged) and checked against the task, the database and the policy.

## Method

For each task:

1. List every request the customer makes in `user_scenario`.
2. Work out the correct outcome from `policy.md` and the facts in `db.json`:
   cabin, membership, insurance, booking time against 2024-05-15 15:00 EST,
   flight status, passenger count and prices. Fare differences follow the
   tool's pricing: a kept flight keeps its original price only when the cabin
   does not change.
3. Compare that outcome with the task's answer key (`actions`, `nl_assertions`,
   `communicate_info`) and with each annotator's decision and reason.
4. Write one final label, reusing the most accurate annotator's wording, which
   was usually annotator 3's.

Rules for the final labels:

- **`golden_decision`** states the outcome for every customer request, with
  reservation IDs, flights and amounts.
- **`golden_reason`** is the policy-based reason the agent should give the
  customer, citing the facts it rests on. It contains no process commentary,
  no guesses about the customer's motives and no mention of the benchmark.
- When a customer's claim is false (for example, gold status when they are
  silver), the reason states the true fact.
- Each label matches the answer key's end state, except in task 39 (see below).

## Excluded from scoring: tasks 9, 37 and 39

These three tasks are left out of scoring and evaluation runs until the
problems below are settled. In `mannual_label.json` they carry
`excluded_from_scoring: true` and an `exclusion_reason`. They still have
labels, which reflect the best reading of the policy. Use the `scored` task
split (47 tasks) for runs:

```bash
uv run tau2 run --task-split-name scored ...
```

`run_midway.sh` and `docs/method_b.md` use the 15 cancellation tasks that remain
after these exclusions. The `train` and `test` splits are unchanged and still
include them: task 9 is in `train`, task 37 in `test` and task 39 in `train`.

## Agreement between annotators

| | Tasks |
|---|---|
| All three agree on the decision | 44 of 50 |
| Partial agreement on the decision | 7, 9, 18, 37, 44 |
| Disagreement on the decision | 39 |
| Reasons only partly agree | 2, 4, 7, 9, 10, 13, 14, 18, 23, 25, 29, 31, 37, 38, 44 |

The partial agreements on the decision come from one annotator leaving out part
of a multi-request task, or adding a transfer, not from opposite verdicts.

## Answer-key problems

**Task 39: the answer key breaks the policy.** The answer key cancels
MSJ4OA. MSJ4OA is economy, was booked on 2024-05-05, and has no
airline-cancelled flight. It is insured, but the customer's reason (giving the
seat to someone else) is not health or weather. No cancellation condition
applies. Annotators 1 and 3 deny it; annotator 2 follows the answer key but
questions it.

The final label **follows the policy: deny MSJ4OA.** No policy-based reason
exists for cancelling it, so a label that cancels it could not have a valid
`golden_reason`. As a result, the label and the DB check disagree on this task:
an agent that matches the label fails the DB check. Task 39 is excluded from
scoring for now.

**Task 44: the condition in the task is ambiguous.** The customer wants
upgrades for flights "under or equal to 3 hours (including layovers)".
KC18K6 (MSP→EWR→CLT, 06:00–13:00 on May 21) takes 7 hours including a 2-hour
layover, but each leg is 3 hours or less. The answer key upgrades it, and the
required $1,380–1,390 total only works if it is included. The label follows the
answer key.

**Tasks 9, 13 and 37: transfer or not.** In tasks 9 and 37, reservation NQNU5R
is fully flown. The policy says to transfer when any part has been flown, but
the answer key only declines to cancel it. Annotator 1 adds a transfer. The
labels deny without a transfer, which matches the answer key and the
`nl_assertion` ("Flights that have already departed cannot be cancelled").
Task 13's answer key does transfer, because the customer asks for one, and its
label keeps the transfer. A transfer does not change the DB. Tasks 9 and 37 are
excluded from scoring for now; task 13 stays in.

**Task 31: a cheaper two-step route exists.** The label (no change) relies on
the $301 cost of upgrading DF89BM to economy, which is over the customer's $100
limit. Upgrading and then switching to the nonstop HAT033 would cost only $81
net, but the answer key expects no change. Task 32 uses the same two-step route
and the answer key accepts it there.

## Annotator issues found

None of the three annotators made a factual error against the database except
the following:

- **Task 10, annotator 1:** gives the full upgrade cost as "about $1,815". It
  is $2,010: $1,197 per passenger in business against $527 paid, for 3
  passengers.
- **Task 29, annotator 1:** says HAT169 and HAT033 are "the direct flights that
  land before 7am". HAT092, HAT088 and HAT212 also do; HAT033 is the cheapest.
- **Task 25, annotator 2:** cites the rule that a reservation's passenger count
  cannot change, but the customer is making a new booking, so that rule does
  not apply.

Recurring issues that were fixed in the final labels:

- **Annotator 1** often writes narrative or commentary instead of a reason (for
  example about supervisors in tasks 3 and 4, or the customer's preferences in
  tasks 44 and 45). It often hedges or drops dollar amounts (tasks 7, 14,
  15–18, 22) and leaves out required totals (task 7: $1,628; task 18: $23,553).
- **Annotator 2** sometimes writes process steps or benchmark commentary instead
  of an outcome ("Reference verdict…", "requires policy review"; tasks 2, 5,
  31, 32, 38, 39, 41, 44).
- **Annotator 3** is the most accurate and complete, with only small omissions
  (for example in tasks 23 and 43).

## Confidence

- **High:** 45 tasks.
- **Medium:** 9, 29, 31 and 44, for the reasons above. In task 29 the
  cancellation rests on a health reason the policy takes at face value.
- **Low:** 39, because of the answer-key conflict.
