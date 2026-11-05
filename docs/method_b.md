# Method B: cancellation checklist and gate

Method B is the baseline `llm_agent` (method A) with two additions, for
cancellation only.

1. **A checklist tool, `check_eligibility`.** Before cancelling, the agent fills
   the policy conditions from tool results: cancel reason, already flown, booked
   within 24 hours, cancelled by the airline, business cabin, insurance. The
   decision (`ALLOW`, `DENY`, `TRANSFER`, `NEED_INFO`) is computed by fixed rules
   in `src/tau2/agent/eligibility.py`, not by the LLM.
2. **A gate on `cancel_reservation`.** If the agent tries to cancel a reservation
   whose latest checklist is not `ALLOW`, the call never reaches the environment.
   The agent gets a `BLOCKED` message instead and has to answer the user.

The checklist tool is handled inside the agent, so the environment, the policy,
the tasks and the scoring are the same as for method A.

## Files

| File | What it is |
|------|------------|
| `src/tau2/agent/eligibility.py` | Checklist rules and the gate. No LLM. |
| `src/tau2/agent/gated_cancel_agent.py` | The agent, registered as `gated_cancel_agent`. |
| `tests/test_method_b/` | Tests. The checklist is checked against all 31 cancel rows of our gold labels. |

## Run

```bash
uv run pytest tests/test_method_b          # no API key needed

# A vs B on the 15 scored tasks that involve a cancellation (9, 37, 39 excluded)
TASKS="0 1 7 19 24 26 28 35 41 43 44 45 47 48 49"
uv run tau2 run --agent llm_agent          --task-ids $TASKS --num-trials 1 --save-to cancel-A
uv run tau2 run --agent gated_cancel_agent --task-ids $TASKS --num-trials 1 --save-to cancel-B
```

Use the same `--agent-llm` and `--user-llm` for both runs so the results are
comparable.
