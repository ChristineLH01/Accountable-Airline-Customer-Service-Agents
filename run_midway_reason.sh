#!/usr/bin/env bash
# The midway experiment plus a reason score. Run from the repo root:
#   bash run_midway_reason.sh
# Default: 3 trials per task, saved as midway_reason_a and midway_reason_b.
# Change with: TRIALS=1 A_NAME=x B_NAME=y bash run_midway_reason.sh
# Steps 1-5 are run_midway.sh's (check key, fix the prompt join in Method B,
# run tests, run A and B on the 18 cancel tasks, print the summary table).
# Step 6 is the addition: grade_reasons.py grades the reason each agent gave
# the customer, with one judge and rubric for both A and B, and reports two
# scores with pass^k:
#   decision           DB x COMMUNICATE
#   decision x reason  DB x COMMUNICATE x REASON
# Tasks 9, 37 and 39 are run and graded; step 6 also reports both scores
# without them, since they are excluded from scoring (docs/golden_labels.md).
set -e

TASKS="0 1 7 9 19 24 26 28 35 37 39 41 43 44 45 47 48 49"
MODEL="openai/gpt-6.1-sol"
TRIALS="${TRIALS:-3}"
A_NAME="${A_NAME:-midway_reason_a}"
B_NAME="${B_NAME:-midway_reason_b}"

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY=your_key"
  exit 1
fi

echo "== 1. Fix the prompt join in Method B (safe to repeat) =="
python3 - <<'PY'
path = "src/tau2/agent/gated_cancel_agent.py"
text = open(path, encoding="utf-8").read()
old = '    AGENT_INSTRUCTION\n    + """\n\nCancellation checklist'
new = '    AGENT_INSTRUCTION\n    + "\\n\\n"\n    + """\nCancellation checklist'
if '+ "\\n\\n"' in text:
    print("Already fixed.")
elif old in text:
    open(path, "w", encoding="utf-8").write(text.replace(old, new, 1))
    print("Fixed.")
else:
    raise SystemExit("Could not find the line to fix. Stop and ask for help.")
PY

echo "== 2. Tests for Method B and the reason grader =="
uv run pytest tests/test_method_b tests/test_grade_reasons.py -q

# --reason-scoring off: tau2 run would otherwise gate its reward with its own
# built-in reason judge. Off keeps tau2's reward at DB x COMMUNICATE, as in the
# original midway runs; step 6 adds REASON on top.
echo "== 3. Run A (llm_agent) =="
uv run tau2 run --agent llm_agent --agent-llm "$MODEL" --user-llm "$MODEL" \
  --task-ids $TASKS --num-trials "$TRIALS" --max-concurrency 1 --retry-delay 60 --max-retries 5 \
  --reason-scoring off --save-to "$A_NAME" --verbose-logs --auto-resume

echo "== 4. Run B (gated_cancel_agent) =="
uv run tau2 run --agent gated_cancel_agent --agent-llm "$MODEL" --user-llm "$MODEL" \
  --task-ids $TASKS --num-trials "$TRIALS" --max-concurrency 1 --retry-delay 60 --max-retries 5 \
  --reason-scoring off --save-to "$B_NAME" --verbose-logs --auto-resume

echo "== 5. Summary =="
uv run python summarize_runs.py "$A_NAME" "$B_NAME"

echo "== 6. Reason score =="
uv run python grade_reasons.py "$A_NAME" "$B_NAME"
