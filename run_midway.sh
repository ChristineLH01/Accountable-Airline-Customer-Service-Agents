#!/usr/bin/env bash
# One command for the midway experiment. Run from the repo root:
#   bash run_midway.sh
# Steps: check key, fix the prompt join in Method B, run tests,
# run A and B on the 15 scored cancel tasks, then print the summary table.
set -e

TASKS="0 1 7 19 24 26 28 35 41 43 44 45 47 48 49"
MODEL="openai/gpt-6.1-sol"

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

echo "== 2. Tests for Method B =="
uv run pytest tests/test_method_b -q

echo "== 3. Run A (llm_agent) =="
uv run tau2 run --agent llm_agent --agent-llm "$MODEL" --user-llm "$MODEL" \
  --task-ids $TASKS --num-trials 1 --max-concurrency 1 --retry-delay 60 --max-retries 5 \
  --save-to midway_a --verbose-logs --auto-resume

echo "== 4. Run B (gated_cancel_agent) =="
uv run tau2 run --agent gated_cancel_agent --agent-llm "$MODEL" --user-llm "$MODEL" \
  --task-ids $TASKS --num-trials 1 --max-concurrency 1 --retry-delay 60 --max-retries 5 \
  --save-to midway_b --verbose-logs --auto-resume

echo "== 5. Summary =="
uv run python summarize_runs.py midway_a midway_b
