#!/usr/bin/env bash
# Decision-reason evaluation. Runs each agent with the reason judge scoring the
# agent's decision explanations against the golden labels, so the final reward
# is DB x COMMUNICATE x REASON, then prints per-task and per-run summaries.
#
# Run from the repo root:
#   bash run_reason_eval.sh                        # 47 scored tasks, agents A and B
#   TASKS="0 1 7 19" bash run_reason_eval.sh       # quick smoke run on 4 tasks
#   AGENTS="llm_agent" NUM_TRIALS=4 bash run_reason_eval.sh
#   REASON_SCORING=report bash run_reason_eval.sh  # judge and log, reward unchanged
#
# Settings (environment variables):
#   MODEL           agent and user-simulator model    (openai/gpt-6.1-sol)
#   AGENTS          agents to run, space separated     (llm_agent gated_cancel_agent)
#   TASKS           task ids; empty = the scored split (empty: tasks 9, 37, 39 left out)
#   NUM_TRIALS      trials per task; 2+ gives pass^k   (1)
#   CONCURRENCY     simultaneous conversations         (1)
#   REASON_SCORING  gate | report | off                (gate)
#   PREFIX          save names are <PREFIX>_<agent>    (reason)
#
# Each run saves to data/simulations/<PREFIX>_<agent>/ with --verbose-logs, so
# every judge call is also logged under artifacts/.../llm_debug/*reason_eval*.
# Reruns resume where they stopped (--auto-resume); changing REASON_SCORING or
# the golden labels for an existing save name is refused, so use a new PREFIX.
set -euo pipefail

MODEL="${MODEL:-openai/gpt-6.1-sol}"
AGENTS="${AGENTS:-llm_agent gated_cancel_agent}"
TASKS="${TASKS:-}"
NUM_TRIALS="${NUM_TRIALS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
REASON_SCORING="${REASON_SCORING:-gate}"
PREFIX="${PREFIX:-reason}"

# The key may be exported or in .env.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ] || [ "${OPENROUTER_API_KEY}" = "<your_key_here>" ]; then
  echo "OPENROUTER_API_KEY is not set. Put it in .env or run: export OPENROUTER_API_KEY=your_key"
  exit 1
fi

if [ -n "$TASKS" ]; then
  TASK_ARGS=(--task-ids $TASKS)
  for t in $TASKS; do
    case "$t" in
      9 | 37 | 39) echo "Note: task $t is excluded from reason scoring; only its task reward counts." ;;
    esac
  done
else
  TASK_ARGS=(--task-split-name scored)
fi

echo "== 1. Offline tests for reason scoring and the golden labels =="
uv run pytest tests/test_evaluator/test_reason_judge.py \
  tests/test_domains/test_airline/test_golden_reasons.py -q

echo "== 2. Reason scoring this run will use =="
uv run tau2 run --reason-scoring "$REASON_SCORING" --dry-run \
  | uv run python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['reason_scoring'], indent=2))"

RUNS=()
STEP=3
for AGENT in $AGENTS; do
  SAVE="${PREFIX}_${AGENT}"
  echo "== ${STEP}. Run ${AGENT} -> data/simulations/${SAVE} =="
  uv run tau2 run --agent "$AGENT" --agent-llm "$MODEL" --user-llm "$MODEL" \
    "${TASK_ARGS[@]}" --num-trials "$NUM_TRIALS" --max-concurrency "$CONCURRENCY" \
    --retry-delay 60 --max-retries 5 --reason-scoring "$REASON_SCORING" \
    --save-to "$SAVE" --verbose-logs --auto-resume
  RUNS+=("$SAVE")
  STEP=$((STEP + 1))
done

echo "== ${STEP}. Summary =="
uv run python summarize_reasons.py "${RUNS[@]}"

echo
echo "Browse a run:          uv run tau2 view --file data/simulations/${RUNS[0]}"
echo "Re-score an older run: uv run tau2 evaluate-trajs data/simulations/midway_a --reason-scoring gate -o data/simulations/rescored"
