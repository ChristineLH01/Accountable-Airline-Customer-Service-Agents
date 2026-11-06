"""Summarize decision-reason scoring for one or more tau2 runs.

Prints one row per simulation (final reward, the task's own reward, each reward
component and the reason judge's verdict) and one row per run (averages and
accuracy), and saves two CSVs next to the runs:

- <runs>_reason_summary.csv   one row per simulation
- <runs>_reason_requests.csv  one row per customer request the judge graded,
                              with the golden and agent outcomes and the
                              judge's explanation, for error analysis

Usage, from the repo root:
    uv run python summarize_reasons.py reason_llm_agent reason_gated_cancel_agent
    uv run python summarize_reasons.py data/simulations/my-run/results.json

Runs are names under data/simulations/ or paths to a run directory or results
file. Runs made without reason scoring show empty reason columns; re-score them
with `tau2 evaluate-trajs <run> --reason-scoring gate -o <dir>`.
"""

import csv
import re
import sys
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.data_model.tasks import RewardType
from tau2.metrics.agent_metrics import compute_metrics

SIM_ROOT = Path("data") / "simulations"


def resolve(run: str) -> Path:
    """A run name or path -> what Results.load reads (results.json or a dir run)."""
    path = Path(run)
    if not path.exists():
        path = SIM_ROOT / run
    if path.is_dir() and not (path / "simulations").is_dir():
        return path / "results.json"  # monolithic results.json inside the run dir
    return path


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def simulation_rows(name: str, results: Results) -> tuple[list[dict], list[dict]]:
    sims, requests = [], []
    for sim in sorted(results.simulations, key=lambda s: (int(s.task_id), s.trial)):
        info = sim.reward_info
        check = info.reason_check if info else None
        breakdown = (info.reward_breakdown or {}) if info else {}
        reason_info = ((info.info or {}).get("reason") or {}) if info else {}
        sims.append(
            {
                "run": name,
                "task_id": sim.task_id,
                "trial": sim.trial,
                "termination": sim.termination_reason.value,
                "reward": info.reward if info else None,
                "task_reward": (
                    info.task_reward
                    if info and info.task_reward is not None
                    else info.reward
                    if info
                    else None
                ),
                "DB": breakdown.get(RewardType.DB),
                "COMMUNICATE": breakdown.get(RewardType.COMMUNICATE),
                "REASON": breakdown.get(RewardType.REASON),
                "explanation_ok": check.met if check else None,
                "decision_ok": check.decision_correct if check else None,
                "reason_ok": check.reason_correct if check else None,
                "partial": check.partial_score if check else None,
                "incorrect_claims": len(check.incorrect_claims) if check else None,
                "reason_note": (
                    check.justification if check else reason_info.get("skipped", "")
                ),
            }
        )
        for i, request in enumerate(check.requests if check else []):
            requests.append(
                {
                    "run": name,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "request_index": i,
                    "request": request.request,
                    "golden_outcome": request.golden_outcome,
                    "agent_outcome": request.agent_outcome,
                    "decision_correct": request.decision_correct,
                    "reason_correct": request.reason_correct,
                    "evidence": request.evidence,
                    "judge_reasoning": request.reasoning,
                    "incorrect_claims": " | ".join(check.incorrect_claims),
                }
            )
    return sims, requests


def run_row(name: str, results: Results) -> dict:
    metrics = compute_metrics(results)
    scoring = results.info.effective_reason_scoring
    return {
        "run": name,
        "agent": results.info.agent_info.implementation,
        "reason_scoring": scoring.mode.value,
        "sims": metrics.total_simulations,
        "avg_reward": metrics.avg_reward,
        "avg_task_reward": metrics.avg_task_reward,
        "explanation_ok": (
            f"{metrics.reason_met}/{metrics.reason_judged}"
            if metrics.reason_judged
            else None
        ),
        "decision_ok": metrics.reason_decision_correct
        if metrics.reason_judged
        else None,
        "reason_ok": metrics.reason_reason_correct if metrics.reason_judged else None,
        "avg_partial": metrics.avg_reason_partial,
        "skipped": metrics.reason_skipped,
        "judge_cost": metrics.reason_judge_cost,
        "infra_errors": metrics.infra_error_count,
    }


def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in columns))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main(names: list[str]) -> None:
    all_sims, all_requests, runs = [], [], []
    for run in names:
        path = resolve(run)
        name = path.parent.name if path.is_file() else path.name  # the run folder
        results = Results.load(path)
        sims, requests = simulation_rows(name, results)
        all_sims += sims
        all_requests += requests
        runs.append(run_row(name, results))

    sim_columns = [c for c in all_sims[0] if c != "reason_note"] if all_sims else []
    print("Per simulation (reward = final score; task_reward = DB x COMMUNICATE)\n")
    if all_sims:
        print_table(all_sims, sim_columns)
    print("\nPer run\n")
    print_table(runs, list(runs[0]))

    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(r["run"] for r in runs))
    print()
    write_csv(SIM_ROOT / f"{stem}_reason_summary.csv", all_sims)
    write_csv(SIM_ROOT / f"{stem}_reason_requests.csv", all_requests)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: uv run python summarize_reasons.py <run> [<run> ...]")
    main(sys.argv[1:])
