"""Summarize one tau2 run: reward, checklist decisions and blocked cancels per task.

Usage, from the repo root:
    uv run python summarize_runs.py smoke_b_task19
    uv run python summarize_runs.py my_run_b my_run_a      # several runs
    uv run python summarize_runs.py passk_a passk_b,passk_b_retry
        # a comma joins folders into one run, for example a run and its retry

At the end it lists the trials that did not finish, so they can be rerun.

The run must have been made with --verbose-logs, because the checklist calls
of Method B are only kept in the llm_debug files, not in results.json.
Method A has no checklist, so its columns stay empty.
"""

import csv
import glob
import json
import os
import re
import sys
from math import comb

SIM_ROOT = os.path.join("data", "simulations")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def text_of(message):
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(str(x) for x in content)
    return content if isinstance(content, str) else json.dumps(content)


def scan_debug(sim_dir):
    """Read checklists and blocked calls from the longest agent_response log."""
    files = glob.glob(os.path.join(sim_dir, "llm_debug", "*agent_response*.json"))
    best = []
    for path in files:
        messages = load_json(path).get("request", {}).get("messages", [])
        if len(messages) >= len(best):
            best = messages

    checklists, blocked = [], 0
    for m in best:
        if m.get("role") != "tool":
            continue
        text = text_of(m)
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            data = None
        if isinstance(data, dict) and "decision" in data and "conditions" in data:
            checklists.append(data)
        elif "BLOCKED" in text.upper():
            blocked += 1
    return checklists, blocked


def reward_of(sim):
    info = sim.get("reward_info") or {}
    return info.get("reward")


def count_cancels(sim):
    n = 0
    for m in sim.get("messages", []):
        for call in m.get("tool_calls") or []:
            name = (call.get("function") or call).get("name")
            if name == "cancel_reservation":
                n += 1
    return n


def summarize(run_name):
    run_dir = os.path.join(SIM_ROOT, run_name)
    sims = load_json(os.path.join(run_dir, "results.json")).get("simulations", [])
    rows = []
    for sim in sims:
        task_id = str(sim.get("task_id"))
        sim_id = str(sim.get("id", ""))
        pattern = os.path.join(run_dir, "artifacts", f"task_{task_id}", "sim_*")
        dirs = glob.glob(pattern)
        match = [d for d in dirs if sim_id and sim_id in os.path.basename(d)]
        sim_dir = match[0] if match else (dirs[0] if len(dirs) == 1 else None)

        checklists, blocked = scan_debug(sim_dir) if sim_dir else ([], 0)
        decisions = " ".join(
            f"{c.get('reservation_id')}={c.get('decision')}" for c in checklists
        )
        rows.append(
            {
                "run": run_name,
                "task_id": task_id,
                "trial": sim.get("trial"),
                "reward": reward_of(sim),
                "cancel_calls": count_cancels(sim),
                "checklists": len(checklists),
                "blocked": blocked,
                "decisions": decisions,
                "debug_found": bool(sim_dir),
            }
        )
    return rows


def pass_hat_k(n, c, k):
    """tau-bench pass^k for one task: chance that k of the n trials all succeed."""
    if n < k:
        return None
    return comb(c, k) / comb(n, k)


def print_pass_k(rows):
    """Average reward and pass^k per run. Unfinished trials are left out."""
    print("\nrun | tasks | trials | avg_reward | pass^k for k = 1..trials")
    runs = {}
    for r in rows:
        if r["reward"] is None:
            continue
        runs.setdefault(r["run"], {}).setdefault(r["task_id"], []).append(
            float(r["reward"]) >= 1.0
        )
    for run, tasks in runs.items():
        n_min = min(len(v) for v in tasks.values())
        rewards = [x for v in tasks.values() for x in v]
        scores = []
        for k in range(1, n_min + 1):
            vals = [pass_hat_k(len(v), sum(v), k) for v in tasks.values()]
            scores.append(f"pass^{k}={sum(vals) / len(vals):.3f}")
        print(
            f"{run} | {len(tasks)} | {n_min} | {sum(rewards) / len(rewards):.3f} | "
            + " ".join(scores)
        )


def print_missing(rows):
    """List tasks with fewer finished trials than the run's largest trial count."""
    runs = {}
    for r in rows:
        tasks = runs.setdefault(r["run"], {})
        tasks.setdefault(r["task_id"], 0)
        if r["reward"] is not None:
            tasks[r["task_id"]] += 1
    print("\nUnfinished trials (task: how many to rerun)")
    for run, tasks in runs.items():
        target = max(tasks.values()) if tasks else 0
        gaps = {t: target - n for t, n in tasks.items() if n < target}
        if not gaps:
            print(f"{run}: none")
            continue
        print(f"{run}: " + ", ".join(f"{t}: {g}" for t, g in gaps.items()))


def main(names):
    all_rows = []
    for group in names:
        parts = group.split(",")
        for name in parts:
            for row in summarize(name):
                row["run"] = parts[0]
                all_rows.append(row)

    cols = [
        "run",
        "task_id",
        "trial",
        "reward",
        "cancel_calls",
        "checklists",
        "blocked",
        "decisions",
        "debug_found",
    ]
    print(" | ".join(cols))
    for r in all_rows:
        print(" | ".join(str(r[c]) for c in cols))

    print_pass_k(all_rows)
    print_missing(all_rows)

    out = re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(names)) + "_summary.csv"
    out_path = os.path.join(SIM_ROOT, out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: uv run python summarize_runs.py <run_name> [<run_name> ...]")
    main(sys.argv[1:])
