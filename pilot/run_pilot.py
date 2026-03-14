"""Run narrow pilot: baseline vs metaslayer_v0 vs metaslayer_v1 vs metaslayer_v1_soft."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from baseline_agent import BaselineAgent
from metaslayer_v0 import MetaLayerAgentV0
from metaslayer_v1 import MetaLayerAgentV1, MetaLayerAgentV1Soft


def generate_tasks(n: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    tasks: list[tuple[float, float]] = []
    for i in range(n):
        # Base complexity
        complexity = rng.uniform(0.65, 1.25)

        # Periodic stress windows
        if 10 <= i % 30 <= 18:
            complexity += rng.uniform(0.15, 0.35)

        noise = rng.uniform(-0.1, 0.3)
        tasks.append((complexity, noise))
    return tasks


def evaluate(agent, tasks: list[tuple[float, float]]) -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    solved_count = 0
    failed_count = 0
    total_time = 0.0

    for i, (complexity, noise) in enumerate(tasks):
        out = agent.solve(complexity, noise)
        solved = bool(out["solved"])
        failed = bool(out["failed"])
        task_time = float(out["time"])

        solved_count += 1 if solved else 0
        failed_count += 1 if failed else 0
        total_time += task_time

        row = {
            "step": i,
            "complexity": complexity,
            "noise": noise,
            "solved": int(solved),
            "failed": int(failed),
            "time": task_time,
            "overload": float(out.get("overload", 0.0)),
            "crisis_mode": int(bool(out.get("crisis_mode", False))),
            "avg_crisis": float(out.get("avg_crisis", 0.0)),
            "I": float(out.get("I", 0.0)),
            "C": float(out.get("C", 0.0)),
            "Ccrit": float(out.get("Ccrit", 0.0)),
        }
        rows.append(row)

    n = len(tasks)
    metrics = {
        "success_rate": solved_count / n,
        "failure_rate": failed_count / n,
        "avg_time": total_time / n,
    }
    return rows, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "step",
        "complexity",
        "noise",
        "solved",
        "failed",
        "time",
        "overload",
        "crisis_mode",
        "avg_crisis",
        "I",
        "C",
        "Ccrit",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = generate_tasks(n=120, seed=2026)

    baseline = BaselineAgent()
    metaslayer_v0 = MetaLayerAgentV0()
    metaslayer_v1 = MetaLayerAgentV1()
    metaslayer_soft = MetaLayerAgentV1Soft()

    b_rows, b_metrics = evaluate(baseline, tasks)
    m0_rows, m0_metrics = evaluate(metaslayer_v0, tasks)
    m1_rows, m1_metrics = evaluate(metaslayer_v1, tasks)
    ms_rows, ms_metrics = evaluate(metaslayer_soft, tasks)

    write_csv(out_dir / "baseline_results.csv", b_rows)
    write_csv(out_dir / "metaslayer_v0_results.csv", m0_rows)
    write_csv(out_dir / "metaslayer_v1_results.csv", m1_rows)
    write_csv(out_dir / "metaslayer_soft_results.csv", ms_rows)

    summary_path = out_dir / "pilot_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "success", "failure", "avg_time"])
        writer.writerow(["baseline", f"{b_metrics['success_rate']:.3f}", f"{b_metrics['failure_rate']:.3f}", f"{b_metrics['avg_time']:.3f}"])
        writer.writerow(["metaslayer_simple", f"{m0_metrics['success_rate']:.3f}", f"{m0_metrics['failure_rate']:.3f}", f"{m0_metrics['avg_time']:.3f}"])
        writer.writerow(["metaslayer_real_I", f"{m1_metrics['success_rate']:.3f}", f"{m1_metrics['failure_rate']:.3f}", f"{m1_metrics['avg_time']:.3f}"])
        writer.writerow(["metaslayer_soft", f"{ms_metrics['success_rate']:.3f}", f"{ms_metrics['failure_rate']:.3f}", f"{ms_metrics['avg_time']:.3f}"])

    print("Pilot complete. Metrics:")
    print(f"baseline  success_rate={b_metrics['success_rate']:.3f}  failure_rate={b_metrics['failure_rate']:.3f}  avg_time={b_metrics['avg_time']:.3f}")
    print(f"metaslayer_v0 success_rate={m0_metrics['success_rate']:.3f}  failure_rate={m0_metrics['failure_rate']:.3f}  avg_time={m0_metrics['avg_time']:.3f}")
    print(f"metaslayer_v1 success_rate={m1_metrics['success_rate']:.3f}  failure_rate={m1_metrics['failure_rate']:.3f}  avg_time={m1_metrics['avg_time']:.3f}")
    print(f"metaslayer_soft success_rate={ms_metrics['success_rate']:.3f}  failure_rate={ms_metrics['failure_rate']:.3f}  avg_time={ms_metrics['avg_time']:.3f}")

    def better_count(m, b):
        better = 0
        if m["success_rate"] >= b["success_rate"]:
            better += 1
        if m["failure_rate"] <= b["failure_rate"]:
            better += 1
        if m["avg_time"] <= b["avg_time"]:
            better += 1
        return better

    print(f"metaslayer_v0 better-or-equal on {better_count(m0_metrics, b_metrics)}/3 metrics")
    print(f"metaslayer_v1 better-or-equal on {better_count(m1_metrics, b_metrics)}/3 metrics")
    print(f"metaslayer_soft better-or-equal on {better_count(ms_metrics, b_metrics)}/3 metrics")


if __name__ == "__main__":
    main()
