"""Run narrow pilot: baseline vs metaslayer_v0."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from baseline_agent import BaselineAgent
from metaslayer_v0 import MetaLayerAgentV0


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
    metaslayer = MetaLayerAgentV0()

    b_rows, b_metrics = evaluate(baseline, tasks)
    m_rows, m_metrics = evaluate(metaslayer, tasks)

    write_csv(out_dir / "baseline_results.csv", b_rows)
    write_csv(out_dir / "metaslayer_results.csv", m_rows)

    print("Pilot complete. Metrics:")
    print(f"baseline  success_rate={b_metrics['success_rate']:.3f}  failure_rate={b_metrics['failure_rate']:.3f}  avg_time={b_metrics['avg_time']:.3f}")
    print(f"metaslayer success_rate={m_metrics['success_rate']:.3f}  failure_rate={m_metrics['failure_rate']:.3f}  avg_time={m_metrics['avg_time']:.3f}")

    better = 0
    if m_metrics["success_rate"] >= b_metrics["success_rate"]:
        better += 1
    if m_metrics["failure_rate"] <= b_metrics["failure_rate"]:
        better += 1
    if m_metrics["avg_time"] <= b_metrics["avg_time"]:
        better += 1

    print(f"metaslayer better-or-equal on {better}/3 metrics")


if __name__ == "__main__":
    main()
