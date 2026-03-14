"""Run pilot across multiple seeds and aggregate metrics."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean, pstdev

from baseline_agent import BaselineAgent
from metaslayer_v0 import MetaLayerAgentV0
from metaslayer_v1 import MetaLayerAgentV1, MetaLayerAgentV1Soft


def generate_tasks(n: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    tasks: list[tuple[float, float]] = []
    for i in range(n):
        complexity = rng.uniform(0.65, 1.25)
        if 10 <= i % 30 <= 18:
            complexity += rng.uniform(0.15, 0.35)
        noise = rng.uniform(-0.1, 0.3)
        tasks.append((complexity, noise))
    return tasks


def evaluate(agent, tasks: list[tuple[float, float]]) -> dict[str, float]:
    solved_count = 0
    failed_count = 0
    total_time = 0.0

    for complexity, noise in tasks:
        out = agent.solve(complexity, noise)
        solved = bool(out["solved"])
        failed = bool(out["failed"])
        task_time = float(out["time"])

        solved_count += 1 if solved else 0
        failed_count += 1 if failed else 0
        total_time += task_time

    n = len(tasks)
    return {
        "success_rate": solved_count / n,
        "failure_rate": failed_count / n,
        "avg_time": total_time / n,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [2026, 2027, 2028, 2029, 2030, 2041, 2042, 2043, 2044, 2045]
    n_tasks = 120

    models = {
        "baseline": BaselineAgent,
        "metaslayer_simple": MetaLayerAgentV0,
        "metaslayer_real_I": MetaLayerAgentV1,
        "metaslayer_soft": MetaLayerAgentV1Soft,
    }

    per_seed_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    metrics_by_model = {name: {"success_rate": [], "failure_rate": [], "avg_time": []} for name in models}

    for seed in seeds:
        tasks = generate_tasks(n=n_tasks, seed=seed)
        for name, cls in models.items():
            metrics = evaluate(cls(), tasks)
            metrics_by_model[name]["success_rate"].append(metrics["success_rate"])
            metrics_by_model[name]["failure_rate"].append(metrics["failure_rate"])
            metrics_by_model[name]["avg_time"].append(metrics["avg_time"])

            per_seed_rows.append(
                {
                    "seed": str(seed),
                    "model": name,
                    "success": f"{metrics['success_rate']:.6f}",
                    "failure": f"{metrics['failure_rate']:.6f}",
                    "avg_time": f"{metrics['avg_time']:.6f}",
                }
            )

    for name, vals in metrics_by_model.items():
        summary_rows.append(
            {
                "model": name,
                "success_mean": f"{mean(vals['success_rate']):.6f}",
                "success_std": f"{pstdev(vals['success_rate']):.6f}",
                "failure_mean": f"{mean(vals['failure_rate']):.6f}",
                "failure_std": f"{pstdev(vals['failure_rate']):.6f}",
                "avg_time_mean": f"{mean(vals['avg_time']):.6f}",
                "avg_time_std": f"{pstdev(vals['avg_time']):.6f}",
            }
        )

    per_seed_path = out_dir / "pilot_multiseed_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "model", "success", "failure", "avg_time"])
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_path = out_dir / "pilot_multiseed_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "success_mean",
                "success_std",
                "failure_mean",
                "failure_std",
                "avg_time_mean",
                "avg_time_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Multi-seed pilot complete.")
    for row in summary_rows:
        print(
            f"{row['model']}: success={row['success_mean']}?{row['success_std']} "
            f"failure={row['failure_mean']}?{row['failure_std']} avg_time={row['avg_time_mean']}?{row['avg_time_std']}"
        )


if __name__ == "__main__":
    main()
