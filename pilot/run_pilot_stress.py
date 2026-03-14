"""Run stress test by increasing noise scale and compare models."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from baseline_agent import BaselineAgent
from metaslayer_v0 import MetaLayerAgentV0
from metaslayer_v1 import MetaLayerAgentV1, MetaLayerAgentV1Soft


def generate_tasks(n: int, seed: int, noise_scale: float) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    tasks: list[tuple[float, float]] = []
    for i in range(n):
        complexity = rng.uniform(0.65, 1.25)
        if 10 <= i % 30 <= 18:
            complexity += rng.uniform(0.15, 0.35)

        noise = rng.uniform(-0.1, 0.3) * noise_scale
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

    # Interpret sigma as a noise multiplier around baseline 0.6
    sigmas = [0.6, 0.7, 0.8]
    noise_scales = {s: s / 0.6 for s in sigmas}

    seed = 2026
    n_tasks = 120

    models = {
        "baseline": BaselineAgent,
        "metaslayer_simple": MetaLayerAgentV0,
        "metaslayer_real_I": MetaLayerAgentV1,
        "metaslayer_soft": MetaLayerAgentV1Soft,
    }

    rows: list[dict[str, str]] = []

    for sigma, scale in noise_scales.items():
        tasks = generate_tasks(n=n_tasks, seed=seed, noise_scale=scale)
        for name, cls in models.items():
            metrics = evaluate(cls(), tasks)
            rows.append(
                {
                    "sigma": f"{sigma:.2f}",
                    "noise_scale": f"{scale:.3f}",
                    "model": name,
                    "success": f"{metrics['success_rate']:.3f}",
                    "failure": f"{metrics['failure_rate']:.3f}",
                    "avg_time": f"{metrics['avg_time']:.3f}",
                }
            )

    out_path = out_dir / "pilot_stress_sigma.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma", "noise_scale", "model", "success", "failure", "avg_time"])
        writer.writeheader()
        writer.writerows(rows)

    print("Stress test complete.")
    for row in rows:
        print(
            f"sigma={row['sigma']} model={row['model']} "
            f"success={row['success']} failure={row['failure']} avg_time={row['avg_time']}"
        )


if __name__ == "__main__":
    main()
