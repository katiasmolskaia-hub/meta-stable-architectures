"""Long-run pilot with variable noise: baseline vs metaslayer_v1."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import random

from baseline_agent import BaselineAgent
from metaslayer_v1 import MetaLayerAgentV1


def generate_tasks(n: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    tasks: list[tuple[float, float]] = []
    for i in range(n):
        base = rng.uniform(0.65, 1.25)
        # variable stress windows
        if 60 <= i < 120:
            base += rng.uniform(0.20, 0.45)
        if 180 <= i < 240:
            base += rng.uniform(0.15, 0.35)
        if 300 <= i < 340:
            base += rng.uniform(0.25, 0.55)

        noise = rng.uniform(-0.1, 0.25)
        # variable noise offsets
        if 60 <= i < 120:
            noise += rng.uniform(0.15, 0.30)
        if 180 <= i < 240:
            noise += rng.uniform(0.10, 0.25)
        if 300 <= i < 340:
            noise += rng.uniform(0.25, 0.45)

        tasks.append((base, noise))
    return tasks


def compute_energy(I: float, C: float, E0: float = 0.25, alpha_I: float = 0.8, alpha_C: float = 0.35) -> float:
    return E0 + alpha_I * I + alpha_C * C


def evaluate(agent, tasks: list[tuple[float, float]]) -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    solved_count = 0
    failed_count = 0
    total_time = 0.0
    total_E = 0.0
    total_E_eff = 0.0

    # fallback for baseline (no C) -> compute from overload
    fail_margin = getattr(agent, "fail_margin", 0.25)

    for i, (complexity, noise) in enumerate(tasks):
        out = agent.solve(complexity, noise)
        solved = bool(out["solved"])
        failed = bool(out["failed"])
        task_time = float(out["time"])
        overload = float(out.get("overload", 0.0))

        solved_count += 1 if solved else 0
        failed_count += 1 if failed else 0
        total_time += task_time

        C = float(out.get("C", (max(0.0, overload) / max(fail_margin, 1e-6)) ** 2))
        Ccrit = float(out.get("Ccrit", 0.0))
        I = float(out.get("I", 0.0))
        K = float(out.get("K", 0.0))

        E = compute_energy(I, C)
        E_eff = E / (1.0 + 0.9 * K)

        total_E += E
        total_E_eff += E_eff

        row = {
            "step": i,
            "complexity": complexity,
            "noise": noise,
            "solved": int(solved),
            "failed": int(failed),
            "time": task_time,
            "overload": overload,
            "I": I,
            "C": C,
            "Ccrit": Ccrit,
            "K": K,
            "E": E,
            "E_eff": E_eff,
        }
        rows.append(row)

    n = len(tasks)
    metrics = {
        "success_rate": solved_count / n,
        "failure_rate": failed_count / n,
        "avg_time": total_time / n,
        "avg_E": total_E / n,
        "avg_E_eff": total_E_eff / n,
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
        "I",
        "C",
        "Ccrit",
        "K",
        "E",
        "E_eff",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_svg_two_lines(x, y1, y2, out_path: Path, title: str, label1: str, label2: str) -> None:
    width, height = 700, 420
    pad = 60

    min_x, max_x = float(min(x)), float(max(x))
    min_y = float(min(min(y1), min(y2)))
    max_y = float(max(max(y1), max(y2)))
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def sx(v: float) -> float:
        return pad + (v - min_x) / span_x * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - (v - min_y) / span_y * (height - 2 * pad)

    pts1 = " ".join(f"{sx(a):.2f},{sy(b):.2f}" for a, b in zip(x, y1))
    pts2 = " ".join(f"{sx(a):.2f},{sy(b):.2f}" for a, b in zip(x, y2))

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>
  <line x1=\"{pad}\" y1=\"{pad}\" x2=\"{pad}\" y2=\"{height - pad}\" stroke=\"#333\" stroke-width=\"1\"/>
  <line x1=\"{pad}\" y1=\"{height - pad}\" x2=\"{width - pad}\" y2=\"{height - pad}\" stroke=\"#333\" stroke-width=\"1\"/>
  <polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"2\" points=\"{pts1}\"/>
  <polyline fill=\"none\" stroke=\"#d62728\" stroke-width=\"2\" points=\"{pts2}\"/>
  <text x=\"{width/2:.1f}\" y=\"{pad/2:.1f}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"16\">{title}</text>
  <text x=\"{width/2:.1f}\" y=\"{height-12}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\">step</text>
  <rect x=\"{width - pad - 160}\" y=\"{pad + 6}\" width=\"150\" height=\"42\" fill=\"white\" stroke=\"#ccc\"/>
  <line x1=\"{width - pad - 150}\" y1=\"{pad + 20}\" x2=\"{width - pad - 130}\" y2=\"{pad + 20}\" stroke=\"#1f77b4\" stroke-width=\"2\"/>
  <text x=\"{width - pad - 120}\" y=\"{pad + 24}\" font-family=\"Arial\" font-size=\"12\">{label1}</text>
  <line x1=\"{width - pad - 150}\" y1=\"{pad + 38}\" x2=\"{width - pad - 130}\" y2=\"{pad + 38}\" stroke=\"#d62728\" stroke-width=\"2\"/>
  <text x=\"{width - pad - 120}\" y=\"{pad + 42}\" font-family=\"Arial\" font-size=\"12\">{label2}</text>
</svg>"""

    out_path.write_text(svg, encoding="utf-8")


def save_svg_one_line(x, y, out_path: Path, title: str, label: str) -> None:
    width, height = 700, 420
    pad = 60

    min_x, max_x = float(min(x)), float(max(x))
    min_y, max_y = float(min(y)), float(max(y))
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def sx(v: float) -> float:
        return pad + (v - min_x) / span_x * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - (v - min_y) / span_y * (height - 2 * pad)

    pts = " ".join(f"{sx(a):.2f},{sy(b):.2f}" for a, b in zip(x, y))

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>
  <line x1=\"{pad}\" y1=\"{pad}\" x2=\"{pad}\" y2=\"{height - pad}\" stroke=\"#333\" stroke-width=\"1\"/>
  <line x1=\"{pad}\" y1=\"{height - pad}\" x2=\"{width - pad}\" y2=\"{height - pad}\" stroke=\"#333\" stroke-width=\"1\"/>
  <polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"2\" points=\"{pts}\"/>
  <text x=\"{width/2:.1f}\" y=\"{pad/2:.1f}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"16\">{title}</text>
  <text x=\"{width/2:.1f}\" y=\"{height-12}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\">step</text>
  <text x=\"{width - pad - 10}\" y=\"{pad + 20}\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"12\">{label}</text>
</svg>"""

    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "results" / "long_variable_noise"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = generate_tasks(n=360, seed=20260314)

    baseline = BaselineAgent()
    metaslayer = MetaLayerAgentV1()

    b_rows, b_metrics = evaluate(baseline, tasks)
    m_rows, m_metrics = evaluate(metaslayer, tasks)

    write_csv(out_dir / "baseline_long.csv", b_rows)
    write_csv(out_dir / "metaslayer_long.csv", m_rows)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "success", "failure", "avg_time", "avg_E", "avg_E_eff"])
        writer.writerow(["baseline", f"{b_metrics['success_rate']:.3f}", f"{b_metrics['failure_rate']:.3f}", f"{b_metrics['avg_time']:.3f}", f"{b_metrics['avg_E']:.3f}", f"{b_metrics['avg_E_eff']:.3f}"])
        writer.writerow(["metaslayer_v1", f"{m_metrics['success_rate']:.3f}", f"{m_metrics['failure_rate']:.3f}", f"{m_metrics['avg_time']:.3f}", f"{m_metrics['avg_E']:.3f}", f"{m_metrics['avg_E_eff']:.3f}"])

    steps = [r["step"] for r in m_rows]
    m_E = [r["E"] for r in m_rows]
    m_Eeff = [r["E_eff"] for r in m_rows]
    m_C = [r["C"] for r in m_rows]
    m_I = [r["I"] for r in m_rows]

    b_steps = [r["step"] for r in b_rows]
    b_E = [r["E"] for r in b_rows]
    b_C = [r["C"] for r in b_rows]

    save_svg_two_lines(steps, m_E, m_Eeff, out_dir / "metaslayer_energy.svg", "Metaslayer energy track", "E(t)", "E_eff(t)")
    save_svg_two_lines(steps, m_C, m_I, out_dir / "metaslayer_C_I.svg", "Metaslayer crisis & isolation", "C(t)", "I(t)")
    save_svg_one_line(b_steps, b_E, out_dir / "baseline_energy.svg", "Baseline energy track", "E(t)")
    save_svg_one_line(b_steps, b_C, out_dir / "baseline_C.svg", "Baseline crisis track", "C(t)")

    print("Long variable-noise pilot complete.")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
