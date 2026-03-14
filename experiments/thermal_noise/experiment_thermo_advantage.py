"""
Thermodynamic advantage of knowledge:
compare raw energy cost E(t) vs effective cost E_eff(t) reduced by K(t).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from simulation_noise import NoiseParams, simulate_master_langevin


def sigma_schedule(t: float) -> float:
    # Two identical high-noise windows to show reduced effective cost after learning.
    if 20.0 <= t < 45.0:
        return 0.6
    if 80.0 <= t < 105.0:
        return 0.6
    return 0.15


def make_svg(
    t: np.ndarray,
    e: np.ndarray,
    e_eff: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    width, height = 700, 420
    pad = 60

    min_x, max_x = float(t[0]), float(t[-1])
    min_y = float(min(e.min(), e_eff.min()))
    max_y = float(max(e.max(), e_eff.max()))
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def sx(x: float) -> float:
        return pad + (x - min_x) / span_x * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / span_y * (height - 2 * pad)

    pts_e = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(t, e))
    pts_eff = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(t, e_eff))

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>
  <line x1=\"{pad}\" y1=\"{pad}\" x2=\"{pad}\" y2=\"{height - pad}\" stroke=\"#333\" stroke-width=\"1\"/>
  <line x1=\"{pad}\" y1=\"{height - pad}\" x2=\"{width - pad}\" y2=\"{height - pad}\" stroke=\"#333\" stroke-width=\"1\"/>
  <polyline fill=\"none\" stroke=\"#d62728\" stroke-width=\"2\" points=\"{pts_e}\"/>
  <polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"2\" points=\"{pts_eff}\"/>
  <text x=\"{width/2:.1f}\" y=\"{pad/2:.1f}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"16\">{title}</text>
  <text x=\"{width/2:.1f}\" y=\"{height-12}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\">time</text>
  <text x=\"18\" y=\"{height/2:.1f}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\" transform=\"rotate(-90 18 {height/2:.1f})\">energy</text>
  <rect x=\"{width - pad - 140}\" y=\"{pad + 6}\" width=\"130\" height=\"42\" fill=\"white\" stroke=\"#ccc\"/>
  <line x1=\"{width - pad - 130}\" y1=\"{pad + 20}\" x2=\"{width - pad - 110}\" y2=\"{pad + 20}\" stroke=\"#d62728\" stroke-width=\"2\"/>
  <text x=\"{width - pad - 100}\" y=\"{pad + 24}\" font-family=\"Arial\" font-size=\"12\">E(t)</text>
  <line x1=\"{width - pad - 130}\" y1=\"{pad + 38}\" x2=\"{width - pad - 110}\" y2=\"{pad + 38}\" stroke=\"#1f77b4\" stroke-width=\"2\"/>
  <text x=\"{width - pad - 100}\" y=\"{pad + 42}\" font-family=\"Arial\" font-size=\"12\">E_eff(t)</text>
</svg>"""

    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    out_dir = Path("outputs/experiments/20260314_thermo_advantage")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = NoiseParams()
    params.eta = 0.85
    params.sigma_a = 0.15

    det = simulate_master_langevin(
        t_end=140.0, dt=0.01, p=params, seed=12, sigma_schedule=sigma_schedule
    )

    # Energy model
    e0 = 0.3
    alpha_i = 1.0
    alpha_v = 0.8
    beta = 0.9

    v_min = -0.25  # minimum of quartic potential
    v_shift = det["V"] - v_min
    e = e0 + alpha_i * det["I"] + alpha_v * v_shift
    e_eff = e / (1.0 + beta * det["K"])

    # Save traces
    csv_path = out_dir / "thermo_traces.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "E", "E_eff", "K", "I", "C"])
        for t, e_i, eef, k, i, c in zip(det["t"], e, e_eff, det["K"], det["I"], det["C"]):
            writer.writerow([f"{t:.4f}", f"{e_i:.6f}", f"{eef:.6f}", f"{k:.6f}", f"{i:.6f}", f"{c:.6f}"])

    # Summary: compare two identical noise windows
    def window_mean(arr: np.ndarray, t0: float, t1: float) -> float:
        mask = (det["t"] >= t0) & (det["t"] < t1)
        return float(arr[mask].mean())

    summary_path = out_dir / "summary.txt"
    w1_e = window_mean(e, 20.0, 45.0)
    w2_e = window_mean(e, 80.0, 105.0)
    w1_eff = window_mean(e_eff, 20.0, 45.0)
    w2_eff = window_mean(e_eff, 80.0, 105.0)
    summary_path.write_text(
        "\n".join(
            [
                "Thermodynamic advantage of knowledge",
                f"E mean window1: {w1_e:.4f}",
                f"E mean window2: {w2_e:.4f}",
                f"E_eff mean window1: {w1_eff:.4f}",
                f"E_eff mean window2: {w2_eff:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    svg_path = out_dir / "thermo_E_Eeff.svg"
    make_svg(det["t"], e, e_eff, svg_path, "Energy cost vs effective cost")

    print(f"Saved: {csv_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
