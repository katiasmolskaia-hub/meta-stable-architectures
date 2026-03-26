from __future__ import annotations

from pathlib import Path
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from simulations.simulation_network_v1 import MasterParams, NetworkParams, simulate_network


def lesson_metrics(out: dict[str, np.ndarray], stress_start: float, stress_duration: float, threshold: float = 0.2) -> dict[str, float]:
    t = out["t"]
    mean_s = out["mean_s"]
    stress_end = stress_start + stress_duration
    start_idx = int(np.searchsorted(t, stress_end, side="left"))

    recovery_time = float("nan")
    for i in range(start_idx, len(t)):
        if mean_s[i] <= threshold:
            recovery_time = float(t[i] - stress_end)
            break

    return {
        "recovery_time": recovery_time,
        "peak_mean_s": float(np.max(mean_s)),
        "peak_mean_error": float(np.max(out["mean_error"])),
        "anchor_floor": float(np.min(out["mean_anchor"])),
        "final_group_memory": float(out["group_memory"][-1]),
        "final_dispersion": float(out["phase_dispersion"][-1]),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_agents = 36
    alpha = np.concatenate([np.full(12, 0.78), np.full(12, 0.86), np.full(12, 0.95)])
    sigma_noise = np.concatenate([np.full(12, 0.30), np.full(12, 0.22), np.full(12, 0.14)])
    h_crit = np.concatenate([np.full(12, 0.62), np.full(12, 0.69), np.full(12, 0.77)])
    p = MasterParams(alpha=alpha, sigma_noise=sigma_noise, h_crit=h_crit)

    lessons = [
        {
            "name": "lesson_1_noise",
            "lesson_type": "noise",
            "stress_amp": 2.8,
            "stress_frac": 0.24,
            "stress_y_amp": 0.90,
            "sigma_scale": 1.25,
            "delay_steps": (0, 3, 7),
            "contagion_weight": 0.75,
        },
        {
            "name": "lesson_2_lag",
            "lesson_type": "lag",
            "stress_amp": 3.0,
            "stress_frac": 0.26,
            "stress_y_amp": 1.00,
            "sigma_scale": 1.00,
            "delay_steps": (0, 5, 11),
            "contagion_weight": 0.75,
        },
        {
            "name": "lesson_3_contagion",
            "lesson_type": "contagion",
            "stress_amp": 3.2,
            "stress_frac": 0.34,
            "stress_y_amp": 1.15,
            "sigma_scale": 1.10,
            "delay_steps": (0, 3, 7),
            "contagion_weight": 1.20,
        },
    ]

    base_common = dict(
        n_agents=n_agents,
        t_end=18.0,
        dt=0.02,
        topology="spatial_local",
        spatial_radius=0.34,
        spatial_falloff=0.16,
        coupling=0.17,
        delay_mode="grouped",
        delay_group_fracs=(0.34, 0.33, 0.33),
        delay_group_steps=(0, 3, 7),
        stress_time=7.0,
        stress_duration=2.5,
        qrc_enabled=True,
        phi_kappa=1.0,
        phi_gain=0.42,
        phi_gain_boost=8.0,
        ccrit_gain=0.6,
        recog_threshold=0.7,
        recog_gain=1.2,
        wake_time_required=4.0,
        wake_relax_gain=0.8,
        coh_relax_gain=0.6,
        instructor_enabled=True,
        instructor_error_weight=0.75,
        instructor_anchor_weight=0.85,
        instructor_contagion_weight=0.75,
        instructor_template_weight=0.70,
        kg_enabled=True,
        kg_lambda=0.08,
        kg_decay=0.02,
        kg_crisis_threshold=0.45,
        kg_phi_boost=1.6,
        kg_wake_boost=1.2,
    )

    rows = []
    kg_memory = 0.0
    phi_init = 0.0

    for idx, lesson in enumerate(lessons, 1):
        p_lesson = MasterParams(
            **{
                **p.__dict__,
                "sigma_noise": p.sigma_noise * lesson["sigma_scale"],
            }
        )
        net = NetworkParams(
            **{
                **base_common,
                "stress_amp": lesson["stress_amp"],
                "stress_frac": lesson["stress_frac"],
                "stress_y_amp": lesson["stress_y_amp"],
                "delay_group_steps": lesson["delay_steps"],
                "instructor_contagion_weight": lesson["contagion_weight"],
                "initial_group_memory": kg_memory,
                "initial_phi": phi_init,
            }
        )
        out = simulate_network(p_lesson, net)
        metrics = lesson_metrics(out, net.stress_time, net.stress_duration)
        row = {
            "lesson": idx,
            "name": lesson["name"],
            "lesson_type": lesson["lesson_type"],
            "stress_amp": lesson["stress_amp"],
            "stress_frac": lesson["stress_frac"],
            "sigma_scale": lesson["sigma_scale"],
            "initial_group_memory": kg_memory,
            **metrics,
        }
        rows.append(row)
        kg_memory = float(out["group_memory"][-1])
        phi_init = float(out["phi"][-1]) if "phi" in out else 0.0
        print(
            f"{lesson['name']}: init_kg={row['initial_group_memory']:.3f}, "
            f"final_kg={row['final_group_memory']:.3f}, rt={row['recovery_time']:.3f}, "
            f"peak_error={row['peak_mean_error']:.3f}, anchor_floor={row['anchor_floor']:.3f}, "
            f"type={row['lesson_type']}"
        )

    csv_path = out_dir / "instructor_v2_lessons_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    lesson_idx = [row["lesson"] for row in rows]
    axes[0].plot(lesson_idx, [row["initial_group_memory"] for row in rows], marker="o", label="initial Kg")
    axes[0].plot(lesson_idx, [row["final_group_memory"] for row in rows], marker="o", label="final Kg")
    axes[0].set_ylabel("group memory")
    axes[0].set_title("Instructor memory across lessons")
    axes[0].legend()

    axes[1].plot(lesson_idx, [row["recovery_time"] for row in rows], marker="o", label="recovery time")
    axes[1].plot(lesson_idx, [row["peak_mean_error"] for row in rows], marker="o", label="peak mean error")
    axes[1].set_ylabel("lesson response")
    axes[1].set_title("Recovery and error")
    axes[1].legend()

    axes[2].plot(lesson_idx, [row["anchor_floor"] for row in rows], marker="o", label="anchor floor")
    axes[2].plot(lesson_idx, [row["final_dispersion"] for row in rows], marker="o", label="final dispersion")
    axes[2].set_ylabel("stability")
    axes[2].set_title("Anchor floor and dispersion")
    axes[2].set_xlabel("lesson")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "instructor_v2_lessons_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
