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
        "final_attunement": float(out["mean_attunement"][-1]),
        "final_dispersion": float(out["phase_dispersion"][-1]),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_agents = 36
    alpha = np.concatenate([np.full(12, 0.78), np.full(12, 0.86), np.full(12, 0.95)])
    sigma_noise = np.concatenate([np.full(12, 0.30), np.full(12, 0.22), np.full(12, 0.14)])
    h_crit = np.concatenate([np.full(12, 0.62), np.full(12, 0.69), np.full(12, 0.77)])
    p_base = MasterParams(alpha=alpha, sigma_noise=sigma_noise, h_crit=h_crit)

    lessons = [
        {"name": "joint_1_noise", "lesson_type": "noise", "stress_amp": 2.8, "stress_frac": 0.24, "stress_y_amp": 0.90, "sigma_scale": 1.20, "delay_steps": (0, 3, 7), "contagion_weight": 0.75},
        {"name": "joint_2_lag", "lesson_type": "lag", "stress_amp": 3.0, "stress_frac": 0.26, "stress_y_amp": 1.00, "sigma_scale": 1.00, "delay_steps": (0, 5, 11), "contagion_weight": 0.75},
        {"name": "joint_3_contagion", "lesson_type": "contagion", "stress_amp": 3.2, "stress_frac": 0.34, "stress_y_amp": 1.15, "sigma_scale": 1.10, "delay_steps": (0, 3, 7), "contagion_weight": 1.20},
        {"name": "joint_4_mix", "lesson_type": "contagion", "stress_amp": 3.1, "stress_frac": 0.30, "stress_y_amp": 1.05, "sigma_scale": 1.05, "delay_steps": (0, 4, 9), "contagion_weight": 1.00},
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
        instructor_template_weight=0.70,
        attunement_enabled=True,
        attunement_gain=0.08,
        attunement_decay=0.010,
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
    agent_attunement = np.concatenate([np.full(12, 0.02), np.full(12, 0.06), np.full(12, 0.10)])

    for idx, lesson in enumerate(lessons, 1):
        p_lesson = MasterParams(
            **{
                **p_base.__dict__,
                "sigma_noise": p_base.sigma_noise * lesson["sigma_scale"],
            }
        )
        net = NetworkParams(
            **{
                **base_common,
                "delay_group_steps": lesson["delay_steps"],
                "stress_amp": lesson["stress_amp"],
                "stress_frac": lesson["stress_frac"],
                "stress_y_amp": lesson["stress_y_amp"],
                "instructor_response_mode": lesson["lesson_type"],
                "instructor_contagion_weight": lesson["contagion_weight"],
                "initial_group_memory": kg_memory,
                "initial_phi": phi_init,
                "initial_attunement": agent_attunement,
            }
        )
        out = simulate_network(p_lesson, net)
        metrics = lesson_metrics(out, net.stress_time, net.stress_duration)
        row = {
            "lesson": idx,
            "name": lesson["name"],
            "lesson_type": lesson["lesson_type"],
            "initial_group_memory": kg_memory,
            "initial_attunement": float(np.mean(agent_attunement)),
            **metrics,
        }
        rows.append(row)
        kg_memory = float(out["group_memory"][-1])
        phi_init = float(out["phi"][-1])
        agent_attunement = out["attunement"][-1].copy()
        print(
            f"{lesson['name']}: init_att={row['initial_attunement']:.3f}, "
            f"final_att={row['final_attunement']:.3f}, init_kg={row['initial_group_memory']:.3f}, "
            f"final_kg={row['final_group_memory']:.3f}, rt={row['recovery_time']:.3f}, "
            f"peak_error={row['peak_mean_error']:.3f}"
        )

    csv_path = out_dir / "instructor_joint_lessons_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    x = [row["lesson"] for row in rows]
    axes[0].plot(x, [row["initial_attunement"] for row in rows], marker="o", label="initial attunement")
    axes[0].plot(x, [row["final_attunement"] for row in rows], marker="o", label="final attunement")
    axes[0].set_ylabel("attunement")
    axes[0].set_title("Group attunement across lessons")
    axes[0].legend()

    axes[1].plot(x, [row["initial_group_memory"] for row in rows], marker="o", label="initial Kg")
    axes[1].plot(x, [row["final_group_memory"] for row in rows], marker="o", label="final Kg")
    axes[1].set_ylabel("group memory")
    axes[1].set_title("Instructor memory across lessons")
    axes[1].legend()

    axes[2].plot(x, [row["recovery_time"] for row in rows], marker="o", label="recovery time")
    axes[2].plot(x, [row["peak_mean_error"] for row in rows], marker="o", label="peak mean error")
    axes[2].set_ylabel("response")
    axes[2].set_title("Lesson quality")
    axes[2].set_xlabel("lesson")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "instructor_joint_lessons_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
