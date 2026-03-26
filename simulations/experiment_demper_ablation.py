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
    s = out["S"]
    stress_end = stress_start + stress_duration
    start_idx = int(np.searchsorted(t, stress_end, side="left"))

    recovery_time = float("nan")
    for i in range(start_idx, len(t)):
        if mean_s[i] <= threshold:
            recovery_time = float(t[i] - stress_end)
            break

    per_agent = []
    for j in range(s.shape[1]):
        rt = float("nan")
        for i in range(start_idx, len(t)):
            if s[i, j] <= threshold:
                rt = float(t[i] - stress_end)
                break
        if not np.isnan(rt):
            per_agent.append(rt)
    per = np.array(per_agent, dtype=float)

    return {
        "recovery_time": recovery_time,
        "tail_span": float(np.max(per) - np.min(per)) if per.size else float("nan"),
        "peak_mean_error": float(np.max(out["mean_error"])),
        "peak_mean_turbulence": float(np.max(out["mean_turbulence"])),
        "mean_final_attunement": float(out["mean_attunement"][-1]),
        "mean_final_demper_load": float(out["mean_demper_load"][-1]),
    }


def block_mean(rows: list[dict[str, float]], start: int, end: int, key: str) -> float:
    vals = [float(r[key]) for r in rows[start:end]]
    return float(np.mean(vals))


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_agents = 36
    alpha = np.concatenate([np.full(12, 0.78), np.full(12, 0.86), np.full(12, 0.95)])
    sigma_noise = np.concatenate([np.full(12, 0.30), np.full(12, 0.22), np.full(12, 0.14)])
    h_crit = np.concatenate([np.full(12, 0.62), np.full(12, 0.69), np.full(12, 0.77)])
    p_base = MasterParams(alpha=alpha, sigma_noise=sigma_noise, h_crit=h_crit)

    lesson_cycle = [
        {"lesson_type": "noise", "stress_amp": 2.8, "stress_frac": 0.24, "stress_y_amp": 0.90, "sigma_scale": 1.20, "delay_steps": (0, 3, 7), "contagion_weight": 0.75},
        {"lesson_type": "lag", "stress_amp": 3.0, "stress_frac": 0.26, "stress_y_amp": 1.00, "sigma_scale": 1.00, "delay_steps": (0, 5, 11), "contagion_weight": 0.75},
        {"lesson_type": "contagion", "stress_amp": 3.2, "stress_frac": 0.34, "stress_y_amp": 1.15, "sigma_scale": 1.10, "delay_steps": (0, 3, 7), "contagion_weight": 1.20},
        {"lesson_type": "contagion", "stress_amp": 3.1, "stress_frac": 0.30, "stress_y_amp": 1.05, "sigma_scale": 1.05, "delay_steps": (0, 4, 9), "contagion_weight": 1.00},
    ]
    lessons = []
    for block in range(4):
        for i, spec in enumerate(lesson_cycle, 1):
            lessons.append({"name": f"block_{block + 1}_lesson_{i}", **spec})

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
        three_channel_enabled=True,
        three_channel_adaptive=True,
        channel_peer_base=0.58,
        channel_instructor_base=0.25,
        channel_anchor_base=0.17,
        channel_turbulence_gain=0.80,
        channel_attunement_gain=0.55,
        channel_anchor_gain=0.55,
        template_multimode_enabled=True,
        template_focus_gain=1.35,
        template_recovery_gain=0.82,
        template_stabilize_gain=0.92,
        template_turbulence_trigger=0.20,
        template_recovery_trigger=0.10,
    )

    variants = {
        "without_demper": {
            "demper_enabled": False,
        },
        "with_demper": {
            "demper_enabled": True,
            "demper_load_gain": 0.95,
            "demper_decay": 0.30,
            "demper_relax_gain": 0.90,
            "demper_trigger": 0.16,
        },
    }

    rows = []
    summary_rows = []
    for variant, extra in variants.items():
        variant_rows = []
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
                    **extra,
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
                "variant": variant,
                "lesson": idx,
                "block": (idx - 1) // 4 + 1,
                "name": lesson["name"],
                "lesson_type": lesson["lesson_type"],
                **metrics,
            }
            rows.append(row)
            variant_rows.append(row)
            kg_memory = float(out["group_memory"][-1])
            phi_init = float(out["phi"][-1])
            agent_attunement = out["attunement"][-1].copy()
            print(
                f"{variant} {lesson['name']}: rt={row['recovery_time']:.3f}, tail={row['tail_span']:.3f}, "
                f"err={row['peak_mean_error']:.3f}, turb={row['peak_mean_turbulence']:.3f}, dem={row['mean_final_demper_load']:.3f}"
            )

        for block in range(4):
            start = block * 4
            end = start + 4
            summary_rows.append(
                {
                    "variant": variant,
                    "block": block + 1,
                    "mean_recovery_time": block_mean(variant_rows, start, end, "recovery_time"),
                    "mean_tail_span": block_mean(variant_rows, start, end, "tail_span"),
                    "mean_peak_error": block_mean(variant_rows, start, end, "peak_mean_error"),
                    "mean_peak_turbulence": block_mean(variant_rows, start, end, "peak_mean_turbulence"),
                    "mean_final_attunement": block_mean(variant_rows, start, end, "mean_final_attunement"),
                    "mean_final_demper_load": block_mean(variant_rows, start, end, "mean_final_demper_load"),
                }
            )

    csv_path = out_dir / "demper_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "demper_ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    x = np.arange(1, 5)
    for variant in variants:
        sub = [row for row in summary_rows if row["variant"] == variant]
        axes[0].plot(x, [row["mean_recovery_time"] for row in sub], marker="o", label=variant)
        axes[1].plot(x, [row["mean_tail_span"] for row in sub], marker="o", label=variant)
        axes[2].plot(x, [row["mean_peak_error"] for row in sub], marker="o", label=variant)
        axes[2].plot(x, [row["mean_peak_turbulence"] for row in sub], marker="x", linestyle="--", label=f"{variant} turbulence")
        axes[3].plot(x, [row["mean_final_demper_load"] for row in sub], marker="o", label=variant)

    axes[0].set_ylabel("recovery")
    axes[0].set_title("Block mean recovery")
    axes[1].set_ylabel("tail")
    axes[1].set_title("Block mean tail")
    axes[2].set_ylabel("field")
    axes[2].set_title("Block mean error and turbulence")
    axes[3].set_ylabel("demper")
    axes[3].set_title("Final demper load by block")
    axes[3].set_xlabel("block")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plot_path = out_dir / "demper_ablation_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
