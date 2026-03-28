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
        "peak_cascade_fraction": float(np.max(out["cascade_fraction"])),
        "final_attunement": float(out["mean_attunement"][-1]),
        "final_group_memory": float(out["group_memory"][-1]),
        "final_demper_load": float(out["mean_demper_load"][-1]),
    }


def grouped_values(counts: tuple[int, int, int], values: tuple[float, float, float]) -> np.ndarray:
    return np.concatenate([np.full(counts[i], values[i]) for i in range(3)])


def build_master_params(heterogeneity_mode: str) -> MasterParams:
    counts = (12, 12, 12)
    if heterogeneity_mode == "standard":
        alpha = grouped_values(counts, (0.78, 0.86, 0.95))
        sigma_noise = grouped_values(counts, (0.30, 0.22, 0.14))
        h_crit = grouped_values(counts, (0.62, 0.69, 0.77))
    elif heterogeneity_mode == "harsh":
        alpha = grouped_values(counts, (0.70, 0.86, 1.02))
        sigma_noise = grouped_values(counts, (0.36, 0.22, 0.10))
        h_crit = grouped_values(counts, (0.56, 0.69, 0.82))
    else:
        raise ValueError(f"Unknown heterogeneity_mode: {heterogeneity_mode}")
    return MasterParams(alpha=alpha, sigma_noise=sigma_noise, h_crit=h_crit)


def initial_attunement(heterogeneity_mode: str) -> np.ndarray:
    if heterogeneity_mode == "standard":
        return grouped_values((12, 12, 12), (0.02, 0.06, 0.10))
    return grouped_values((12, 12, 12), (0.00, 0.05, 0.16))


def base_common(topology: str) -> dict[str, object]:
    common = dict(
        n_agents=36,
        t_end=18.0,
        dt=0.02,
        coupling=0.17,
        topology=topology,
        delay_mode="grouped",
        delay_group_fracs=(0.34, 0.33, 0.33),
        stress_time=7.0,
        stress_duration=2.5,
        stress_amp=3.45,
        stress_frac=0.36,
        stress_y_amp=1.20,
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
        instructor_response_mode="contagion",
        instructor_contagion_weight=1.30,
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
        demper_enabled=True,
        demper_load_gain=0.95,
        demper_decay=0.30,
        demper_relax_gain=0.90,
        demper_trigger=0.16,
        delay_group_steps=(1, 7, 14),
    )
    topo_specific = {
        "ring": {"ring_k": 3},
        "erdos_renyi": {"er_p": 0.13},
        "small_world": {"ring_k": 3, "sw_rewire": 0.18},
        "scale_free": {"ba_m": 3},
        "spatial_local": {"spatial_radius": 0.30, "spatial_falloff": 0.16},
    }
    return {**common, **topo_specific[topology]}


def warmup_lessons() -> list[dict[str, object]]:
    return [
        {"lesson_type": "noise", "stress_amp": 2.8, "stress_frac": 0.24, "stress_y_amp": 0.90, "sigma_scale": 1.20, "delay_steps": (0, 3, 7), "contagion_weight": 0.75},
        {"lesson_type": "lag", "stress_amp": 3.0, "stress_frac": 0.26, "stress_y_amp": 1.00, "sigma_scale": 1.00, "delay_steps": (0, 5, 11), "contagion_weight": 0.75},
        {"lesson_type": "contagion", "stress_amp": 3.2, "stress_frac": 0.34, "stress_y_amp": 1.15, "sigma_scale": 1.10, "delay_steps": (0, 3, 7), "contagion_weight": 1.20},
        {"lesson_type": "contagion", "stress_amp": 3.1, "stress_frac": 0.30, "stress_y_amp": 1.05, "sigma_scale": 1.05, "delay_steps": (0, 4, 9), "contagion_weight": 1.00},
    ]


def apply_lesson(master: MasterParams, common: dict[str, object], lesson: dict[str, object], group_memory: float, phi_init: float, agent_attunement: np.ndarray) -> tuple[dict[str, np.ndarray], NetworkParams]:
    p_lesson = MasterParams(
        **{
            **master.__dict__,
            "sigma_noise": master.sigma_noise * float(lesson["sigma_scale"]),
        }
    )
    net = NetworkParams(
        **{
            **common,
            "stress_amp": lesson["stress_amp"],
            "stress_frac": lesson["stress_frac"],
            "stress_y_amp": lesson["stress_y_amp"],
            "instructor_response_mode": lesson["lesson_type"],
            "instructor_contagion_weight": lesson["contagion_weight"],
            "delay_group_steps": lesson["delay_steps"],
            "initial_group_memory": group_memory,
            "initial_phi": phi_init,
            "initial_attunement": agent_attunement,
        }
    )
    out = simulate_network(p_lesson, net)
    return out, net


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    topologies = ["ring", "erdos_renyi", "small_world", "scale_free", "spatial_local"]
    heterogeneity_modes = ["standard", "harsh"]
    evaluation_lesson = {
        "lesson_type": "contagion",
        "stress_amp": 3.45,
        "stress_frac": 0.36,
        "stress_y_amp": 1.20,
        "sigma_scale": 1.15,
        "delay_steps": (1, 7, 14),
        "contagion_weight": 1.35,
    }

    rows: list[dict[str, float | str]] = []

    for topology in topologies:
        for heterogeneity_mode in heterogeneity_modes:
            master = build_master_params(heterogeneity_mode)
            common = base_common(topology)
            group_memory = 0.0
            phi_init = 0.0
            agent_att = initial_attunement(heterogeneity_mode)

            for lesson in warmup_lessons():
                out, _ = apply_lesson(master, common, lesson, group_memory, phi_init, agent_att)
                group_memory = float(out["group_memory"][-1])
                phi_init = float(out["phi"][-1])
                agent_att = out["attunement"][-1].copy()

            out, net = apply_lesson(master, common, evaluation_lesson, group_memory, phi_init, agent_att)
            metrics = lesson_metrics(out, net.stress_time, net.stress_duration)
            row = {
                "topology": topology,
                "heterogeneity_mode": heterogeneity_mode,
                **metrics,
            }
            rows.append(row)
            print(
                f"{topology} {heterogeneity_mode}: rt={metrics['recovery_time']:.3f}, "
                f"tail={metrics['tail_span']:.3f}, err={metrics['peak_mean_error']:.3f}, "
                f"turb={metrics['peak_mean_turbulence']:.3f}, att={metrics['final_attunement']:.3f}"
            )

    csv_path = out_dir / "topology_heterogeneity_stress.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(topologies))
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    for heterogeneity_mode, marker in [("standard", "o"), ("harsh", "x")]:
        sub = [row for row in rows if row["heterogeneity_mode"] == heterogeneity_mode]
        axes[0].plot(x, [row["recovery_time"] for row in sub], marker=marker, label=heterogeneity_mode)
        axes[1].plot(x, [row["tail_span"] for row in sub], marker=marker, label=heterogeneity_mode)
        axes[2].plot(x, [row["peak_mean_error"] for row in sub], marker=marker, label=f"{heterogeneity_mode} error")
        axes[2].plot(x, [row["peak_mean_turbulence"] for row in sub], marker=marker, linestyle="--", label=f"{heterogeneity_mode} turbulence")
        axes[3].plot(x, [row["final_attunement"] for row in sub], marker=marker, label=heterogeneity_mode)

    axes[0].set_ylabel("recovery")
    axes[0].set_title("Shifted-contagion recovery by topology")
    axes[1].set_ylabel("tail")
    axes[1].set_title("Shifted-contagion tail by topology")
    axes[2].set_ylabel("field")
    axes[2].set_title("Shifted-contagion field quality by topology")
    axes[3].set_ylabel("attunement")
    axes[3].set_title("Final attunement by topology")
    axes[3].set_xlabel("topology")
    axes[3].set_xticks(x, topologies, rotation=15)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plot_path = out_dir / "topology_heterogeneity_stress_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
