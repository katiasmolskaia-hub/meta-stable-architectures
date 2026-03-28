from __future__ import annotations

from dataclasses import asdict
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


def configure_master(base: MasterParams, sigma_noise: float) -> MasterParams:
    return MasterParams(
        **{
            **asdict(base),
            "sigma_h": base.sigma_h * 1.35,
            "delta_h": base.delta_h * 0.75,
            "h_crit": base.h_crit * 0.85,
            "eps_s": base.eps_s * 0.6,
            "eps_s2": 0.25,
            "c_crit": 0.6,
            "sigma_noise": sigma_noise,
        }
    )


def make_grouped_values(n: int, values: tuple[float, float, float]) -> np.ndarray:
    base = n // 3
    rem = n - 3 * base
    counts = [base, base, base]
    for i in range(rem):
        counts[i] += 1
    return np.concatenate([np.full(counts[i], values[i]) for i in range(3)])


def full_model_network(topology: str, seed: int) -> NetworkParams:
    net = NetworkParams(
        n_agents=180,
        t_end=45.0,
        dt=0.025,
        coupling=0.15,
        seed=seed,
        topology=topology,
        stress_time=22.0,
        stress_amp=1.95,
        stress_frac=0.28,
        stress_duration=2.0,
        stress_y_amp=1.05,
    )
    if topology == "ring":
        net.ring_k = 3
    elif topology == "erdos_renyi":
        net.er_p = 0.035
    elif topology == "small_world":
        net.ring_k = 3
        net.sw_rewire = 0.12
    elif topology == "scale_free":
        net.ba_m = 3
    elif topology == "spatial_local":
        net.spatial_radius = 0.16
        net.spatial_falloff = 0.10

    net.delay_mode = "grouped"
    net.delay_group_fracs = (0.34, 0.33, 0.33)
    net.delay_group_steps = (1, 6, 12)

    net.qrc_enabled = True
    net.phi_kappa = 1.0
    net.phi_gain = 0.5
    net.phi_gain_boost = 8.0
    net.qrc_g_min = 0.4
    net.recog_threshold = 0.7
    net.recog_gain = 1.2
    net.coh_relax_gain = 0.6
    net.wake_disp_threshold = 0.3
    net.wake_time_required = 4.0
    net.wake_relax_gain = 0.8
    net.phi_iso_threshold = 0.2

    net.instructor_enabled = True
    net.instructor_error_weight = 0.75
    net.instructor_anchor_weight = 0.85
    net.instructor_template_weight = 0.70
    net.instructor_response_mode = "contagion"
    net.instructor_contagion_weight = 1.25

    net.attunement_enabled = True
    net.attunement_gain = 0.08
    net.attunement_decay = 0.010
    net.initial_attunement = make_grouped_values(net.n_agents, (0.02, 0.06, 0.10))

    net.kg_enabled = True
    net.kg_lambda = 0.08
    net.kg_decay = 0.02
    net.kg_crisis_threshold = 0.45
    net.kg_phi_boost = 1.6
    net.kg_wake_boost = 1.2

    net.three_channel_enabled = True
    net.three_channel_adaptive = True
    net.channel_peer_base = 0.58
    net.channel_instructor_base = 0.25
    net.channel_anchor_base = 0.17
    net.channel_turbulence_gain = 0.80
    net.channel_attunement_gain = 0.55
    net.channel_anchor_gain = 0.55

    net.template_multimode_enabled = True
    net.template_focus_gain = 1.35
    net.template_recovery_gain = 0.82
    net.template_stabilize_gain = 0.92
    net.template_turbulence_trigger = 0.20
    net.template_recovery_trigger = 0.10

    net.demper_enabled = True
    net.demper_load_gain = 0.95
    net.demper_decay = 0.30
    net.demper_relax_gain = 0.90
    net.demper_trigger = 0.16

    return net


def run_metrics(out: dict[str, np.ndarray], threshold: float = 0.2) -> dict[str, float]:
    t = out["t"]
    s = out["S"]
    mean_s = out["mean_s"]
    stress_start = float(out["stress_windows"][0][0])
    stress_duration = float(out["stress_windows"][0][1])
    stress_end = stress_start + stress_duration
    start_idx = int(np.searchsorted(t, stress_end, side="left"))

    recovery_time = float("nan")
    for i in range(start_idx, len(t)):
        if mean_s[i] <= threshold:
            recovery_time = float(t[i] - stress_end)
            break

    per_agent = []
    for j in range(s.shape[1]):
        agent_rt = float("nan")
        for i in range(start_idx, len(t)):
            if s[i, j] <= threshold:
                agent_rt = float(t[i] - stress_end)
                break
        if not np.isnan(agent_rt):
            per_agent.append(agent_rt)
    per = np.array(per_agent, dtype=float)

    return {
        "success": 0.0 if np.isnan(recovery_time) else 1.0,
        "recovery_time": recovery_time,
        "tail_span": float(np.max(per) - np.min(per)) if per.size else float("nan"),
        "peak_mean_s": float(np.max(mean_s)),
        "peak_mean_error": float(np.max(out["mean_error"])),
        "peak_mean_turbulence": float(np.max(out["mean_turbulence"])),
        "peak_cascade_fraction": float(np.max(out["cascade_fraction"])),
        "final_attunement": float(out["mean_attunement"][-1]),
        "mean_channel_instructor": float(np.mean(out["mean_channel_instructor"][start_idx:])),
        "mean_channel_anchor": float(np.mean(out["mean_channel_anchor"][start_idx:])),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [7.8, 8.0, 8.2]
    topologies = ["small_world", "scale_free", "erdos_renyi", "spatial_local"]
    seeds = [7, 11]

    rows = []
    for sigma in sigmas:
        alpha = make_grouped_values(180, (0.78, 0.86, 0.95))
        sigma_arr = make_grouped_values(180, (0.30, 0.22, 0.14))
        h_crit = make_grouped_values(180, (0.62, 0.69, 0.77))
        base_master = MasterParams(alpha=alpha, sigma_noise=sigma_arr, h_crit=h_crit)
        p = configure_master(base_master, sigma)

        for topology in topologies:
            for seed in seeds:
                net = full_model_network(topology, seed)
                out = simulate_network(p=p, net=net)
                metrics = run_metrics(out)
                row = {
                    "sigma_noise": sigma,
                    "topology": topology,
                    "seed": seed,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"sigma={sigma:.1f} {topology} seed={seed}: success={row['success']:.0f}, "
                    f"rt={row['recovery_time']}, tail={row['tail_span']}, "
                    f"err={row['peak_mean_error']:.3f}, turb={row['peak_mean_turbulence']:.3f}, "
                    f"att={row['final_attunement']:.3f}"
                )

    summary_rows = []
    for sigma in sigmas:
        for topology in topologies:
            sub = [row for row in rows if row["sigma_noise"] == sigma and row["topology"] == topology]
            finite_rt = [float(r["recovery_time"]) for r in sub if not np.isnan(float(r["recovery_time"]))]
            finite_tail = [float(r["tail_span"]) for r in sub if not np.isnan(float(r["tail_span"]))]
            summary_rows.append(
                {
                    "sigma_noise": sigma,
                    "topology": topology,
                    "success_rate": float(np.mean([float(r["success"]) for r in sub])),
                    "mean_recovery_time": float(np.mean(finite_rt)) if finite_rt else float("nan"),
                    "mean_tail_span": float(np.mean(finite_tail)) if finite_tail else float("nan"),
                    "mean_peak_mean_s": float(np.mean([float(r["peak_mean_s"]) for r in sub])),
                    "mean_peak_error": float(np.mean([float(r["peak_mean_error"]) for r in sub])),
                    "mean_peak_turbulence": float(np.mean([float(r["peak_mean_turbulence"]) for r in sub])),
                    "mean_peak_cascade_fraction": float(np.mean([float(r["peak_cascade_fraction"]) for r in sub])),
                    "mean_final_attunement": float(np.mean([float(r["final_attunement"]) for r in sub])),
                }
            )

    csv_path = out_dir / "labile_field_boundary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "labile_field_boundary_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    x = np.arange(len(sigmas))
    labels = [str(s) for s in sigmas]
    for topology in topologies:
        sub = [row for row in summary_rows if row["topology"] == topology]
        axes[0].plot(x, [row["success_rate"] for row in sub], marker="o", label=topology)
        axes[1].plot(x, [row["mean_peak_error"] for row in sub], marker="o", label=topology)
        axes[2].plot(x, [row["mean_peak_turbulence"] for row in sub], marker="o", label=topology)
        axes[3].plot(x, [row["mean_final_attunement"] for row in sub], marker="o", label=topology)

    axes[0].set_ylabel("success")
    axes[0].set_title("Recovery success in a labile-field near-boundary regime")
    axes[1].set_ylabel("peak error")
    axes[1].set_title("Peak field error by topology")
    axes[2].set_ylabel("peak turbulence")
    axes[2].set_title("Peak turbulence by topology")
    axes[3].set_ylabel("attunement")
    axes[3].set_title("Final attunement by topology")
    axes[3].set_xlabel("sigma_noise")
    axes[3].set_xticks(x, labels)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plot_path = out_dir / "labile_field_boundary_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
