from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
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


def base_network(seed: int) -> NetworkParams:
    net = NetworkParams(
        n_agents=1000,
        t_end=60.0,
        dt=0.025,
        coupling=0.15,
        ring_k=2,
        seed=seed,
        topology="small_world",
        sw_rewire=0.1,
        stress_time=30.0,
        stress_amp=1.8,
        stress_frac=0.24,
        stress_duration=2.0,
        stress_y_amp=1.0,
    )
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
    return net


def run_metrics(out: dict[str, np.ndarray]) -> dict[str, float]:
    rt = float(out["recovery_time"])
    recovered = not np.isnan(rt)

    t = out["t"]
    s = out["S"]
    mean_s = out["mean_s"]
    stress_start = float(out["stress_windows"][0][0])
    stress_duration = float(out["stress_windows"][0][1])
    stress_end = stress_start + stress_duration
    start_idx = int(np.searchsorted(t, stress_end, side="left"))

    per_agent = []
    for j in range(s.shape[1]):
        art = float("nan")
        for i in range(start_idx, len(t)):
            if s[i, j] <= 0.2:
                art = float(t[i] - stress_end)
                break
        if not np.isnan(art):
            per_agent.append(art)
    per = np.array(per_agent, dtype=float)

    return {
        "success": 1.0 if recovered else 0.0,
        "recovery_time": rt,
        "tail_span": float(np.max(per) - np.min(per)) if per.size else float("nan"),
        "peak_mean_s": float(np.max(mean_s)),
        "peak_mean_error": float(np.max(out["mean_error"])),
        "peak_mean_turbulence": float(np.max(out["mean_turbulence"])),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [7.9, 8.0, 8.1]
    seeds = [7]

    variants = {
        "minimal_rc": {
            "instructor_enabled": False,
            "attunement_enabled": False,
            "kg_enabled": False,
            "three_channel_enabled": False,
            "template_multimode_enabled": False,
            "demper_enabled": False,
        },
        "full_model": {
            "instructor_enabled": True,
            "instructor_error_weight": 0.75,
            "instructor_anchor_weight": 0.85,
            "instructor_template_weight": 0.70,
            "attunement_enabled": True,
            "attunement_gain": 0.08,
            "attunement_decay": 0.010,
            "kg_enabled": True,
            "kg_lambda": 0.08,
            "kg_decay": 0.02,
            "kg_crisis_threshold": 0.45,
            "kg_phi_boost": 1.6,
            "kg_wake_boost": 1.2,
            "three_channel_enabled": True,
            "three_channel_adaptive": True,
            "channel_peer_base": 0.58,
            "channel_instructor_base": 0.25,
            "channel_anchor_base": 0.17,
            "channel_turbulence_gain": 0.80,
            "channel_attunement_gain": 0.55,
            "channel_anchor_gain": 0.55,
            "template_multimode_enabled": True,
            "template_focus_gain": 1.35,
            "template_recovery_gain": 0.82,
            "template_stabilize_gain": 0.92,
            "template_turbulence_trigger": 0.20,
            "template_recovery_trigger": 0.10,
            "demper_enabled": True,
            "demper_load_gain": 0.95,
            "demper_decay": 0.30,
            "demper_relax_gain": 0.90,
            "demper_trigger": 0.16,
        },
    }

    rows = []
    summary_rows = []

    for sigma in sigmas:
        p = configure_master(MasterParams(), sigma)
        for variant, extra in variants.items():
            variant_rows = []
            for seed in seeds:
                net = base_network(seed)
                for k, v in extra.items():
                    setattr(net, k, v)
                out = simulate_network(p=p, net=net)
                metrics = run_metrics(out)
                row = {
                    "sigma_noise": sigma,
                    "variant": variant,
                    "seed": seed,
                    **metrics,
                }
                rows.append(row)
                variant_rows.append(row)
                print(
                    f"sigma={sigma:.1f} {variant} seed={seed}: "
                    f"success={row['success']:.0f}, rt={row['recovery_time']}, tail={row['tail_span']}, "
                    f"err={row['peak_mean_error']:.3f}, turb={row['peak_mean_turbulence']:.3f}"
                )

            finite_rt = [float(r["recovery_time"]) for r in variant_rows if not np.isnan(float(r["recovery_time"]))]
            finite_tail = [float(r["tail_span"]) for r in variant_rows if not np.isnan(float(r["tail_span"]))]
            summary_rows.append(
                {
                    "sigma_noise": sigma,
                    "variant": variant,
                    "success_rate": float(np.mean([r["success"] for r in variant_rows])),
                    "mean_recovery_time": float(np.mean(finite_rt)) if finite_rt else float("nan"),
                    "mean_tail_span": float(np.mean(finite_tail)) if finite_tail else float("nan"),
                    "mean_peak_mean_s": float(np.mean([r["peak_mean_s"] for r in variant_rows])),
                    "mean_peak_error": float(np.mean([r["peak_mean_error"] for r in variant_rows])),
                    "mean_peak_turbulence": float(np.mean([r["peak_mean_turbulence"] for r in variant_rows])),
                }
            )

    csv_path = out_dir / "rc_evolution_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "rc_evolution_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    x = np.arange(len(sigmas))
    labels = [str(s) for s in sigmas]
    for variant in variants:
        sub = [row for row in summary_rows if row["variant"] == variant]
        axes[0].plot(x, [row["success_rate"] for row in sub], marker="o", label=variant)
        axes[1].plot(x, [row["mean_recovery_time"] for row in sub], marker="o", label=variant)
        axes[2].plot(x, [row["mean_tail_span"] for row in sub], marker="o", label=variant)
        axes[3].plot(x, [row["mean_peak_error"] for row in sub], marker="o", label=variant)
        axes[3].plot(x, [row["mean_peak_turbulence"] for row in sub], marker="x", linestyle="--", label=f"{variant} turbulence")

    axes[0].set_ylabel("success")
    axes[0].set_title("Success rate")
    axes[1].set_ylabel("recovery")
    axes[1].set_title("Mean recovery time")
    axes[2].set_ylabel("tail")
    axes[2].set_title("Mean tail span")
    axes[3].set_ylabel("field")
    axes[3].set_title("Mean error and turbulence")
    axes[3].set_xlabel("sigma_noise")
    axes[3].set_xticks(x, labels)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plot_path = out_dir / "rc_evolution_comparison_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
