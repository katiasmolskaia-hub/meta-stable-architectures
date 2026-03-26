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


WINDOWS = (
    (8.0, 2.5),
    (18.0, 2.5),
    (28.0, 2.5),
)


def episode_metrics(out: dict[str, np.ndarray], start: float, duration: float, threshold: float = 0.2) -> dict[str, float]:
    t = out["t"]
    mean_s = out["mean_s"]
    s = out["S"]
    stress_end = start + duration
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
        "start": start,
        "duration": duration,
        "recovery_time": recovery_time,
        "agent_std": float(np.std(per)) if per.size else float("nan"),
        "agent_p90": float(np.percentile(per, 90)) if per.size else float("nan"),
        "tail_span": float(np.max(per) - np.min(per)) if per.size else float("nan"),
        "peak_mean_error": float(np.max(out["mean_error"][start_idx:])),
        "peak_mean_anchor": float(np.max(out["mean_anchor"][start_idx:])),
        "mean_access": float(np.mean(out["mean_access"][start_idx:])),
        "final_dispersion": float(out["phase_dispersion"][-1]),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_agents = 36
    # Three practical cohorts:
    # novices lag more and are noisier, intermediates are mixed,
    # experienced agents are more stable and can serve as anchors more often.
    alpha = np.concatenate(
        [
            np.full(12, 0.78),
            np.full(12, 0.86),
            np.full(12, 0.95),
        ]
    )
    sigma_noise = np.concatenate(
        [
            np.full(12, 0.30),
            np.full(12, 0.22),
            np.full(12, 0.14),
        ]
    )
    h_crit = np.concatenate(
        [
            np.full(12, 0.62),
            np.full(12, 0.69),
            np.full(12, 0.77),
        ]
    )
    p = MasterParams(alpha=alpha, sigma_noise=sigma_noise, h_crit=h_crit)

    common = dict(
        n_agents=n_agents,
        t_end=40.0,
        dt=0.02,
        topology="spatial_local",
        spatial_radius=0.34,
        spatial_falloff=0.16,
        coupling=0.17,
        delay_mode="grouped",
        delay_group_fracs=(0.34, 0.33, 0.33),
        delay_group_steps=(0, 3, 7),
        stress_windows=WINDOWS,
        stress_frac=0.28,
        stress_amp=3.1,
        stress_y_amp=1.0,
        qrc_enabled=True,
        phi_kappa=1.0,
        recog_threshold=0.7,
        recog_gain=1.2,
        wake_time_required=4.0,
        wake_relax_gain=0.8,
        coh_relax_gain=0.6,
    )

    fixed_rc = NetworkParams(
        **{
            **common,
            "phi_gain": 0.42,
            "phi_gain_boost": 0.0,
            "ccrit_gain": 0.0,
        }
    )
    adaptive_rc = NetworkParams(
        **{
            **common,
            "phi_gain": 0.42,
            "phi_gain_boost": 8.0,
            "ccrit_gain": 0.6,
        }
    )
    instructor_v2 = NetworkParams(
        **{
            **common,
            "phi_gain": 0.42,
            "phi_gain_boost": 8.0,
            "ccrit_gain": 0.6,
            "instructor_enabled": True,
            "instructor_error_weight": 0.75,
            "instructor_anchor_weight": 0.85,
            "instructor_contagion_weight": 0.75,
        }
    )

    runs = [
        ("fixed_rc", simulate_network(p, fixed_rc)),
        ("adaptive_rc", simulate_network(p, adaptive_rc)),
        ("instructor_v2", simulate_network(p, instructor_v2)),
    ]

    rows = []
    run_rows = []
    for label, out in runs:
        episode_rows = [episode_metrics(out, *window) for window in WINDOWS]
        row = {
            "label": label,
            "ep1_rt": float(episode_rows[0]["recovery_time"]),
            "ep2_rt": float(episode_rows[1]["recovery_time"]),
            "ep3_rt": float(episode_rows[2]["recovery_time"]),
            "ep3_std": float(episode_rows[2]["agent_std"]),
            "ep3_p90": float(episode_rows[2]["agent_p90"]),
            "ep3_tail": float(episode_rows[2]["tail_span"]),
            "peak_error": float(np.max(out["mean_error"])),
            "anchor_floor": float(np.min(out["mean_anchor"])),
            "access_mean": float(np.mean(out["mean_access"])),
            "final_dispersion": float(out["phase_dispersion"][-1]),
        }
        run_rows.append(row)
        for episode_idx, ep in enumerate(episode_rows, 1):
            rows.append({"label": label, "episode": episode_idx, **ep})
        print(
            f"{label}: ep1={row['ep1_rt']:.3f}, ep2={row['ep2_rt']:.3f}, ep3={row['ep3_rt']:.3f}, "
            f"ep3_std={row['ep3_std']:.3f}, ep3_p90={row['ep3_p90']:.3f}, "
            f"peak_error={row['peak_error']:.3f}, anchor_floor={row['anchor_floor']:.3f}, access={row['access_mean']:.3f}"
        )

    csv_path = out_dir / "instructor_v2_spatial_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    metrics_path = out_dir / "instructor_v2_spatial_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for label, out in runs:
        axes[0].plot(out["t"], out["mean_s"], label=label)
        axes[1].plot(out["t"], out["mean_error"], label=label)
        axes[2].plot(out["t"], out["mean_anchor"], label=label)
        axes[3].plot(out["t"], out["mean_access"], label=label)

    axes[0].set_ylabel("mean S")
    axes[0].set_title("Suppression")
    axes[1].set_ylabel("mean error")
    axes[1].set_title("Local error proxy")
    axes[2].set_ylabel("mean anchor")
    axes[2].set_title("Anchor reliability")
    axes[3].set_ylabel("mean access")
    axes[3].set_title("Visibility / hearing quality")
    axes[3].set_xlabel("time")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plot_path = out_dir / "instructor_v2_spatial_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
