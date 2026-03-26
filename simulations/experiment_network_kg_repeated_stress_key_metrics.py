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

from simulations.simulation_network_v1 import MasterParams, NetworkParams, simulate_network


WINDOWS = [
    (8.0, 2.5),
    (22.0, 2.5),
    (36.0, 2.5),
]


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
    if per.size:
        p10 = float(np.percentile(per, 10))
        p50 = float(np.percentile(per, 50))
        p90 = float(np.percentile(per, 90))
        agent_min = float(np.min(per))
        agent_max = float(np.max(per))
        agent_std = float(np.std(per))
    else:
        p10 = p50 = p90 = agent_min = agent_max = agent_std = float("nan")

    spread = float(p90 - p10) if np.isfinite(p10) and np.isfinite(p90) else float("nan")
    tail_span = float(agent_max - agent_min) if np.isfinite(agent_max) and np.isfinite(agent_min) else float("nan")
    p90_over_max = float(p90 / agent_max) if np.isfinite(p90) and np.isfinite(agent_max) and abs(agent_max) > 1e-12 else float("nan")

    return {
        "start": start,
        "duration": duration,
        "recovery_time": recovery_time,
        "delta_recovery_time_prev": float("nan"),
        "agent_std": agent_std,
        "agent_min": agent_min,
        "agent_p10": p10,
        "agent_p50": p50,
        "agent_p90": p90,
        "agent_max": agent_max,
        "spread": spread,
        "tail_span": tail_span,
        "p90_over_max": p90_over_max,
    }


def summarize_run(out: dict[str, np.ndarray], label: str) -> dict[str, float | str]:
    return {
        "label": label,
        "final_group_memory": float(out["group_memory"][-1]),
        "peak_group_memory": float(np.max(out["group_memory"])),
        "final_dispersion": float(out["phase_dispersion"][-1]),
        "final_fraction_iso": float(out["fraction_isolated"][-1]),
        "mean_phi_gain": float(np.mean(out["phi_gain"])),
        "mean_kappa": float(np.mean(out["kappa"])),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = MasterParams()
    base_net = NetworkParams(
        n_agents=24,
        t_end=55.0,
        dt=0.02,
        coupling=0.15,
        topology="small_world",
        ring_k=2,
        sw_rewire=0.15,
        delay_mode="grouped",
        delay_group_fracs=(0.5, 0.3, 0.2),
        delay_group_steps=(0, 2, 4),
        stress_amp=3.0,
        stress_frac=0.24,
        stress_y_amp=1.0,
        qrc_enabled=True,
        phi_kappa=1.0,
        phi_gain=0.5,
        phi_gain_boost=8.0,
        qrc_g_min=0.4,
        recog_threshold=0.7,
        recog_gain=1.2,
        wake_disp_threshold=0.3,
        wake_time_required=4.0,
        wake_relax_gain=0.8,
        coh_relax_gain=0.6,
        stress_windows=tuple(WINDOWS),
    )
    kg_net = NetworkParams(
        **{
            **base_net.__dict__,
            "kg_enabled": True,
            "kg_lambda": 0.08,
            "kg_decay": 0.02,
            "kg_phi_boost": 2.0,
            "kg_wake_boost": 1.5,
            "kg_crisis_threshold": 0.45,
        }
    )

    runs = [
        ("qrc_no_kg", simulate_network(p, base_net)),
        ("qrc_with_kg", simulate_network(p, kg_net)),
    ]

    rows: list[dict[str, float | str]] = []
    run_rows: list[dict[str, float | str]] = []

    for label, out in runs:
        run_rows.append(summarize_run(out, label))
        prev_rt = float("nan")
        for episode_idx, (start, duration) in enumerate(WINDOWS, 1):
            row = episode_metrics(out, start, duration)
            row["label"] = label
            row["episode"] = float(episode_idx)
            if episode_idx > 1 and np.isfinite(prev_rt) and np.isfinite(row["recovery_time"]):
                row["delta_recovery_time_prev"] = float(row["recovery_time"] - prev_rt)
            prev_rt = float(row["recovery_time"])
            row["final_group_memory"] = float(out["group_memory"][-1])
            row["peak_group_memory"] = float(np.max(out["group_memory"]))
            row["final_dispersion"] = float(out["phase_dispersion"][-1])
            row["mean_phi_gain"] = float(np.mean(out["phi_gain"]))
            row["mean_kappa"] = float(np.mean(out["kappa"]))
            rows.append(row)

    csv_path = out_dir / "network_kg_repeated_stress_key_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    run_csv = out_dir / "network_kg_repeated_stress_key_metrics_runs.csv"
    with run_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    for row in run_rows:
        print(
            f"{row['label']}: final_kg={row['final_group_memory']:.3f}, "
            f"peak_kg={row['peak_group_memory']:.3f}, "
            f"final_dispersion={row['final_dispersion']:.4f}, "
            f"mean_phi={row['mean_phi_gain']:.3f}"
        )

    for row in rows:
        print(
            f"{row['label']} ep{int(row['episode'])}: rt={row['recovery_time']:.3f}, "
            f"delta_prev={row['delta_recovery_time_prev']:.3f}, "
            f"spread={row['spread']:.3f}, tail_span={row['tail_span']:.3f}, "
            f"p90={row['agent_p90']:.3f}, max={row['agent_max']:.3f}, std={row['agent_std']:.3f}"
        )

    print(f"Wrote {csv_path}")
    print(f"Wrote {run_csv}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        axes[0].plot(runs[0][1]["t"], runs[0][1]["phase_dispersion"], label="no kg")
        axes[0].plot(runs[1][1]["t"], runs[1][1]["phase_dispersion"], label="with kg")
        axes[0].set_ylabel("phase dispersion")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(runs[0][1]["t"], runs[0][1]["fraction_isolated"], label="no kg")
        axes[1].plot(runs[1][1]["t"], runs[1][1]["fraction_isolated"], label="with kg")
        axes[1].set_ylabel("fraction isolated")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(runs[0][1]["t"], runs[0][1]["group_memory"], label="no kg")
        axes[2].plot(runs[1][1]["t"], runs[1][1]["group_memory"], label="with kg")
        axes[2].set_ylabel("group memory")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(runs[0][1]["t"], runs[0][1]["mean_s"], label="no kg")
        axes[3].plot(runs[1][1]["t"], runs[1][1]["mean_s"], label="with kg")
        axes[3].set_ylabel("mean S")
        axes[3].set_xlabel("time")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "network_kg_repeated_stress_key_metrics_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
