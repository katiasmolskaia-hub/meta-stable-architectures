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


def episode_metrics(out: dict[str, np.ndarray], start: float, duration: float, threshold: float = 0.2) -> dict[str, float]:
    t = out["t"]
    mean_s = out["mean_s"]
    s = out["S"]
    stress_end = start + duration
    start_idx = int(np.searchsorted(t, stress_end, side="left"))

    recovery_time = float("nan")
    for i in range(start_idx, len(t)):
        if mean_s[i] <= threshold:
            recovery_time = float(t[i] - start)
            break

    per_agent = []
    for j in range(s.shape[1]):
        rt = float("nan")
        for i in range(start_idx, len(t)):
            if s[i, j] <= threshold:
                rt = float(t[i] - start)
                break
        if not np.isnan(rt):
            per_agent.append(rt)

    per = np.array(per_agent, dtype=float)
    return {
        "start": start,
        "duration": duration,
        "recovery_time": recovery_time,
        "agent_mean": float(np.mean(per)) if per.size else float("nan"),
        "agent_std": float(np.std(per)) if per.size else float("nan"),
        "agent_min": float(np.min(per)) if per.size else float("nan"),
        "agent_max": float(np.max(per)) if per.size else float("nan"),
    }


def summarize(out: dict[str, np.ndarray], label: str, windows: list[tuple[float, float]]) -> dict[str, float | str]:
    rows = [episode_metrics(out, start, dur) for start, dur in windows]
    return {
        "label": label,
        "final_recovery_time": float(rows[-1]["recovery_time"]),
        "final_agent_mean": float(rows[-1]["agent_mean"]),
        "final_agent_std": float(rows[-1]["agent_std"]),
        "final_agent_min": float(rows[-1]["agent_min"]),
        "final_agent_max": float(rows[-1]["agent_max"]),
        "final_group_memory": float(out["group_memory"][-1]),
        "final_dispersion": float(out["phase_dispersion"][-1]),
        "final_fraction_iso": float(out["fraction_isolated"][-1]),
        "mean_phi_gain": float(np.mean(out["phi_gain"])),
        "mean_kappa": float(np.mean(out["kappa"])),
        "ep1_rt": float(rows[0]["recovery_time"]),
        "ep2_rt": float(rows[1]["recovery_time"]),
        "ep3_rt": float(rows[2]["recovery_time"]),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = [
        (8.0, 2.5),
        (22.0, 2.5),
        (36.0, 2.5),
    ]

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
        stress_windows=tuple(windows),
    )
    kg_net = NetworkParams(
        **{**base_net.__dict__, "kg_enabled": True, "kg_lambda": 0.08, "kg_decay": 0.02, "kg_phi_boost": 2.0, "kg_wake_boost": 1.5}
    )

    base_out = simulate_network(p, base_net)
    kg_out = simulate_network(p, kg_net)

    summaries = [summarize(base_out, "qrc_no_kg", windows), summarize(kg_out, "qrc_with_kg", windows)]
    for row in summaries:
        print(
            f"{row['label']}: final_rt={row['final_recovery_time']:.3f}, "
            f"ep1={row['ep1_rt']:.3f}, ep2={row['ep2_rt']:.3f}, ep3={row['ep3_rt']:.3f}, "
            f"final_kg={row['final_group_memory']:.3f}, mean_phi={row['mean_phi_gain']:.3f}, mean_kappa={row['mean_kappa']:.3f}"
        )

    csv_path = out_dir / "network_kg_repeated_stress_tiny_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(base_out["t"], base_out["phase_dispersion"], label="no kg")
        axes[0].plot(kg_out["t"], kg_out["phase_dispersion"], label="with kg")
        axes[0].set_ylabel("phase dispersion")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(base_out["t"], base_out["fraction_isolated"], label="no kg")
        axes[1].plot(kg_out["t"], kg_out["fraction_isolated"], label="with kg")
        axes[1].set_ylabel("fraction isolated")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(base_out["t"], base_out["group_memory"], label="no kg")
        axes[2].plot(kg_out["t"], kg_out["group_memory"], label="with kg")
        axes[2].set_ylabel("group memory")
        axes[2].set_xlabel("time")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "network_kg_repeated_stress_tiny_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
