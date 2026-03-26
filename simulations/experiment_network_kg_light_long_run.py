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
    (4.0, 1.5),
    (11.0, 1.5),
    (18.0, 1.5),
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

    def frac(when: float) -> float:
        idx = min(len(t) - 1, int(np.searchsorted(t, when, side="left")))
        return float(np.mean(s[idx] > threshold))

    return {
        "start": start,
        "duration": duration,
        "recovery_time": recovery_time,
        "agent_mean": float(np.mean(per)) if per.size else float("nan"),
        "agent_std": float(np.std(per)) if per.size else float("nan"),
        "agent_min": float(np.min(per)) if per.size else float("nan"),
        "agent_max": float(np.max(per)) if per.size else float("nan"),
        "agent_p90": float(np.percentile(per, 90)) if per.size else float("nan"),
        "tail_span": float(np.max(per) - np.min(per)) if per.size else float("nan"),
        "frac_t1": frac(stress_end + 1.0),
        "frac_t2": frac(stress_end + 2.0),
        "frac_t4": frac(stress_end + 4.0),
    }


def summarize(out: dict[str, np.ndarray], label: str) -> dict[str, float | str]:
    rows = [episode_metrics(out, start, dur) for start, dur in WINDOWS]
    return {
        "label": label,
        "final_group_memory": float(out["group_memory"][-1]),
        "final_dispersion": float(out["phase_dispersion"][-1]),
        "final_fraction_iso": float(out["fraction_isolated"][-1]),
        "mean_phi_gain": float(np.mean(out["phi_gain"])),
        "mean_kappa": float(np.mean(out["kappa"])),
        "ep1_rt": float(rows[0]["recovery_time"]),
        "ep2_rt": float(rows[1]["recovery_time"]),
        "ep3_rt": float(rows[2]["recovery_time"]),
        "ep3_tail_span": float(rows[2]["tail_span"]),
        "ep3_agent_std": float(rows[2]["agent_std"]),
        "ep3_p90": float(rows[2]["agent_p90"]),
        "ep3_tplus1": float(rows[2]["frac_t1"]),
        "ep3_tplus2": float(rows[2]["frac_t2"]),
        "ep3_tplus4": float(rows[2]["frac_t4"]),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = MasterParams()
    base = NetworkParams(
        n_agents=12,
        t_end=30.0,
        dt=0.02,
        coupling=0.15,
        topology="ring",
        ring_k=2,
        delay_mode="grouped",
        delay_group_fracs=(0.5, 0.3, 0.2),
        delay_group_steps=(0, 1, 2),
        stress_amp=3.0,
        stress_frac=0.25,
        stress_duration=1.5,
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
    with_kg = NetworkParams(
        **{
            **base.__dict__,
            "kg_enabled": True,
            "kg_lambda": 0.08,
            "kg_decay": 0.02,
            "kg_phi_boost": 1.8,
            "kg_wake_boost": 1.2,
            "kg_crisis_threshold": 0.45,
        }
    )

    no_kg_out = simulate_network(p, base)
    kg_out = simulate_network(p, with_kg)

    summaries = [summarize(no_kg_out, "network_no_kg"), summarize(kg_out, "network_with_kg")]
    for row in summaries:
        print(
            f"{row['label']}: final_kg={row['final_group_memory']:.3f}, "
            f"mean_phi={row['mean_phi_gain']:.3f}, mean_kappa={row['mean_kappa']:.3f}, "
            f"ep3_rt={row['ep3_rt']:.3f}, ep3_std={row['ep3_agent_std']:.3f}, ep3_p90={row['ep3_p90']:.3f}, "
            f"t+1={row['ep3_tplus1']:.3f}, t+2={row['ep3_tplus2']:.3f}, t+4={row['ep3_tplus4']:.3f}"
        )

    csv_path = out_dir / "network_kg_light_long_run.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(no_kg_out["t"], no_kg_out["phase_dispersion"], label="no kg")
        axes[0].plot(kg_out["t"], kg_out["phase_dispersion"], label="with kg")
        axes[0].set_ylabel("phase dispersion")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(no_kg_out["t"], no_kg_out["fraction_isolated"], label="no kg")
        axes[1].plot(kg_out["t"], kg_out["fraction_isolated"], label="with kg")
        axes[1].set_ylabel("fraction isolated")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(no_kg_out["t"], no_kg_out["group_memory"], label="no kg")
        axes[2].plot(kg_out["t"], kg_out["group_memory"], label="with kg")
        axes[2].set_ylabel("group memory")
        axes[2].set_xlabel("time")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "network_kg_light_long_run_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
