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

from simulations.adaptive_rc_hetero_demo import HeteroDemoParams
from simulations.adaptive_rc_with_kg_multi_episode import run_with_kg
from simulations.adaptive_rc_hetero_demo import recovery_times_per_agent


WINDOWS = [
    (20.0, 6.0),
    (60.0, 8.0),
    (105.0, 10.0),
    (155.0, 12.0),
    (210.0, 14.0),
    (270.0, 16.0),
]


def summarize_run(run: dict[str, np.ndarray], windows: list[tuple[float, float]], label: str) -> dict[str, float | str]:
    rows = run["episode_stats"]  # type: ignore[index]
    tail_spans = [float(r["tail_span"]) for r in rows]
    recovery_times = [float(r["recovery_time"]) for r in rows]
    agent_stds = [float(r["agent_recovery_std"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    gates = [float(r["gate"]) for r in rows]
    return {
        "label": label,
        "final_mean_s": float(np.mean(run["S"][-1])),
        "final_dispersion": float(run["D"][-1]),
        "final_mean_k": float(np.mean(run["K"][-1])),
        "final_kg": float(run["Kg"][-1]),
        "mean_phi_gain": float(np.mean(run["phi_gain"])),
        "max_phi_gain": float(np.max(run["phi_gain"])),
        "mean_kappa": float(np.mean(run["kappa"])),
        "mean_recovery_time": float(np.mean(recovery_times)),
        "mean_tail_span": float(np.mean(tail_spans)),
        "mean_agent_std": float(np.mean(agent_stds)),
        "episode1_score": scores[0] if len(scores) > 0 else float("nan"),
        "episode2_score": scores[1] if len(scores) > 1 else float("nan"),
        "episode3_score": scores[2] if len(scores) > 2 else float("nan"),
        "episode4_score": scores[3] if len(scores) > 3 else float("nan"),
        "episode5_score": scores[4] if len(scores) > 4 else float("nan"),
        "episode6_score": scores[5] if len(scores) > 5 else float("nan"),
        "episode1_gate": gates[0] if len(gates) > 0 else float("nan"),
        "episode2_gate": gates[1] if len(gates) > 1 else float("nan"),
        "episode3_gate": gates[2] if len(gates) > 2 else float("nan"),
        "episode4_gate": gates[3] if len(gates) > 3 else float("nan"),
        "episode5_gate": gates[4] if len(gates) > 4 else float("nan"),
        "episode6_gate": gates[5] if len(gates) > 5 else float("nan"),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = HeteroDemoParams(
        t_end=360.0,
        stress_time=20.0,
        stress_duration=6.0,
        stress_amp=3.0,
        stress_y_amp=1.0,
        hetero_sigma=0.45,
        noise_scale=0.05,
        phi0=0.42,
        a1=0.55,
        a2=0.35,
        a3=0.25,
        kappa0=0.90,
        coh_gain=0.45,
        recog_gain=1.00,
    )

    base = run_with_kg(params, seed=13, use_kg=False, windows=WINDOWS)
    kg = run_with_kg(params, seed=13, use_kg=True, windows=WINDOWS, kg_strength=(0.50, 0.40, 0.20), kg_threshold=0.48, lambda_g=0.20)

    summaries = [
        summarize_run(base, WINDOWS, "no_kg"),
        summarize_run(kg, WINDOWS, "with_kg"),
    ]

    csv_path = out_dir / "adaptive_rc_with_kg_long_hetero_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    for row in summaries:
        print(
            f"{row['label']}: final_kg={row['final_kg']:.4f}, mean_recovery={row['mean_recovery_time']:.3f}, "
            f"mean_tail_span={row['mean_tail_span']:.3f}, mean_agent_std={row['mean_agent_std']:.3f}, "
            f"mean_phi_gain={row['mean_phi_gain']:.4f}, mean_kappa={row['mean_kappa']:.4f}"
        )
        print(
            f"  scores: {row['episode1_score']:.3f}, {row['episode2_score']:.3f}, {row['episode3_score']:.3f}, "
            f"{row['episode4_score']:.3f}, {row['episode5_score']:.3f}, {row['episode6_score']:.3f}"
        )

    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(base["t"], np.mean(base["S"], axis=1), label="no kg")
        axes[0].plot(kg["t"], np.mean(kg["S"], axis=1), label="with kg")
        axes[0].set_ylabel("mean S")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(base["t"], base["phi_gain"], label="no kg")
        axes[1].plot(kg["t"], kg["phi_gain"], label="with kg")
        axes[1].set_ylabel("phi gain")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(base["t"], np.mean(base["K"], axis=1), label="no kg")
        axes[2].plot(kg["t"], np.mean(kg["K"], axis=1), label="with kg")
        axes[2].set_ylabel("mean K")
        axes[2].set_xlabel("time")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "adaptive_rc_with_kg_long_hetero_test_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
