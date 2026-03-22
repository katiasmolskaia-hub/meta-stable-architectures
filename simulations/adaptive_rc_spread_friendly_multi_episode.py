from __future__ import annotations

from dataclasses import replace
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
from simulations.adaptive_rc_multi_episode_check import run_multi_episode, episode_summary
from simulations.adaptive_rc_hetero_demo import recovery_times_per_agent


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = HeteroDemoParams(
        t_end=200.0,
        stress_time=20.0,
        stress_duration=6.0,
        stress_amp=2.7,
        stress_y_amp=0.9,
        hetero_sigma=0.35,
        noise_scale=0.04,
        phi0=0.42,
        a1=0.55,
        a2=0.35,
        a3=0.25,
        kappa0=0.90,
        coh_gain=0.45,
        recog_gain=1.00,
    )

    out = run_multi_episode(params, seed=11, adaptive=True)

    windows = [
        (20.0, 6.0),
        (55.0, 8.0),
        (95.0, 10.0),
        (140.0, 12.0),
    ]

    rows: list[dict[str, float]] = []
    for i, (start, duration) in enumerate(windows, 1):
        row = episode_summary(out, start, duration)
        per = recovery_times_per_agent(out["S"], out["t"], start + duration)
        ok = per[~np.isnan(per)]
        row.update(
            {
                "episode": float(i),
                "min_agent_rt": float(np.min(ok)) if ok.size else float("nan"),
                "mean_agent_rt": float(np.mean(ok)) if ok.size else float("nan"),
                "max_agent_rt": float(np.max(ok)) if ok.size else float("nan"),
                "p10_agent_rt": float(np.percentile(ok, 10)) if ok.size else float("nan"),
                "p50_agent_rt": float(np.percentile(ok, 50)) if ok.size else float("nan"),
                "p90_agent_rt": float(np.percentile(ok, 90)) if ok.size else float("nan"),
            }
        )
        rows.append(row)

    csv_path = out_dir / "adaptive_rc_spread_friendly_multi_episode.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"episode {int(row['episode'])}: group_rt={row['group_recovery_time']:.3f}, "
            f"min={row['min_agent_rt']:.3f}, mean={row['mean_agent_rt']:.3f}, max={row['max_agent_rt']:.3f}, "
            f"p10={row['p10_agent_rt']:.3f}, p50={row['p50_agent_rt']:.3f}, p90={row['p90_agent_rt']:.3f}"
        )
    print(f"Final mean S: {float(np.mean(out['S'][-1])):.4f}")
    print(f"Final dispersion: {float(out['D'][-1]):.4f}")
    print(f"Final mean K: {float(np.mean(out['K'][-1])):.4f}")
    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(out["t"], np.mean(out["S"], axis=1))
        axes[0, 0].set_title("Mean suppression")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(out["t"], out["D"])
        axes[0, 1].set_title("Phase dispersion")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(out["t"], np.mean(out["K"], axis=1))
        axes[1, 0].set_title("Mean K")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(out["t"], out["phi_gain"])
        axes[1, 1].set_title("Phi gain")
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "adaptive_rc_spread_friendly_multi_episode_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
