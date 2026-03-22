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

from simulations.adaptive_rc_hetero_demo import HeteroDemoParams, recovery_times_per_agent, run_demo


def repeated_stress_windows() -> list[tuple[float, float]]:
    return [
        (20.0, 6.0),
        (55.0, 8.0),
        (95.0, 10.0),
        (140.0, 12.0),
    ]


def episode_summary(out: dict[str, np.ndarray], stress_start: float, stress_duration: float) -> dict[str, float]:
    stress_end = stress_start + stress_duration
    mean_s = np.mean(out["S"], axis=1)
    rec_time = float("nan")
    for i, tt in enumerate(out["t"]):
        if tt > stress_end and mean_s[i] <= 0.2:
            rec_time = float(tt - stress_start)
            break
    per_agent = recovery_times_per_agent(out["S"], out["t"], stress_end)
    ok = per_agent[~np.isnan(per_agent)]
    return {
        "stress_start": stress_start,
        "stress_duration": stress_duration,
        "group_recovery_time": rec_time,
        "agent_recovery_mean": float(np.mean(ok)) if ok.size else float("nan"),
        "agent_recovery_std": float(np.std(ok)) if ok.size else float("nan"),
        "agent_recovery_p95": float(np.percentile(ok, 95)) if ok.size else float("nan"),
        "final_mean_s": float(np.mean(out["S"][-1])),
        "final_dispersion": float(out["D"][-1]),
        "final_mean_k": float(np.mean(out["K"][-1])),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = HeteroDemoParams(
        t_end=200.0,
        stress_time=20.0,
        stress_duration=6.0,
        stress_amp=2.7,
        stress_y_amp=0.9,
        hetero_sigma=0.35,
        noise_scale=0.04,
    )
    responsive = replace(base, phi0=0.62, a1=1.05, a2=0.90, a3=0.75, kappa0=1.15, coh_gain=0.60, recog_gain=1.20)

    out = run_demo(responsive, seed=11, adaptive=True)

    rows: list[dict[str, float]] = []
    for stress_start, stress_duration in repeated_stress_windows():
        # The demo internally uses a single stress window, so we record the
        # protocol windows used for interpretation of the long horizon.
        rows.append({
            "stress_start": stress_start,
            "stress_duration": stress_duration,
        })

    summary = episode_summary(out, responsive.stress_time, responsive.stress_duration)

    csv_path = out_dir / "adaptive_rc_long_run_check_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Long-run group recovery time: {summary['group_recovery_time']:.3f}")
    print(f"Long-run agent recovery std: {summary['agent_recovery_std']:.3f}")
    print(f"Long-run agent recovery p95: {summary['agent_recovery_p95']:.3f}")
    print(f"Final mean S: {summary['final_mean_s']:.4f}")
    print(f"Final dispersion: {summary['final_dispersion']:.4f}")
    print(f"Final mean K: {summary['final_mean_k']:.4f}")
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
        plot_path = out_dir / "adaptive_rc_long_run_check_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
