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

from simulations.adaptive_rc_multi_episode_check import run_multi_episode
from simulations.adaptive_rc_hetero_demo import HeteroDemoParams, recovery_times_per_agent


WINDOWS = [
    (20.0, 6.0),
    (55.0, 8.0),
    (95.0, 10.0),
    (140.0, 12.0),
]


def episode_metrics(out: dict[str, np.ndarray], stress_start: float, stress_duration: float) -> dict[str, float]:
    stress_end = stress_start + stress_duration
    t = out["t"]
    mean_s = np.mean(out["S"], axis=1)
    phi_gain = out["phi_gain"]
    kappa = out["kappa"]

    pre_mask = (t >= max(0.0, stress_start - 5.0)) & (t < stress_start)
    post_mask = (t >= stress_start) & (t <= stress_end + 6.0)
    post_recovery_mask = t > stress_end

    phi_base = float(np.mean(phi_gain[pre_mask])) if np.any(pre_mask) else float(phi_gain[0])
    kappa_base = float(np.mean(kappa[pre_mask])) if np.any(pre_mask) else float(kappa[0])
    s_base = float(np.mean(mean_s[pre_mask])) if np.any(pre_mask) else float(mean_s[0])

    phi_peak_idx = int(np.argmax(phi_gain[post_mask])) if np.any(post_mask) else 0
    kappa_peak_idx = int(np.argmax(kappa[post_mask])) if np.any(post_mask) else 0
    s_peak_idx = int(np.argmax(mean_s[post_mask])) if np.any(post_mask) else 0

    post_t = t[post_mask]
    phi_peak_time = float(post_t[phi_peak_idx]) if post_t.size else stress_start
    kappa_peak_time = float(post_t[kappa_peak_idx]) if post_t.size else stress_start
    s_peak_time = float(post_t[s_peak_idx]) if post_t.size else stress_start

    recovery_time = float("nan")
    for i in range(len(t)):
        if t[i] > stress_end and mean_s[i] <= 0.2:
            recovery_time = float(t[i] - stress_start)
            break

    per_agent = recovery_times_per_agent(out["S"], t, stress_end)
    ok = per_agent[~np.isnan(per_agent)]

    return {
        "stress_start": stress_start,
        "stress_duration": stress_duration,
        "phi_base": phi_base,
        "kappa_base": kappa_base,
        "s_base": s_base,
        "phi_peak_time": phi_peak_time - stress_start,
        "kappa_peak_time": kappa_peak_time - stress_start,
        "s_peak_time": s_peak_time - stress_start,
        "recovery_time": recovery_time,
        "agent_recovery_mean": float(np.mean(ok)) if ok.size else float("nan"),
        "agent_recovery_std": float(np.std(ok)) if ok.size else float("nan"),
        "agent_recovery_min": float(np.min(ok)) if ok.size else float("nan"),
        "agent_recovery_max": float(np.max(ok)) if ok.size else float("nan"),
        "agent_recovery_p50": float(np.percentile(ok, 50)) if ok.size else float("nan"),
        "agent_recovery_p90": float(np.percentile(ok, 90)) if ok.size else float("nan"),
        "tail_span": float(np.max(ok) - np.min(ok)) if ok.size else float("nan"),
        "frac_gt_0p2_tplus2": float(np.mean(out["S"][min(len(t) - 1, int(np.searchsorted(t, stress_end + 2.0, side="left")))] > 0.2)),
        "frac_gt_0p2_tplus4": float(np.mean(out["S"][min(len(t) - 1, int(np.searchsorted(t, stress_end + 4.0, side="left")))] > 0.2)),
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

    presets = [
        ("baseline", replace(base, phi0=0.62, a1=1.05, a2=0.90, a3=0.75, kappa0=1.15, coh_gain=0.60, recog_gain=1.20)),
        ("spread_friendly", replace(base, phi0=0.42, a1=0.55, a2=0.35, a3=0.25, kappa0=0.90, coh_gain=0.45, recog_gain=1.00)),
        ("responsive", replace(base, phi0=0.68, a1=1.15, a2=1.00, a3=0.80, kappa0=1.20, coh_gain=0.62, recog_gain=1.22)),
    ]

    rows: list[dict[str, float | str]] = []
    for label, params in presets:
        out = run_multi_episode(params, seed=11, adaptive=True)
        for episode_idx, (start, duration) in enumerate(WINDOWS, 1):
            row = episode_metrics(out, start, duration)
            row["preset"] = label
            row["episode"] = float(episode_idx)
            rows.append(row)

    csv_path = out_dir / "adaptive_rc_long_term_lag_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['preset']} ep{int(row['episode'])}: "
            f"phi_lag={row['phi_peak_time']:.3f}, kappa_lag={row['kappa_peak_time']:.3f}, "
            f"recovery={row['recovery_time']:.3f}, tail_span={row['tail_span']:.3f}, "
            f"t+2={row['frac_gt_0p2_tplus2']:.3f}, t+4={row['frac_gt_0p2_tplus4']:.3f}"
        )

    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for label in {r["preset"] for r in rows}:
            sub = [r for r in rows if r["preset"] == label]
            episodes = [r["episode"] for r in sub]
            axes[0].plot(episodes, [r["phi_peak_time"] for r in sub], marker="o", label=label)
            axes[1].plot(episodes, [r["recovery_time"] for r in sub], marker="o", label=label)
            axes[2].plot(episodes, [r["tail_span"] for r in sub], marker="o", label=label)

        axes[0].set_ylabel("phi lag")
        axes[1].set_ylabel("recovery time")
        axes[2].set_ylabel("tail span")
        axes[2].set_xlabel("episode")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        plot_path = out_dir / "adaptive_rc_long_term_lag_check_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
