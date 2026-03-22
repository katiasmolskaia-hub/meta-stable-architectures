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

from simulations.adaptive_rc_hetero_demo import HeteroDemoParams, recovery_times_per_agent
from simulations.adaptive_rc_multi_episode_check import run_multi_episode


WINDOWS = [
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
        "agent_recovery_min": float(np.min(ok)) if ok.size else float("nan"),
        "agent_recovery_max": float(np.max(ok)) if ok.size else float("nan"),
        "agent_recovery_p10": float(np.percentile(ok, 10)) if ok.size else float("nan"),
        "agent_recovery_p50": float(np.percentile(ok, 50)) if ok.size else float("nan"),
        "agent_recovery_p90": float(np.percentile(ok, 90)) if ok.size else float("nan"),
        "final_mean_s": float(np.mean(out["S"][-1])),
        "final_dispersion": float(out["D"][-1]),
        "final_mean_k": float(np.mean(out["K"][-1])),
    }


def infection_profile(out: dict[str, np.ndarray], threshold: float = 0.2) -> dict[str, np.ndarray]:
    S = out["S"]
    t = out["t"]
    frac = np.mean(S > threshold, axis=1)
    frac_mid = np.mean(S > 0.5, axis=1)
    frac_high = np.mean(S > 0.8, axis=1)
    return {
        "t": t,
        "frac_gt_0p2": frac,
        "frac_gt_0p5": frac_mid,
        "frac_gt_0p8": frac_high,
    }


def fraction_at_time(out: dict[str, np.ndarray], when: float, threshold: float = 0.2) -> float:
    idx = int(np.searchsorted(out["t"], when, side="left"))
    idx = min(idx, len(out["t"]) - 1)
    return float(np.mean(out["S"][idx] > threshold))


def recovery_matrix(out: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for start, duration in WINDOWS:
        rows.append(recovery_times_per_agent(out["S"], out["t"], start + duration))
    return np.vstack(rows)


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

    summaries: list[dict[str, float | str]] = []
    profile_rows: list[dict[str, float | str]] = []
    mats: dict[str, np.ndarray] = {}
    profiles: dict[str, dict[str, np.ndarray]] = {}

    for label, params in presets:
        out = run_multi_episode(params, seed=11, adaptive=True)
        profiles[label] = infection_profile(out)
        mats[label] = recovery_matrix(out)

        for ep_idx, (start, duration) in enumerate(WINDOWS, 1):
            row = episode_summary(out, start, duration)
            row["preset"] = label
            row["episode"] = float(ep_idx)
            row["frac_gt_0p2_tplus1"] = fraction_at_time(out, start + duration + 1.0)
            row["frac_gt_0p2_tplus2"] = fraction_at_time(out, start + duration + 2.0)
            row["frac_gt_0p2_tplus4"] = fraction_at_time(out, start + duration + 4.0)
            row["frac_gt_0p2_tplus6"] = fraction_at_time(out, start + duration + 6.0)
            row["recovery_span"] = float(row["agent_recovery_max"] - row["agent_recovery_min"])
            summaries.append(row)

        frac = profiles[label]
        profile_rows.append(
            {
                "preset": label,
                "t0_frac_gt_0p2": float(frac["frac_gt_0p2"][0]),
                "tpeak_frac_gt_0p2": float(np.max(frac["frac_gt_0p2"])),
                "tpeak_frac_gt_0p5": float(np.max(frac["frac_gt_0p5"])),
                "tpeak_frac_gt_0p8": float(np.max(frac["frac_gt_0p8"])),
                "final_frac_gt_0p2": float(frac["frac_gt_0p2"][-1]),
                "final_frac_gt_0p5": float(frac["frac_gt_0p5"][-1]),
                "final_frac_gt_0p8": float(frac["frac_gt_0p8"][-1]),
            }
        )

    summary_csv = out_dir / "adaptive_rc_infection_profile_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    profile_csv = out_dir / "adaptive_rc_infection_profile_overview.csv"
    with profile_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    for row in profile_rows:
        print(
            f"{row['preset']}: peak>0.2={row['tpeak_frac_gt_0p2']:.3f}, "
            f"peak>0.5={row['tpeak_frac_gt_0p5']:.3f}, peak>0.8={row['tpeak_frac_gt_0p8']:.3f}, "
            f"final>0.2={row['final_frac_gt_0p2']:.3f}"
        )

    print(f"Wrote {summary_csv}")
    print(f"Wrote {profile_csv}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(presets), 2, figsize=(13, 10), sharex="col")
        if len(presets) == 1:
            axes = np.array([axes])

        for row_idx, (label, _) in enumerate(presets):
            frac = profiles[label]
            mat = mats[label]
            ax_line = axes[row_idx, 0]
            ax_heat = axes[row_idx, 1]

            ax_line.plot(frac["t"], frac["frac_gt_0p2"], label="S > 0.2")
            ax_line.plot(frac["t"], frac["frac_gt_0p5"], label="S > 0.5")
            ax_line.plot(frac["t"], frac["frac_gt_0p8"], label="S > 0.8")
            ax_line.set_ylabel(label)
            ax_line.grid(True, alpha=0.3)
            if row_idx == 0:
                ax_line.legend(loc="upper right")

            im = ax_heat.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=8.5)
            ax_heat.set_yticks(range(len(WINDOWS)))
            ax_heat.set_yticklabels([f"ep {i}" for i in range(1, len(WINDOWS) + 1)])
            ax_heat.set_ylabel(label)
            if row_idx == 0:
                ax_heat.set_title("Recovery-time heatmap by agent")

        axes[-1, 0].set_xlabel("time")
        axes[-1, 1].set_xlabel("agent index")
        fig.colorbar(im, ax=axes[:, 1].ravel().tolist(), shrink=0.8, label="recovery time")
        fig.tight_layout()
        plot_path = out_dir / "adaptive_rc_infection_profile_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
