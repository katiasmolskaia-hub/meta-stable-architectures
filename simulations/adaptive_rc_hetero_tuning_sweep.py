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

from simulations.adaptive_rc_hetero_demo import HeteroDemoParams, recovery_times_per_agent, run_demo


def group_rec_time(S: np.ndarray, t: np.ndarray, stress_start: float, stress_end: float, threshold: float = 0.2) -> float:
    mean_s = np.mean(S, axis=1)
    for i, tt in enumerate(t):
        if tt > stress_end and mean_s[i] <= threshold:
            return float(tt - stress_start)
    return float("nan")


def summarize(label: str, out: dict[str, np.ndarray], stress_start: float, stress_end: float) -> dict[str, float | str]:
    per_agent = recovery_times_per_agent(out["S"], out["t"], stress_end)
    ok = per_agent[~np.isnan(per_agent)]
    return {
        "label": label,
        "group_recovery_time": group_rec_time(out["S"], out["t"], stress_start, stress_end),
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

    base = HeteroDemoParams()
    variants = [
        ("baseline", replace(base)),
        ("spread_friendly", replace(base, phi0=0.42, a1=0.55, a2=0.35, a3=0.25, kappa0=0.90, coh_gain=0.45, recog_gain=1.00)),
        ("responsive", replace(base, phi0=0.62, a1=1.05, a2=0.90, a3=0.75, kappa0=1.15, coh_gain=0.60, recog_gain=1.20)),
        ("lag_sensitive", replace(base, phi0=0.50, a1=0.75, a2=1.15, a3=0.45, kappa0=1.05, coh_gain=0.55, recog_gain=1.10)),
    ]

    rows: list[dict[str, float | str]] = []
    for label, params in variants:
        out = run_demo(params, seed=11, adaptive=True)
        rows.append(summarize(label, out, params.stress_time, params.stress_time + params.stress_duration))

    csv_path = out_dir / "adaptive_rc_hetero_tuning_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['label']}: group_rt={row['group_recovery_time']:.3f}, "
            f"agent_rt_std={row['agent_recovery_std']:.3f}, "
            f"agent_rt_p95={row['agent_recovery_p95']:.3f}, "
            f"final_S={row['final_mean_s']:.4f}, final_D={row['final_dispersion']:.4f}"
        )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
