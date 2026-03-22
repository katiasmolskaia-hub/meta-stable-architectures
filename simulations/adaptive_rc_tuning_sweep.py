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

from simulations.adaptive_rc_first_demo import DemoParams, run_demo


def recovery_times_per_agent(S: np.ndarray, t: np.ndarray, stress_end: float, threshold: float = 0.2) -> np.ndarray:
    out = np.full(S.shape[1], np.nan, dtype=float)
    start_idx = int(np.searchsorted(t, stress_end, side="right"))
    for j in range(S.shape[1]):
        for i in range(start_idx, len(t)):
            if S[i, j] <= threshold:
                out[j] = float(t[i] - stress_end)
                break
    return out


def summarize(label: str, out: dict[str, np.ndarray], stress_start: float, stress_end: float) -> dict[str, float | str]:
    mean_s = np.mean(out["S"], axis=1)
    rec_time = float("nan")
    for i, tt in enumerate(out["t"]):
        if tt > stress_end and mean_s[i] <= 0.2:
            rec_time = float(tt - stress_start)
            break
    per_agent = recovery_times_per_agent(out["S"], out["t"], stress_end)
    recovered = per_agent[~np.isnan(per_agent)]
    return {
        "label": label,
        "group_recovery_time": rec_time,
        "agent_recovery_mean": float(np.mean(recovered)) if recovered.size else float("nan"),
        "agent_recovery_std": float(np.std(recovered)) if recovered.size else float("nan"),
        "agent_recovery_p95": float(np.percentile(recovered, 95)) if recovered.size else float("nan"),
        "final_mean_s": float(np.mean(out["S"][-1])),
        "final_dispersion": float(out["D"][-1]),
        "final_mean_k": float(np.mean(out["K"][-1])),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = DemoParams()
    variants = [
        ("conservative", replace(base, phi0=0.40, a1=0.45, a2=0.30, a3=0.25, kappa0=0.90)),
        ("baseline", replace(base)),
        ("responsive", replace(base, phi0=0.60, a1=1.05, a2=0.85, a3=0.70, kappa0=1.15)),
        ("lag_sensitive", replace(base, phi0=0.50, a1=0.80, a2=1.20, a3=0.50, kappa0=1.10)),
    ]

    rows: list[dict[str, float | str]] = []
    for idx, (label, params) in enumerate(variants):
        out = run_demo(params, seed=7, adaptive=True)
        rows.append(summarize(label, out, params.stress_time, params.stress_time + params.stress_duration))

    csv_path = out_dir / "adaptive_rc_tuning_sweep.csv"
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
