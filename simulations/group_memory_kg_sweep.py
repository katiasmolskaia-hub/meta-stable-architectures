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
from simulations.adaptive_rc_multi_episode_check import run_multi_episode, recovery_times_per_agent


WINDOWS = [
    (20.0, 6.0),
    (55.0, 8.0),
    (95.0, 10.0),
    (140.0, 12.0),
]


def episode_features(out: dict[str, np.ndarray], stress_start: float, stress_duration: float) -> dict[str, float]:
    stress_end = stress_start + stress_duration
    t = out["t"]
    S = out["S"]
    mean_s = np.mean(S, axis=1)
    D = out["D"]

    idx_stress = (t >= stress_start) & (t <= stress_end)
    idx_post = t > stress_end

    peak_s = float(np.max(mean_s[idx_stress])) if np.any(idx_stress) else float(np.max(mean_s))
    recovery_time = float("nan")
    for i, tt in enumerate(t):
        if tt > stress_end and mean_s[i] <= 0.2:
            recovery_time = float(tt - stress_start)
            break

    per_agent = recovery_times_per_agent(S, t, stress_end)
    ok = per_agent[~np.isnan(per_agent)]
    tail_span = float(np.max(ok) - np.min(ok)) if ok.size else float("nan")
    disp = float(np.max(D[idx_stress])) if np.any(idx_stress) else float(np.max(D))

    return {
        "peak_s": peak_s,
        "recovery_time": recovery_time,
        "tail_span": tail_span,
        "dispersion": disp,
    }


def normalize(v: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


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
        phi0=0.62,
        a1=1.05,
        a2=0.90,
        a3=0.75,
        kappa0=1.15,
        coh_gain=0.60,
        recog_gain=1.20,
    )

    out = run_multi_episode(params, seed=11, adaptive=True)

    raw_rows: list[dict[str, float]] = []
    for start, duration in WINDOWS:
        feat = episode_features(out, start, duration)
        feat["stress_start"] = start
        feat["stress_duration"] = duration
        raw_rows.append(feat)

    peak_s = normalize(np.array([r["peak_s"] for r in raw_rows], dtype=float))
    rec_t = normalize(np.array([r["recovery_time"] for r in raw_rows], dtype=float))
    tail = normalize(np.array([r["tail_span"] for r in raw_rows], dtype=float))
    disp = normalize(np.array([r["dispersion"] for r in raw_rows], dtype=float))

    weights = {"ws": 0.35, "wr": 0.30, "wt": 0.20, "wd": 0.15}
    C = weights["ws"] * peak_s + weights["wr"] * rec_t + weights["wt"] * tail + weights["wd"] * disp

    thresholds = [0.55, 0.60, 0.65]
    rows: list[dict[str, float | str]] = []
    for crit in thresholds:
        q = (C > crit).astype(float)
        kg = 0.0
        for i in range(len(C)):
            kg = (1.0 - 0.15) * kg + 0.15 * (C[i] * q[i])
            rows.append({
                "threshold": crit,
                "episode": float(i + 1),
                "C_E": float(C[i]),
                "gate": float(q[i]),
                "K_g": float(kg),
                "peak_s_norm": float(peak_s[i]),
                "recovery_norm": float(rec_t[i]),
                "tail_norm": float(tail[i]),
                "disp_norm": float(disp[i]),
            })

    csv_path = out_dir / "group_memory_kg_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for crit in thresholds:
        sub = [r for r in rows if abs(r["threshold"] - crit) < 1e-12]
        print(f"threshold {crit:.2f}: " + ", ".join(f"ep{int(r['episode'])}=K_g:{r['K_g']:.3f}/gate:{int(r['gate'])}" for r in sub))

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
