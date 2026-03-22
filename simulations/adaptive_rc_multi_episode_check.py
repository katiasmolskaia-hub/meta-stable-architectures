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
from simulations.adaptive_rc_skeleton import clamp, normalize_weights


def observe(theta: np.ndarray, s: np.ndarray, prev_s: np.ndarray) -> dict[str, np.ndarray | float]:
    mean_vec = np.mean(np.exp(1j * theta))
    dispersion = float(1.0 - abs(mean_vec))
    lag = float(np.mean(s))
    contagion = float(np.mean(prev_s))
    anchors = 1.0 - s
    return {"D": dispersion, "L": lag, "C": contagion, "A": anchors}


def recovery_times_per_agent(S: np.ndarray, t: np.ndarray, stress_end: float, threshold: float = 0.2) -> np.ndarray:
    out = np.full(S.shape[1], np.nan, dtype=float)
    start_idx = int(np.searchsorted(t, stress_end, side="right"))
    for j in range(S.shape[1]):
        for i in range(start_idx, len(t)):
            if S[i, j] <= threshold:
                out[j] = float(t[i] - stress_end)
                break
    return out


def run_multi_episode(params: HeteroDemoParams, seed: int, adaptive: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = params.n_agents
    steps = int(params.t_end / params.dt) + 1
    t = np.linspace(0.0, params.t_end, steps)

    x = rng.normal(0.0, 1.0, size=n)
    y = rng.normal(0.0, 1.0, size=n)
    s = np.clip(rng.normal(0.15, 0.05, size=n), 0.0, 1.0)
    k = np.zeros(n)
    Phi = 0.0
    prev_s = s.copy()

    stress_sensitivity = rng.lognormal(mean=0.0, sigma=params.hetero_sigma, size=n)
    recog_threshold = np.clip(rng.normal(0.70, 0.06, size=n), 0.55, 0.90)
    noise_scale = params.noise_scale * rng.lognormal(mean=0.0, sigma=params.hetero_sigma, size=n)

    stress_windows = [
        (20.0, 6.0),
        (55.0, 8.0),
        (95.0, 10.0),
        (140.0, 12.0),
    ]

    X = np.zeros((steps, n))
    Y = np.zeros((steps, n))
    S = np.zeros((steps, n))
    K = np.zeros((steps, n))
    Phi_hist = np.zeros(steps)
    D_hist = np.zeros(steps)
    phi_gain_hist = np.zeros(steps)
    kappa_hist = np.zeros(steps)

    for idx, tt in enumerate(t):
        X[idx] = x
        Y[idx] = y
        S[idx] = s
        K[idx] = k
        Phi_hist[idx] = Phi

        theta = np.arctan2(y, x)
        obs = observe(theta, s, prev_s)
        D = float(obs["D"])
        L = float(obs["L"])
        C = float(obs["C"])
        A = np.asarray(obs["A"], dtype=float)
        D_hist[idx] = D

        if adaptive:
            phi_gain = clamp(params.phi0 * (1.0 + params.a1 * D + params.a2 * L + params.a3 * C), params.phi_min, params.phi_max)
            kappa = clamp(params.kappa0 * (1.0 + params.b1 * D + params.b2 * L), params.kappa_min, params.kappa_max)
        else:
            phi_gain = params.phi0
            kappa = params.kappa0
        phi_gain_hist[idx] = phi_gain
        kappa_hist[idx] = kappa

        if idx == steps - 1:
            break

        mean_phase = float(np.angle(np.mean(np.exp(1j * theta))))
        Phi = Phi + params.dt * (-kappa * Phi + mean_phase)

        w = normalize_weights(A)
        neighbor_signal = np.dot(w, y)
        anti_contagion = (1.0 - s) * (1.0 - np.mean(s))

        dx = -0.8 * x - 0.25 * y
        dy = 0.9 * x - 0.85 * (y**3)
        coupling_term = params.coupling * anti_contagion * (neighbor_signal - y)
        coordinator_term = phi_gain * (Phi - y) * (1.0 - s)

        x = x + params.dt * dx
        y = y + params.dt * (dy + coupling_term + coordinator_term)
        y = y + np.sqrt(params.dt) * noise_scale * rng.normal(size=n)

        in_stress = any(start <= tt < start + duration for start, duration in stress_windows)
        if in_stress:
            active = next(duration for start, duration in stress_windows if start <= tt < start + duration)
            s = np.clip(s + 0.08 * params.stress_amp * stress_sensitivity * (1.0 + 0.02 * active), 0.0, 1.0)
            y = y + 0.06 * params.stress_y_amp * stress_sensitivity

        match = np.cos(theta - Phi)
        recog = np.maximum(0.0, match - recog_threshold)
        dS = params.base_suppression * (0.4 - D) * s * (1.0 - s)
        dS -= params.recog_gain * recog * s * (1.0 - s)
        dS -= params.coh_gain * (1.0 - D) * s
        dS = dS * stress_sensitivity
        s = np.clip(s + params.dt * dS, 0.0, 1.0)

        k = np.maximum(0.0, k + params.dt * (0.25 * y**2 - 0.3 * k + 0.2 * s))
        prev_s = s.copy()

    return {
        "t": t,
        "X": X,
        "Y": Y,
        "S": S,
        "K": K,
        "Phi": Phi_hist,
        "D": D_hist,
        "phi_gain": phi_gain_hist,
        "kappa": kappa_hist,
        "stress_windows": np.array(stress_windows, dtype=float),
    }


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
    }


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

    windows = [
        (20.0, 6.0),
        (55.0, 8.0),
        (95.0, 10.0),
        (140.0, 12.0),
    ]

    rows: list[dict[str, float]] = []
    for start, duration in windows:
        rows.append(episode_summary(out, start, duration))

    csv_path = out_dir / "adaptive_rc_multi_episode_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for i, row in enumerate(rows, 1):
        print(
            f"episode {i}: group_rt={row['group_recovery_time']:.3f}, "
            f"agent_rt_std={row['agent_recovery_std']:.3f}, "
            f"agent_rt_p95={row['agent_recovery_p95']:.3f}"
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
        plot_path = out_dir / "adaptive_rc_multi_episode_check_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
