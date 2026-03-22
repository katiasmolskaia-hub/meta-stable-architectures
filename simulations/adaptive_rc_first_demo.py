from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

from simulations.adaptive_rc_skeleton import AdaptiveRCParams, clamp, normalize_weights


@dataclass
class DemoParams(AdaptiveRCParams):
    stress_time: float = 25.0
    stress_duration: float = 6.0
    stress_amp: float = 2.5
    stress_y_amp: float = 0.8


def observe(theta: np.ndarray, s: np.ndarray, prev_s: np.ndarray) -> dict[str, np.ndarray | float]:
    mean_vec = np.mean(np.exp(1j * theta))
    dispersion = float(1.0 - abs(mean_vec))
    lag = float(np.mean(s))
    contagion = float(np.mean(prev_s))
    anchors = 1.0 - s
    return {"D": dispersion, "L": lag, "C": contagion, "A": anchors}


def run_demo(params: DemoParams, seed: int, adaptive: bool = True) -> dict[str, np.ndarray]:
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

        if params.stress_time <= tt < params.stress_time + params.stress_duration:
            # Stress: a burst of suppression and a push into the fast channel.
            s = np.clip(s + 0.08 * params.stress_amp, 0.0, 1.0)
            y = y + 0.06 * params.stress_y_amp

        match = np.cos(theta - Phi)
        recog = np.maximum(0.0, match - 0.7)
        dS = params.base_suppression * (0.4 - D) * s * (1.0 - s)
        dS -= params.recog_gain * recog * s * (1.0 - s)
        dS -= params.coh_gain * (1.0 - D) * s
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
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = DemoParams()
    adaptive = run_demo(params, seed=7, adaptive=True)
    fixed = run_demo(replace(params, a1=0.0, a2=0.0, a3=0.0, b1=0.0, b2=0.0), seed=7, adaptive=False)

    def rec_time(arr: np.ndarray, t: np.ndarray, threshold: float = 0.2) -> float:
        mean_s = np.mean(arr, axis=1)
        for i in range(len(t)):
            if t[i] > params.stress_time + params.stress_duration and mean_s[i] <= threshold:
                return float(t[i] - params.stress_time)
        return float("nan")

    print(f"Adaptive recovery time: {rec_time(adaptive['S'], adaptive['t']):.3f}")
    print(f"Fixed recovery time: {rec_time(fixed['S'], fixed['t']):.3f}")
    print(f"Adaptive final mean S: {float(np.mean(adaptive['S'][-1])):.4f}")
    print(f"Fixed final mean S: {float(np.mean(fixed['S'][-1])):.4f}")
    print(f"Adaptive final dispersion: {float(adaptive['D'][-1]):.4f}")
    print(f"Fixed final dispersion: {float(fixed['D'][-1]):.4f}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(adaptive["t"], np.mean(adaptive["S"], axis=1), label="adaptive")
        axes[0, 0].plot(fixed["t"], np.mean(fixed["S"], axis=1), label="fixed", linestyle="--")
        axes[0, 0].set_title("Mean suppression")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(adaptive["t"], adaptive["D"], label="adaptive")
        axes[0, 1].plot(fixed["t"], fixed["D"], label="fixed", linestyle="--")
        axes[0, 1].set_title("Phase dispersion")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(adaptive["t"], np.mean(adaptive["K"], axis=1), label="adaptive")
        axes[1, 0].plot(fixed["t"], np.mean(fixed["K"], axis=1), label="fixed", linestyle="--")
        axes[1, 0].set_title("Mean K")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(adaptive["t"], adaptive["phi_gain"], label="adaptive")
        axes[1, 1].plot(fixed["t"], fixed["phi_gain"], label="fixed", linestyle="--")
        axes[1, 1].set_title("Phi gain")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        out_plot = out_dir / "adaptive_rc_first_demo_plot.png"
        fig.savefig(out_plot, dpi=160)
        plt.close(fig)
        print(f"Wrote {out_plot}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
