from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np


@dataclass
class AdaptiveRCParams:
    n_agents: int = 100
    t_end: float = 80.0
    dt: float = 0.02
    coupling: float = 0.15
    seed: int = 7

    # Adaptive coordinator
    phi0: float = 0.5
    kappa0: float = 1.0
    phi_min: float = 0.1
    phi_max: float = 1.2
    kappa_min: float = 0.2
    kappa_max: float = 2.0
    a1: float = 0.8
    a2: float = 0.6
    a3: float = 0.5
    b1: float = 0.4
    b2: float = 0.3

    # Suppression dynamics
    recog_gain: float = 1.2
    coh_gain: float = 0.6
    base_suppression: float = 0.15


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_weights(w: np.ndarray) -> np.ndarray:
    s = float(np.sum(w))
    if s <= 1e-12:
        return np.full_like(w, 1.0 / len(w))
    return w / s


def estimate_lag(s: np.ndarray, history: np.ndarray) -> float:
    # Minimal proxy: if suppression is high on average, lag is high.
    # Later this can be replaced by explicit delay tracking.
    return float(np.mean(s))


def estimate_contagion(s: np.ndarray, neighbors: np.ndarray) -> float:
    # Minimal proxy: instability spreads when nearby suppression is high.
    return float(np.mean(neighbors))


def estimate_anchors(s: np.ndarray) -> np.ndarray:
    # Stable agents are those with low suppression.
    return 1.0 - s


def observe_group_state(theta: np.ndarray, s: np.ndarray, history: np.ndarray) -> dict[str, np.ndarray | float]:
    mean_vec = np.mean(np.exp(1j * theta))
    dispersion = float(1.0 - abs(mean_vec))
    lag = estimate_lag(s, history)
    contagion = estimate_contagion(s, history[-1] if len(history) else s)
    anchors = estimate_anchors(s)
    return {
        "D": dispersion,
        "L": lag,
        "C": contagion,
        "A": anchors,
    }


def update_coordinator(params: AdaptiveRCParams, D: float, L: float, C: float, Phi: float) -> tuple[float, float, float]:
    phi_gain = clamp(params.phi0 * (1.0 + params.a1 * D + params.a2 * L + params.a3 * C), params.phi_min, params.phi_max)
    kappa = clamp(params.kappa0 * (1.0 + params.b1 * D + params.b2 * L), params.kappa_min, params.kappa_max)
    return phi_gain, kappa, Phi


def step_state(
    params: AdaptiveRCParams,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    Phi: float,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    theta = np.arctan2(y, x)
    obs = observe_group_state(theta, s, history)
    D = float(obs["D"])
    L = float(obs["L"])
    C = float(obs["C"])
    A = np.asarray(obs["A"], dtype=float)

    phi_gain, kappa, _ = update_coordinator(params, D, L, C, Phi)

    mean_phase = float(np.angle(np.mean(np.exp(1j * theta))))
    Phi = Phi + params.dt * (-kappa * Phi + mean_phase)

    # Stable anchors carry more weight.
    w = normalize_weights(A)
    neighbor_signal = np.dot(w, y)
    anti_contagion = (1.0 - s) * (1.0 - np.mean(s))

    dx = -0.8 * x - 0.25 * y
    dy = 0.9 * x - 0.85 * (y**3)
    coupling_term = params.coupling * anti_contagion * (neighbor_signal - y)
    coordinator_term = phi_gain * (Phi - y) * (1.0 - s)

    x = x + params.dt * dx
    y = y + params.dt * (dy + coupling_term + coordinator_term)

    match = np.cos(theta - Phi)
    recog = np.maximum(0.0, match - 0.7)
    dS = params.base_suppression * (0.4 - D) * s * (1.0 - s)
    dS -= params.recog_gain * recog * s * (1.0 - s)
    dS -= params.coh_gain * (1.0 - D) * s
    s = np.clip(s + params.dt * dS, 0.0, 1.0)

    # Minimal bounded experience update.
    k = np.maximum(0.0, k + params.dt * (0.25 * y**2 - 0.3 * k + 0.2 * s))

    metrics = {
        "D": D,
        "L": L,
        "C": C,
        "phi_gain": phi_gain,
        "kappa": kappa,
        "mean_s": float(np.mean(s)),
        "mean_k": float(np.mean(k)),
    }
    return x, y, s, k, Phi, metrics


def simulate_adaptive_rc(params: AdaptiveRCParams | None = None, seed: int | None = None) -> dict[str, np.ndarray]:
    if params is None:
        params = AdaptiveRCParams()
    if seed is None:
        seed = params.seed

    rng = np.random.default_rng(seed)
    n = params.n_agents
    steps = int(params.t_end / params.dt) + 1
    t = np.linspace(0.0, params.t_end, steps)

    x = rng.normal(0.0, 1.0, size=n)
    y = rng.normal(0.0, 1.0, size=n)
    s = np.clip(rng.normal(0.15, 0.05, size=n), 0.0, 1.0)
    k = np.zeros(n)
    Phi = 0.0

    X = np.zeros((steps, n))
    Y = np.zeros((steps, n))
    S = np.zeros((steps, n))
    K = np.zeros((steps, n))
    Phi_hist = np.zeros(steps)
    D_hist = np.zeros(steps)
    L_hist = np.zeros(steps)
    C_hist = np.zeros(steps)
    phi_gain_hist = np.zeros(steps)
    kappa_hist = np.zeros(steps)

    history = np.zeros((1, n))

    for idx in range(steps):
        X[idx] = x
        Y[idx] = y
        S[idx] = s
        K[idx] = k
        Phi_hist[idx] = Phi

        theta = np.arctan2(y, x)
        obs = observe_group_state(theta, s, history)
        D_hist[idx] = float(obs["D"])
        L_hist[idx] = float(obs["L"])
        C_hist[idx] = float(obs["C"])

        phi_gain, kappa, _ = update_coordinator(params, D_hist[idx], L_hist[idx], C_hist[idx], Phi)
        phi_gain_hist[idx] = phi_gain
        kappa_hist[idx] = kappa

        if idx == steps - 1:
            break

        x, y, s, k, Phi, _ = step_state(params, x, y, s, k, Phi, history)
        history = np.vstack([history[-1], s])

    return {
        "t": t,
        "x": X,
        "y": Y,
        "S": S,
        "K": K,
        "Phi": Phi_hist,
        "D": D_hist,
        "L": L_hist,
        "C": C_hist,
        "phi_gain": phi_gain_hist,
        "kappa": kappa_hist,
    }


def main() -> None:
    out = simulate_adaptive_rc()
    print(f"Final mean S: {float(np.mean(out['S'][-1])):.4f}")
    print(f"Final mean K: {float(np.mean(out['K'][-1])):.4f}")
    print(f"Final dispersion: {float(out['D'][-1]):.4f}")


if __name__ == "__main__":
    main()
