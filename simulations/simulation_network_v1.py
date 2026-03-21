"""
Network simulation for Meta-Stable Architectures (multi-agent, no metronome).

Usage:
    python simulations/simulation_network_v1.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import os

import numpy as np

try:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class MasterParams:
    # Volume I / II core (used in thresholds)
    c0: float = 0.45
    mu_k: float = 0.25
    zeta: float = 0.55

    # Volume III memory block
    k: float = 0.8
    lam: float = 0.25
    alpha: float = 0.9
    gamma: float = 0.85
    sigma_h: float = 0.35
    delta_h: float = 0.45
    eta_s: float = 0.55
    beta: float = 0.25
    rho1: float = 0.4
    rho2: float = 0.3
    rho3: float = 0.35
    delta_k: float = 0.3
    eps_s: float = 0.6
    h_crit: float = 0.7
    sigma_noise: float = 0.22
    xi: float = 0.9
    iso_cool: float = 1.2
    iso_noise_scale: float = 0.2
    eps_s2: float = 0.25
    c_crit: float = 0.6


@dataclass
class NetworkParams:
    n_agents: int = 32
    t_end: float = 120.0
    dt: float = 0.01
    coupling: float = 0.2
    ring_k: int = 2  # neighbors on each side
    seed: int = 42
    topology: str = "ring"  # ring | erdos_renyi | small_world | scale_free
    er_p: float = 0.1
    sw_rewire: float = 0.1
    ba_m: int = 2
    delay_mode: str = "fixed"  # fixed | grouped
    delay_steps: int = 0
    delay_group_fracs: tuple[float, ...] = (0.5, 0.5)
    delay_group_steps: tuple[int, ...] = (2, 6)

    stress_time: float = 30.0
    stress_amp: float = 3.0
    stress_frac: float = 0.5
    stress_duration: float = 4.0
    stress_y_amp: float = 1.0
    y_cap: float = 3.5

    iso_threshold: float = 0.8
    recovery_threshold: float = 0.2
    metro_amp: float = 0.0
    metro_freq: float = 1.1
    metro_phase: float = 0.0
    # QRC / reflexive layer
    qrc_enabled: bool = False
    phi_kappa: float = 1.0
    phi_gain: float = 0.15
    qrc_eta: float = 0.6
    qrc_g_min: float = 0.3
    qrc_g_max: float = 1.2
    phi_gain_boost: float = 4.0
    phi_listen_isolated: bool = False
    phi_iso_threshold: float = 0.5
    recog_threshold: float = 0.7
    recog_gain: float = 1.2
    wake_disp_threshold: float = 0.3
    wake_time_required: float = 4.0
    wake_relax_gain: float = 0.8
    coh_relax_gain: float = 0.6
    ccrit_gain: float = 0.6
    ccrit_floor: float = 0.2
    ccrit_cap: float = 1.2


def _build_ring_adjacency(n: int, k: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        for offset in range(1, k + 1):
            a[i, (i + offset) % n] = 1.0
            a[i, (i - offset) % n] = 1.0
    return a


def _build_erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    a = rng.random((n, n)) < p
    np.fill_diagonal(a, 0)
    a = np.triu(a, 1)
    a = a + a.T
    return a.astype(float)


def _build_small_world(n: int, k: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    a = _build_ring_adjacency(n, k)
    for i in range(n):
        for offset in range(1, k + 1):
            j = (i + offset) % n
            if rng.random() < beta:
                a[i, j] = 0.0
                a[j, i] = 0.0
                candidates = [x for x in range(n) if x != i and a[i, x] == 0.0]
                if candidates:
                    new_j = rng.choice(candidates)
                    a[i, new_j] = 1.0
                    a[new_j, i] = 1.0
                else:
                    a[i, j] = 1.0
                    a[j, i] = 1.0
    return a


def _build_scale_free(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    m = max(1, min(m, n - 1))
    a = np.zeros((n, n), dtype=float)
    core = m + 1
    for i in range(core):
        for j in range(i + 1, core):
            a[i, j] = 1.0
            a[j, i] = 1.0
    degrees = a.sum(axis=1)
    for new in range(core, n):
        probs = degrees[:new]
        if probs.sum() == 0:
            targets = rng.choice(new, size=m, replace=False)
        else:
            probs = probs / probs.sum()
            targets = rng.choice(new, size=m, replace=False, p=probs)
        for t in targets:
            a[new, t] = 1.0
            a[t, new] = 1.0
        degrees = a.sum(axis=1)
    return a


def _circular_variance(theta: np.ndarray) -> float:
    mean_vec = np.mean(np.exp(1j * theta))
    return float(1.0 - np.abs(mean_vec))


def simulate_network(
    p: MasterParams | None = None,
    net: NetworkParams | None = None,
) -> dict[str, np.ndarray]:
    if p is None:
        p = MasterParams()
    if net is None:
        net = NetworkParams()

    rng = np.random.default_rng(net.seed)
    n_steps = int(net.t_end / net.dt) + 1
    t = np.linspace(0.0, net.t_end, n_steps)

    n = net.n_agents
    if net.topology == "ring":
        a = _build_ring_adjacency(n, net.ring_k)
    elif net.topology == "erdos_renyi":
        a = _build_erdos_renyi(n, net.er_p, rng)
    elif net.topology == "small_world":
        a = _build_small_world(n, net.ring_k, net.sw_rewire, rng)
    elif net.topology == "scale_free":
        a = _build_scale_free(n, net.ba_m, rng)
    else:
        raise ValueError(f"Unknown topology: {net.topology}")
    deg = a.sum(axis=1)
    deg[deg == 0.0] = 1.0

    # State
    x = np.zeros((n_steps, n))
    y = np.zeros((n_steps, n))
    h = np.zeros((n_steps, n))
    k_struct = np.zeros((n_steps, n))
    mu = np.zeros((n_steps, n))
    s_buf = np.zeros((n_steps, n))
    phi = np.zeros(n_steps)

    x[0] = 0.2 + 0.02 * rng.normal(size=n)
    y[0] = 0.1 + 0.02 * rng.normal(size=n)
    h[0] = 0.05
    k_struct[0] = 0.0
    mu[0] = 0.1
    s_buf[0] = 0.05
    phi[0] = 0.0

    stress_idx = int(net.stress_time / net.dt)
    stress_steps = max(1, int(net.stress_duration / net.dt))
    stress_agents = rng.choice(n, size=max(1, int(n * net.stress_frac)), replace=False)

    sqrt_dt = math.sqrt(net.dt)

    phase_disp = np.zeros(n_steps)
    frac_iso = np.zeros(n_steps)
    mean_h = np.zeros(n_steps)
    calm_time = np.zeros(n_steps)

    # Delay profile (for grouped delays)
    delay_per_agent = np.zeros(n, dtype=int)
    if net.delay_mode == "grouped" and sum(net.delay_group_fracs) > 0:
        fracs = np.array(net.delay_group_fracs, dtype=float)
        fracs = fracs / fracs.sum()
        steps = np.array(net.delay_group_steps, dtype=int)
        groups = rng.choice(len(fracs), size=n, p=fracs)
        delay_per_agent = steps[groups]

    for idx in range(n_steps - 1):
        # Reflexive phase (global mediator)
        if net.qrc_enabled:
            theta = np.arctan2(y[idx], x[idx])
            if net.phi_listen_isolated:
                mask = s_buf[idx] > net.phi_iso_threshold
                if np.any(mask):
                    u_global = float(np.arctan2(np.mean(np.sin(theta[mask])), np.mean(np.cos(theta[mask]))))
                else:
                    u_global = float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))
            else:
                u_global = float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))
            phi[idx + 1] = phi[idx] + net.dt * (-net.phi_kappa * phi[idx] + u_global)
            c_local = y[idx] ** 2
            c_mean = float(np.mean(c_local))
            g = max(net.qrc_g_min, min(net.qrc_g_max, 1.0 - net.qrc_eta * c_mean))
            mean_s_now = float(np.mean(s_buf[idx]))
            phi_gain = net.phi_gain * (1.0 + net.phi_gain_boost * (mean_s_now ** 2))
        else:
            phi[idx + 1] = phi[idx]
            g = 1.0
            phi_gain = 0.0

        # Coupling term (isolation-gated) with optional delays
        if net.delay_mode == "fixed" and net.delay_steps > 0:
            src = y[max(0, idx - net.delay_steps)]
        elif net.delay_mode == "grouped" and np.any(delay_per_agent > 0):
            src = y[idx].copy()
            unique_delays = np.unique(delay_per_agent)
            for d in unique_delays:
                if d == 0:
                    continue
                mask = delay_per_agent == d
                src[mask] = y[max(0, idx - d)][mask]
        else:
            src = y[idx]
        y_neighbor = a @ (src * (1.0 - s_buf[idx]))
        coupling_term = g * net.coupling * (1.0 - s_buf[idx]) * (y_neighbor - deg * y[idx]) / deg
        if net.qrc_enabled and phi_gain != 0.0:
            coupling_term += phi_gain * (phi[idx] - y[idx]) * (1.0 - s_buf[idx])

        dx = -p.k * x[idx] - p.lam * y[idx]
        alpha = p.alpha
        if isinstance(alpha, np.ndarray):
            alpha = alpha
        else:
            alpha = np.full(n, float(alpha))
        drift_y = alpha * x[idx] + mu[idx] * y[idx] - p.gamma * y[idx] ** 3
        metro = net.metro_amp * math.sin(net.metro_freq * t[idx] + net.metro_phase)
        sigma_noise = p.sigma_noise
        if isinstance(sigma_noise, np.ndarray):
            sigma_noise = sigma_noise
        else:
            sigma_noise = np.full(n, float(sigma_noise))
        noise_scale = sigma_noise * (p.iso_noise_scale + (1.0 - p.iso_noise_scale) * (1.0 - s_buf[idx]))
        noise = noise_scale * sqrt_dt * rng.normal(size=n)

        dH = p.sigma_h * y[idx] ** 2 - (p.delta_h + p.eta_s * s_buf[idx] + p.iso_cool * s_buf[idx]) * h[idx]
        dK = p.beta * h[idx] - p.delta_k * k_struct[idx] + p.xi * s_buf[idx] * h[idx]
        dmu = p.rho1 * h[idx] - p.rho2 * k_struct[idx] - p.rho3 * mu[idx]
        c_local = y[idx] ** 2
        ccrit_eff = p.c_crit
        if net.qrc_enabled:
            ccrit_eff = p.c_crit + net.ccrit_gain * (1.0 - phase_disp[idx])
            ccrit_eff = float(np.clip(ccrit_eff, net.ccrit_floor, net.ccrit_cap))
        h_crit = p.h_crit
        if isinstance(h_crit, np.ndarray):
            h_crit = h_crit
        else:
            h_crit = np.full(n, float(h_crit))
        dS = p.eps_s * (h[idx] - h_crit) * s_buf[idx] * (1.0 - s_buf[idx])
        dS += p.eps_s2 * (c_local - ccrit_eff) * s_buf[idx] * (1.0 - s_buf[idx])
        if net.qrc_enabled:
            theta = np.arctan2(y[idx], x[idx])
            match = np.cos(theta - phi[idx])
            recog = np.maximum(0.0, match - net.recog_threshold)
            dS -= net.recog_gain * recog * s_buf[idx] * (1.0 - s_buf[idx])
            if calm_time[idx] >= net.wake_time_required:
                dS -= net.wake_relax_gain * s_buf[idx] * (1.0 - s_buf[idx])
            dS -= net.coh_relax_gain * (1.0 - phase_disp[idx]) * s_buf[idx]

        x[idx + 1] = x[idx] + net.dt * dx
        y[idx + 1] = y[idx] + net.dt * (drift_y + coupling_term + metro) + noise
        if net.y_cap is not None:
            y[idx + 1] = np.clip(y[idx + 1], -net.y_cap, net.y_cap)
        h[idx + 1] = np.maximum(0.0, h[idx] + net.dt * dH)
        k_struct[idx + 1] = np.maximum(0.0, k_struct[idx] + net.dt * dK)
        mu[idx + 1] = mu[idx] + net.dt * dmu
        s_buf[idx + 1] = np.clip(s_buf[idx] + net.dt * dS, 0.0, 1.0)

        if stress_idx <= idx < stress_idx + stress_steps:
            h[idx + 1, stress_agents] += net.stress_amp
            y[idx + 1, stress_agents] += net.stress_y_amp

        theta = np.arctan2(y[idx], x[idx])
        phase_disp[idx] = _circular_variance(theta)
        if phase_disp[idx] < net.wake_disp_threshold:
            calm_time[idx] = (calm_time[idx - 1] + net.dt) if idx > 0 else net.dt
        else:
            calm_time[idx] = 0.0
        frac_iso[idx] = float(np.mean(s_buf[idx] >= net.iso_threshold))
        mean_h[idx] = float(np.mean(h[idx]))

    theta = np.arctan2(y[-1], x[-1])
    phase_disp[-1] = _circular_variance(theta)
    frac_iso[-1] = float(np.mean(s_buf[-1] >= net.iso_threshold))
    mean_h[-1] = float(np.mean(h[-1]))

    # Recovery time: first time after post-stress peak where mean S <= recovery_threshold
    start_idx = min(stress_idx + stress_steps, n_steps - 1)
    mean_s = np.mean(s_buf, axis=1)
    recovery_time = math.nan
    peak_idx = int(np.argmax(mean_s[start_idx:])) + start_idx
    for idx in range(peak_idx, n_steps):
        if mean_s[idx] <= net.recovery_threshold:
            recovery_time = t[idx] - t[stress_idx]
            break

    return {
        "t": t,
        "x": x,
        "y": y,
        "H": h,
        "K": k_struct,
        "mu": mu,
        "S": s_buf,
        "phase_dispersion": phase_disp,
        "fraction_isolated": frac_iso,
        "mean_h": mean_h,
        "mean_s": mean_s,
        "recovery_time": recovery_time,
        "stress_agents": stress_agents,
    }


def _plot_results(out: dict[str, np.ndarray], save_path: str | None = None, show: bool = False) -> None:
    if plt is None:
        print("matplotlib is not available. Numerical run completed without plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(out["t"], out["phase_dispersion"], label="phase dispersion")
    axes[0, 0].set_title("Phase coherence")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(out["t"], out["fraction_isolated"], label="fraction isolated")
    axes[0, 1].set_title("Isolation fraction")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(out["t"], out["mean_h"], label="mean H")
    axes[1, 0].plot(out["t"], out["mean_s"], label="mean S")
    axes[1, 0].set_title("Mean stress and suppression")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(out["t"], out["y"][:, 0], label="y[0]")
    axes[1, 1].plot(out["t"], out["y"][:, 1], label="y[1]")
    axes[1, 1].set_title("Sample fast dynamics")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    out = simulate_network()
    print("Network simulation completed.")
    print(f"Recovery time (mean S <= threshold): {out['recovery_time']}")
    _plot_results(out, save_path="E:\\MyProject\\meta-stable-architectures\\outputs\\network_v1_summary.png")


if __name__ == "__main__":
    main()
