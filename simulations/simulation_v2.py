"""
Simulation v2 for Meta-Stable Architectures (Volumes I-III).

Includes:
- Deterministic master model (Euler integration)
- Stochastic memory block (Euler-Maruyama integration)

Usage:
    python simulations/simulation_v2.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class MasterParams:
    # Volume I / II core
    kappa: float = 1.2
    lam: float = 0.25
    c_s: float = 0.35
    u_max: float = 0.8
    c0: float = 0.45
    mu_k: float = 0.25
    zeta: float = 0.55
    tau: float = 6.0
    eta: float = 0.45
    lam_c: float = 0.8
    g_min: float = 0.15
    delta_k: float = 0.3
    nu: float = 0.2
    xi: float = 0.9

    # Volume III memory block
    k: float = 0.8
    alpha: float = 0.9
    gamma: float = 0.85
    sigma_h: float = 0.35
    delta_h: float = 0.45
    eta_s: float = 0.55
    beta: float = 0.25
    rho1: float = 0.4
    rho2: float = 0.3
    rho3: float = 0.35
    eps_s: float = 0.6
    h_crit: float = 0.7
    sigma_noise: float = 0.22


def grad_v(a: float) -> float:
    """Gradient of quartic potential V(a)=1/4 a^4 - 1/2 a^2."""
    return a**3 - a


def sat(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def theta_sigmoid(x: float, sharpness: float = 5.0) -> float:
    """Smooth activation in [0,1]."""
    return 1.0 / (1.0 + math.exp(-sharpness * x))


def simulate_master_deterministic(
    t_end: float = 120.0,
    dt: float = 0.01,
    p: MasterParams | None = None,
) -> dict[str, np.ndarray]:
    """
    Simulates unified deterministic model:
        da/dt = -g(C,K,t)*gradV(a) - lam*S(a,phi)
        dphi/dt = -kappa*phi + u(a)
        dK/dt = nu*C - delta_k*K
        dI/dt = xi*(C-Ccrit)*I*(1-I)
    """
    if p is None:
        p = MasterParams()

    n = int(t_end / dt) + 1
    t = np.linspace(0.0, t_end, n)

    a = np.zeros(n)
    phi = np.zeros(n)
    k_struct = np.zeros(n)
    i_iso = np.zeros(n)
    c = np.zeros(n)
    c_crit = np.zeros(n)
    g = np.zeros(n)
    v = np.zeros(n)

    # Initial state
    a[0] = 1.15
    phi[0] = 0.25
    k_struct[0] = 0.0
    i_iso[0] = 0.05

    window = max(1, int(p.tau / dt))

    for idx in range(n - 1):
        gv = grad_v(a[idx])
        c[idx] = gv * gv
        v[idx] = 0.25 * a[idx] ** 4 - 0.5 * a[idx] ** 2

        left = max(0, idx - window)
        c_avg = float(np.mean(c[left : idx + 1]))

        c_crit[idx] = p.c0 * math.exp(-p.mu_k * k_struct[idx]) + p.zeta * c_avg
        th = theta_sigmoid(c[idx] - c_crit[idx])

        g_raw = 1.0 - p.eta * p.lam_c * th
        g[idx] = max(p.g_min, g_raw)

        s_term = p.c_s * math.tanh(a[idx] + 0.7 * phi[idx])
        u_term = sat(c[idx], p.u_max)

        da = -g[idx] * gv - p.lam * s_term
        dphi = -p.kappa * phi[idx] + u_term
        dK = p.nu * c[idx] - p.delta_k * k_struct[idx]
        dI = p.xi * (c[idx] - c_crit[idx]) * i_iso[idx] * (1.0 - i_iso[idx])

        a[idx + 1] = a[idx] + dt * da
        phi[idx + 1] = max(0.0, phi[idx] + dt * dphi)
        k_struct[idx + 1] = max(0.0, k_struct[idx] + dt * dK)
        i_iso[idx + 1] = float(np.clip(i_iso[idx] + dt * dI, 0.0, 1.0))

    gv_last = grad_v(a[-1])
    c[-1] = gv_last * gv_last
    v[-1] = 0.25 * a[-1] ** 4 - 0.5 * a[-1] ** 2
    c_crit[-1] = p.c0 * math.exp(-p.mu_k * k_struct[-1]) + p.zeta * float(np.mean(c[max(0, n - window) : n]))
    g[-1] = max(p.g_min, 1.0 - p.eta * p.lam_c * theta_sigmoid(c[-1] - c_crit[-1]))

    return {
        "t": t,
        "a": a,
        "phi": phi,
        "K": k_struct,
        "I": i_iso,
        "C": c,
        "Ccrit": c_crit,
        "g": g,
        "V": v,
    }


def simulate_memory_sde(
    t_end: float = 120.0,
    dt: float = 0.01,
    p: MasterParams | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Simulates Volume III memory block with noise in y:
      dx = (-k*x - lam*y)dt
      dy = (alpha*x + mu*y - gamma*y^3)dt + sigma dW
      dH = (sigma_h*y^2 - (delta_h + eta_s*S)*H)dt
      dK = (beta*H - delta_k*K + xi*S*H)dt
      dmu = (rho1*H - rho2*K - rho3*mu)dt
      dS = eps_s*(H-h_crit)*S*(1-S)dt
    """
    if p is None:
        p = MasterParams()

    rng = np.random.default_rng(seed)
    n = int(t_end / dt) + 1
    t = np.linspace(0.0, t_end, n)

    x = np.zeros(n)
    y = np.zeros(n)
    h = np.zeros(n)
    k_struct = np.zeros(n)
    mu = np.zeros(n)
    s_buf = np.zeros(n)

    x[0], y[0] = 0.2, 0.1
    h[0], k_struct[0], mu[0], s_buf[0] = 0.05, 0.0, 0.1, 0.05

    sqrt_dt = math.sqrt(dt)

    for idx in range(n - 1):
        dx = -p.k * x[idx] - p.lam * y[idx]
        drift_y = p.alpha * x[idx] + mu[idx] * y[idx] - p.gamma * y[idx] ** 3
        noise = p.sigma_noise * sqrt_dt * rng.normal()

        dH = p.sigma_h * y[idx] ** 2 - (p.delta_h + p.eta_s * s_buf[idx]) * h[idx]
        dK = p.beta * h[idx] - p.delta_k * k_struct[idx] + p.xi * s_buf[idx] * h[idx]
        dmu = p.rho1 * h[idx] - p.rho2 * k_struct[idx] - p.rho3 * mu[idx]
        dS = p.eps_s * (h[idx] - p.h_crit) * s_buf[idx] * (1.0 - s_buf[idx])

        x[idx + 1] = x[idx] + dt * dx
        y[idx + 1] = y[idx] + dt * drift_y + noise
        h[idx + 1] = max(0.0, h[idx] + dt * dH)
        k_struct[idx + 1] = max(0.0, k_struct[idx] + dt * dK)
        mu[idx + 1] = mu[idx] + dt * dmu
        s_buf[idx + 1] = float(np.clip(s_buf[idx] + dt * dS, 0.0, 1.0))

    return {"t": t, "x": x, "y": y, "H": h, "K": k_struct, "mu": mu, "S": s_buf}


def _plot_results(det: dict[str, np.ndarray], sde: dict[str, np.ndarray]) -> None:
    if plt is None:
        print("matplotlib is not available. Numerical run completed without plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(det["t"], det["a"], label="a(t)")
    axes[0, 0].plot(det["t"], det["phi"], label="phi(t)")
    axes[0, 0].set_title("Deterministic core")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(det["t"], det["C"], label="C(t)")
    axes[0, 1].plot(det["t"], det["Ccrit"], label="Ccrit(t)")
    axes[0, 1].set_title("Crisis intensity vs threshold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(sde["t"], sde["y"], label="y(t)")
    axes[1, 0].plot(sde["t"], sde["mu"], label="mu(t)")
    axes[1, 0].set_title("Stochastic fast block")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(sde["t"], sde["H"], label="H(t)")
    axes[1, 1].plot(sde["t"], sde["K"], label="K(t)")
    axes[1, 1].plot(sde["t"], sde["S"], label="S(t)")
    axes[1, 1].set_title("Memory and suppression buffer")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    params = MasterParams()
    det = simulate_master_deterministic(p=params)
    sde = simulate_memory_sde(p=params)

    print("Simulation v2 completed.")
    print(
        f"Final deterministic state: a={det['a'][-1]:.4f}, phi={det['phi'][-1]:.4f}, "
        f"K={det['K'][-1]:.4f}, I={det['I'][-1]:.4f}"
    )
    print(
        f"Final stochastic state: x={sde['x'][-1]:.4f}, y={sde['y'][-1]:.4f}, "
        f"H={sde['H'][-1]:.4f}, K={sde['K'][-1]:.4f}, mu={sde['mu'][-1]:.4f}, S={sde['S'][-1]:.4f}"
    )

    _plot_results(det, sde)


if __name__ == "__main__":
    main()
