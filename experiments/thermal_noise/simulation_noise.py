"""
Langevin variant of the deterministic core for Meta-Stable Architectures.
Adds thermal noise directly to a(t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class NoiseParams:
    # Core (deterministic) parameters
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

    # Langevin noise strength on a
    sigma_a: float = 0.0


def grad_v(a: float) -> float:
    """Gradient of quartic potential V(a)=1/4 a^4 - 1/2 a^2."""
    return a**3 - a


def sat(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def theta_sigmoid(x: float, sharpness: float = 5.0) -> float:
    """Smooth activation in [0,1]."""
    return 1.0 / (1.0 + math.exp(-sharpness * x))


def simulate_master_langevin(
    t_end: float = 120.0,
    dt: float = 0.01,
    p: NoiseParams | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Langevin version of the deterministic core:
        da = (-g*gradV(a) - lam*S(a,phi)) dt + sigma_a * sqrt(dt) * N(0,1)
        dphi/dt = -kappa*phi + u(a)
        dK/dt = nu*C - delta_k*K
        dI/dt = xi*(C-Ccrit)*I*(1-I)
    """
    if p is None:
        p = NoiseParams()

    rng = np.random.default_rng(seed)
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
    sqrt_dt = math.sqrt(dt)

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

        noise = p.sigma_a * sqrt_dt * rng.normal()
        a[idx + 1] = a[idx] + dt * da + noise
        phi[idx + 1] = max(0.0, phi[idx] + dt * dphi)
        k_struct[idx + 1] = max(0.0, k_struct[idx] + dt * dK)
        i_iso[idx + 1] = float(np.clip(i_iso[idx] + dt * dI, 0.0, 1.0))

    gv_last = grad_v(a[-1])
    c[-1] = gv_last * gv_last
    v[-1] = 0.25 * a[-1] ** 4 - 0.5 * a[-1] ** 2
    c_crit[-1] = p.c0 * math.exp(-p.mu_k * k_struct[-1]) + p.zeta * float(
        np.mean(c[max(0, n - window) : n])
    )
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


def _plot_results(det: dict[str, np.ndarray]) -> None:
    if plt is None:
        print("matplotlib is not available. Numerical run completed without plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(det["t"], det["a"], label="a(t)")
    axes[0, 0].plot(det["t"], det["phi"], label="phi(t)")
    axes[0, 0].set_title("Deterministic core + noise on a")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(det["t"], det["C"], label="C(t)")
    axes[0, 1].plot(det["t"], det["Ccrit"], label="Ccrit(t)")
    axes[0, 1].set_title("Crisis intensity vs threshold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(det["t"], det["I"], label="I(t)")
    axes[1, 0].set_title("Isolation variable I(t)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(det["t"], det["V"], label="V(a)")
    axes[1, 1].set_title("Potential energy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    params = NoiseParams()
    det = simulate_master_langevin(p=params)

    print("Langevin simulation completed.")
    print(
        f"Final state: a={det['a'][-1]:.4f}, phi={det['phi'][-1]:.4f}, "
        f"K={det['K'][-1]:.4f}, I={det['I'][-1]:.4f}"
    )

    _plot_results(det)


if __name__ == "__main__":
    main()
