"""Early-warning extension: compare reactive I(t) vs predictive I(t)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation_noise import NoiseParams, grad_v, sat, theta_sigmoid


def simulate_master_early_warning(
    t_end: float,
    dt: float,
    p: NoiseParams,
    seed: int,
    alpha_pred: float,
) -> dict[str, np.ndarray]:
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

    a[0] = 1.15
    phi[0] = 0.25
    k_struct[0] = 0.0
    i_iso[0] = 0.05

    window = max(1, int(p.tau / dt))
    sqrt_dt = math.sqrt(dt)

    for idx in range(n - 1):
        gv = grad_v(a[idx])
        c[idx] = gv * gv

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

        # Early-warning predictor
        c_pred = (1.0 + alpha_pred) * c_avg
        dI = p.xi * (c_pred - c_crit[idx]) * i_iso[idx] * (1.0 - i_iso[idx])

        noise = p.sigma_a * sqrt_dt * rng.normal()
        a[idx + 1] = a[idx] + dt * da + noise
        phi[idx + 1] = max(0.0, phi[idx] + dt * dphi)
        k_struct[idx + 1] = max(0.0, k_struct[idx] + dt * dK)
        i_iso[idx + 1] = float(np.clip(i_iso[idx] + dt * dI, 0.0, 1.0))

    c[-1] = grad_v(a[-1]) ** 2
    c_crit[-1] = p.c0 * math.exp(-p.mu_k * k_struct[-1]) + p.zeta * float(np.mean(c[max(0, n - window) : n]))

    return {"t": t, "C": c, "Ccrit": c_crit, "I": i_iso}


def simulate_master_reactive(t_end: float, dt: float, p: NoiseParams, seed: int) -> dict[str, np.ndarray]:
    from simulation_noise import simulate_master_langevin

    det = simulate_master_langevin(t_end=t_end, dt=dt, p=p, seed=seed)
    return {"t": det["t"], "C": det["C"], "Ccrit": det["Ccrit"], "I": det["I"]}


def main() -> None:
    out_dir = Path('outputs/experiments/20260313_early_warning')
    out_dir.mkdir(parents=True, exist_ok=True)

    params = replace(NoiseParams(), eta=0.75, sigma_a=0.58)
    t_end = 120.0
    dt = 0.01
    seed = 42
    alpha_pred = 0.25

    reactive = simulate_master_reactive(t_end, dt, params, seed)
    early = simulate_master_early_warning(t_end, dt, params, seed, alpha_pred)

    # Save CSVs
    for name, data in [("reactive", reactive), ("early", early)]:
        keys = ["t", "C", "Ccrit", "I"]
        matrix = np.column_stack([data[k] for k in keys])
        np.savetxt(out_dir / f"{name}_traces.csv", matrix, delimiter=',', header=','.join(keys), comments='')

    # Plot I(t) comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(reactive["t"], reactive["I"], label="I(t) reactive")
    ax.plot(early["t"], early["I"], label=f"I(t) early (alpha={alpha_pred})")
    ax.set_xlabel('t')
    ax.set_ylabel('I')
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "early_warning_I_compare.png", dpi=160)
    plt.close(fig)

    # Plot C vs Ccrit for context
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(reactive["t"], reactive["C"], label="C(t)")
    ax.plot(reactive["t"], reactive["Ccrit"], label="Ccrit(t)", alpha=0.7)
    ax.set_xlabel('t')
    ax.set_ylabel('C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "early_warning_C_context.png", dpi=160)
    plt.close(fig)

    with (out_dir / "RUN_NOTES.txt").open('w', encoding='utf-8') as f:
        f.write("Early-warning extension test\n")
        f.write(f"eta={params.eta}\n")
        f.write(f"sigma_a={params.sigma_a}\n")
        f.write(f"alpha_pred={alpha_pred}\n")
        f.write(f"t_end={t_end}\n")
        f.write(f"dt={dt}\n")

    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
