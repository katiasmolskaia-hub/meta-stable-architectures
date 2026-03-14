"""Controlled thaw with explicit hold until switch time."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation_noise import NoiseParams, grad_v, sat, theta_sigmoid


def simulate_thaw_controlled(t_end: float, dt: float, p: NoiseParams, seed: int, t_switch: float, sigma_low: float):
    rng = np.random.default_rng(seed)
    n = int(t_end / dt) + 1
    t = np.linspace(0.0, t_end, n)

    a = np.zeros(n)
    phi = np.zeros(n)
    k_struct = np.zeros(n)
    i_iso = np.zeros(n)
    c = np.zeros(n)
    c_crit = np.zeros(n)

    a[0] = 1.15
    phi[0] = 0.25
    k_struct[0] = 0.0
    i_iso[0] = 0.05

    window = max(1, int(p.tau / dt))
    sqrt_dt = math.sqrt(dt)
    steps_below_crit = 0

    for idx in range(n - 1):
        gv = grad_v(a[idx])
        c[idx] = gv * gv

        left = max(0, idx - window)
        c_avg = float(np.mean(c[left : idx + 1]))
        c_crit[idx] = p.c0 * math.exp(-p.mu_k * k_struct[idx]) + p.zeta * c_avg

        th = theta_sigmoid(c[idx] - c_crit[idx])
        g_raw = 1.0 - p.eta * p.lam_c * th
        g = max(p.g_min, g_raw)

        s_term = p.c_s * math.tanh(a[idx] + 0.7 * phi[idx])
        u_term = sat(c[idx], p.u_max)

        da = -g * gv - p.lam * s_term
        dphi = -p.kappa * phi[idx] + u_term
        dK = p.nu * c[idx] - p.delta_k * k_struct[idx]

        if t[idx] < t_switch:
            # Hold phase: allow only activation, no decay
            if c[idx] >= c_crit[idx]:
                dI = p.xi * (c[idx] - c_crit[idx]) * i_iso[idx] * (1.0 - i_iso[idx])
            else:
                dI = 0.0
        else:
            # Recovery phase with trust filter
            if c[idx] < c_crit[idx]:
                steps_below_crit += 1
            else:
                steps_below_crit = 0

            if steps_below_crit >= p.rec_steps:
                dI = -p.delta_rec * i_iso[idx]
            elif c[idx] >= c_crit[idx]:
                dI = p.xi * (c[idx] - c_crit[idx]) * i_iso[idx] * (1.0 - i_iso[idx])
            else:
                dI = 0.0

        sigma_a = p.sigma_a if t[idx] < t_switch else sigma_low
        noise = sigma_a * sqrt_dt * rng.normal()

        a[idx + 1] = a[idx] + dt * da + noise
        phi[idx + 1] = max(0.0, phi[idx] + dt * dphi)
        k_struct[idx + 1] = max(0.0, k_struct[idx] + dt * dK)
        i_iso[idx + 1] = float(np.clip(i_iso[idx] + dt * dI, 0.0, 1.0))

    c[-1] = grad_v(a[-1]) ** 2
    c_crit[-1] = p.c0 * math.exp(-p.mu_k * k_struct[-1]) + p.zeta * float(
        np.mean(c[max(0, n - window) : n])
    )

    return {"t": t, "C": c, "Ccrit": c_crit, "I": i_iso}


def main() -> None:
    out_dir = Path('outputs/experiments/20260314_thaw_controlled')
    out_dir.mkdir(parents=True, exist_ok=True)

    params = replace(NoiseParams(), eta=0.85, sigma_a=0.6, rec_steps=5, delta_rec=0.05)
    t_end = 140.0
    dt = 0.01
    t_switch = 70.0
    sigma_low = 0.1

    det = simulate_thaw_controlled(t_end, dt, params, seed=42, t_switch=t_switch, sigma_low=sigma_low)

    keys = ["t", "C", "Ccrit", "I"]
    data = np.column_stack([det[k] for k in keys])
    np.savetxt(out_dir / "thaw_controlled.csv", data, delimiter=",", header=",".join(keys), comments="")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(det["t"], det["C"], label="C(t)")
    axes[0].plot(det["t"], det["Ccrit"], label="Ccrit(t)", alpha=0.7)
    axes[0].axvline(t_switch, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("C")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(det["t"], det["I"], label="I(t)")
    axes[1].axvline(t_switch, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("I")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "thaw_controlled.png", dpi=160)
    plt.close(fig)

    with (out_dir / "RUN_NOTES.txt").open("w", encoding="utf-8") as fh:
        fh.write("Controlled thaw with hold phase\n")
        fh.write(f"eta={params.eta}\n")
        fh.write(f"sigma_high={params.sigma_a}\n")
        fh.write(f"sigma_low={sigma_low}\n")
        fh.write(f"t_switch={t_switch}\n")
        fh.write(f"rec_steps={params.rec_steps}\n")
        fh.write(f"delta_rec={params.delta_rec}\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

