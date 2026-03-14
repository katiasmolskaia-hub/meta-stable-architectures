"""Wisdom transfer experiment: gated learning + regime jump detection."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation_noise import NoiseParams, grad_v, sat, theta_sigmoid


def simulate_wisdom(
    t_end: float,
    dt: float,
    p: NoiseParams,
    seed: int,
    sigma_high: float,
    sigma_low: float,
    sigma_mid: float,
    t1: float,
    t2: float,
    jump_threshold: float,
    kappa_jump: float,
    k_gate: float,
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
    c0_base = np.zeros(n)

    a[0] = 1.15
    phi[0] = 0.25
    k_struct[0] = 0.0
    i_iso[0] = 0.05
    c0_base[0] = p.c0

    window = max(1, int(p.tau / dt))
    sqrt_dt = math.sqrt(dt)

    for idx in range(n - 1):
        gv = grad_v(a[idx])
        c[idx] = gv * gv

        left = max(0, idx - window)
        c_avg = float(np.mean(c[left : idx + 1]))

        # dynamic threshold with base + knowledge
        c_crit[idx] = c0_base[idx] * math.exp(-p.mu_k * k_struct[idx]) + p.zeta * c_avg
        th = theta_sigmoid(c[idx] - c_crit[idx])

        g_raw = 1.0 - p.eta * p.lam_c * th
        g[idx] = max(p.g_min, g_raw)

        s_term = p.c_s * math.tanh(a[idx] + 0.7 * phi[idx])
        u_term = sat(c[idx], p.u_max)

        da = -g[idx] * gv - p.lam * s_term
        dphi = -p.kappa * phi[idx] + u_term

        # Gated learning: grow K only when isolation is high
        if i_iso[idx] > k_gate:
            dK = p.nu * c[idx] - p.delta_k * k_struct[idx]
        else:
            dK = -p.delta_k * k_struct[idx]

        # Isolation dynamics (reactive)
        dI = p.xi * (c[idx] - c_crit[idx]) * i_iso[idx] * (1.0 - i_iso[idx])

        # Regime jump detection -> wisdom transfer to base threshold
        if idx > 0 and abs(a[idx] - a[idx - 1]) > jump_threshold:
            c0_base[idx] = c0_base[idx] + kappa_jump * k_struct[idx]
        else:
            c0_base[idx] = c0_base[idx]

        # noise schedule: crisis -> recovery -> new stress
        if t[idx] < t1:
            sigma_a = sigma_high
        elif t[idx] < t2:
            sigma_a = sigma_low
        else:
            sigma_a = sigma_mid

        noise = sigma_a * sqrt_dt * rng.normal()
        a[idx + 1] = a[idx] + dt * da + noise
        phi[idx + 1] = max(0.0, phi[idx] + dt * dphi)
        k_struct[idx + 1] = max(0.0, k_struct[idx] + dt * dK)
        i_iso[idx + 1] = float(np.clip(i_iso[idx] + dt * dI, 0.0, 1.0))
        c0_base[idx + 1] = c0_base[idx]

    c[-1] = grad_v(a[-1]) ** 2
    c_crit[-1] = c0_base[-1] * math.exp(-p.mu_k * k_struct[-1]) + p.zeta * float(
        np.mean(c[max(0, n - window) : n])
    )

    return {
        "t": t,
        "a": a,
        "C": c,
        "Ccrit": c_crit,
        "I": i_iso,
        "K": k_struct,
        "c0_base": c0_base,
    }


def main() -> None:
    out_dir = Path('outputs/experiments/20260314_wisdom')
    out_dir.mkdir(parents=True, exist_ok=True)

    params = replace(NoiseParams(), eta=0.85, sigma_a=0.6)
    det = simulate_wisdom(
        t_end=180.0,
        dt=0.01,
        p=params,
        seed=42,
        sigma_high=0.6,
        sigma_low=0.1,
        sigma_mid=0.4,
        t1=60.0,
        t2=120.0,
        jump_threshold=0.06,
        kappa_jump=0.10,
        k_gate=0.5,
    )

    keys = ["t", "C", "Ccrit", "I", "K", "c0_base"]
    data = np.column_stack([det[k] for k in keys])
    np.savetxt(out_dir / "wisdom_traces.csv", data, delimiter=",", header=",".join(keys), comments="")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(det["t"], det["C"], label="C(t)")
    axes[0].plot(det["t"], det["Ccrit"], label="Ccrit(t)", alpha=0.7)
    axes[0].set_ylabel("C")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(det["t"], det["I"], label="I(t)")
    axes[1].plot(det["t"], det["K"], label="K(t)")
    axes[1].set_ylabel("I, K")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(det["t"], det["c0_base"], label="C0 base")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("C0 base")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "wisdom_traces.png", dpi=160)
    plt.close(fig)

    with (out_dir / "RUN_NOTES.txt").open("w", encoding="utf-8") as fh:
        fh.write("Wisdom transfer experiment\n")
        fh.write("gated learning: K grows only when I > 0.5\n")
        fh.write("jump detection: |a_t - a_{t-1}| > 0.06\n")
        fh.write("C0 base increment: 0.10 * K\n")
        fh.write("sigma schedule: 0.6 -> 0.1 -> 0.4 (t1=60, t2=120)\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
