"""
Multi-stress experiment to test experience accumulation (recovery time across episodes).
"""
from __future__ import annotations

import math
from dataclasses import asdict
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from simulations.simulation_network_v1 import NetworkParams, MasterParams


def run_multi_stress(
    p: MasterParams,
    net: NetworkParams,
    stress_times: list[float],
    stress_duration: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(net.seed)
    n_steps = int(net.t_end / net.dt) + 1
    t = np.linspace(0.0, net.t_end, n_steps)

    n = net.n_agents
    # ring adjacency
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        for offset in range(1, net.ring_k + 1):
            a[i, (i + offset) % n] = 1.0
            a[i, (i - offset) % n] = 1.0
    deg = a.sum(axis=1)
    deg[deg == 0.0] = 1.0

    # state
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

    stress_agents = rng.choice(n, size=max(1, int(n * net.stress_frac)), replace=False)
    stress_windows = []
    for st in stress_times:
        start = int(st / net.dt)
        end = start + max(1, int(stress_duration / net.dt))
        stress_windows.append((start, end))

    sqrt_dt = math.sqrt(net.dt)

    phase_disp = np.zeros(n_steps)
    mean_s = np.zeros(n_steps)
    calm_time = np.zeros(n_steps)

    for idx in range(n_steps - 1):
        # QRC phase
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
            c_mean = float(np.mean(y[idx] ** 2))
            g = max(net.qrc_g_min, min(net.qrc_g_max, 1.0 - net.qrc_eta * c_mean))
            mean_s_now = float(np.mean(s_buf[idx]))
            phi_gain = net.phi_gain * (1.0 + net.phi_gain_boost * (mean_s_now ** 2))
        else:
            phi[idx + 1] = phi[idx]
            g = 1.0
            phi_gain = 0.0

        # coupling
        y_neighbor = a @ (y[idx] * (1.0 - s_buf[idx]))
        coupling_term = g * net.coupling * (1.0 - s_buf[idx]) * (y_neighbor - deg * y[idx]) / deg
        if net.qrc_enabled and phi_gain != 0.0:
            coupling_term += phi_gain * (phi[idx] - y[idx]) * (1.0 - s_buf[idx])

        dx = -p.k * x[idx] - p.lam * y[idx]
        drift_y = p.alpha * x[idx] + mu[idx] * y[idx] - p.gamma * y[idx] ** 3
        metro = net.metro_amp * math.sin(net.metro_freq * t[idx] + net.metro_phase)
        noise_scale = p.sigma_noise * (p.iso_noise_scale + (1.0 - p.iso_noise_scale) * (1.0 - s_buf[idx]))
        noise = noise_scale * sqrt_dt * rng.normal(size=n)

        dH = p.sigma_h * y[idx] ** 2 - (p.delta_h + p.eta_s * s_buf[idx] + p.iso_cool * s_buf[idx]) * h[idx]
        dK = p.beta * h[idx] - p.delta_k * k_struct[idx] + p.xi * s_buf[idx] * h[idx]
        dmu = p.rho1 * h[idx] - p.rho2 * k_struct[idx] - p.rho3 * mu[idx]
        c_local = y[idx] ** 2
        ccrit_eff = p.c_crit
        theta = np.arctan2(y[idx], x[idx])
        phase_disp[idx] = 1.0 - np.abs(np.mean(np.exp(1j * theta)))
        if net.qrc_enabled:
            ccrit_eff = p.c_crit + net.ccrit_gain * (1.0 - phase_disp[idx])
            ccrit_eff = float(np.clip(ccrit_eff, net.ccrit_floor, net.ccrit_cap))
        dS = p.eps_s * (h[idx] - p.h_crit) * s_buf[idx] * (1.0 - s_buf[idx])
        dS += p.eps_s2 * (c_local - ccrit_eff) * s_buf[idx] * (1.0 - s_buf[idx])
        if net.qrc_enabled:
            match = np.cos(theta - phi[idx])
            recog = np.maximum(0.0, match - net.recog_threshold)
            dS -= net.recog_gain * recog * s_buf[idx] * (1.0 - s_buf[idx])
            if phase_disp[idx] < net.wake_disp_threshold:
                calm_time[idx] = (calm_time[idx - 1] + net.dt) if idx > 0 else net.dt
            else:
                calm_time[idx] = 0.0
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

        for start, end in stress_windows:
            if start <= idx < end:
                h[idx + 1, stress_agents] += net.stress_amp
                y[idx + 1, stress_agents] += net.stress_y_amp

        mean_s[idx] = float(np.mean(s_buf[idx]))

    mean_s[-1] = float(np.mean(s_buf[-1]))
    phase_disp[-1] = 1.0 - np.abs(np.mean(np.exp(1j * np.arctan2(y[-1], x[-1]))))

    return {
        "t": t,
        "mean_s": mean_s,
        "phase_dispersion": phase_disp,
        "stress_times": stress_times,
    }


def compute_recovery_times(t: np.ndarray, mean_s: np.ndarray, stress_times: list[float], dt: float, recovery_threshold: float) -> list[float]:
    rec = []
    n_steps = len(t)
    for st in stress_times:
        start_idx = min(int(st / dt), n_steps - 1)
        peak_idx = start_idx + int(np.argmax(mean_s[start_idx:]))
        rt = math.nan
        for idx in range(peak_idx, n_steps):
            if mean_s[idx] <= recovery_threshold:
                rt = t[idx] - t[start_idx]
                break
        rec.append(rt)
    return rec


def main() -> None:
    p = MasterParams()
    p.sigma_h = p.sigma_h * 1.35
    p.delta_h = p.delta_h * 0.75
    p.h_crit = p.h_crit * 0.85
    p.eps_s = p.eps_s * 0.6
    p.eps_s2 = 0.25
    p.c_crit = 0.6

    net = NetworkParams(
        n_agents=24,
        t_end=200.0,
        dt=0.02,
        ring_k=2,
        stress_frac=1.0,
        stress_amp=3.0,
        stress_y_amp=1.0,
        coupling=0.15,
    )
    # QRC settings
    net.qrc_enabled = True
    net.phi_gain = 0.5
    net.phi_gain_boost = 8.0
    net.qrc_g_min = 0.4
    net.phi_listen_isolated = True
    net.phi_iso_threshold = 0.2
    net.recog_threshold = 0.7
    net.recog_gain = 1.2
    net.wake_disp_threshold = 0.3
    net.wake_time_required = 4.0
    net.wake_relax_gain = 0.8
    net.ccrit_gain = 0.6
    net.ccrit_floor = 0.2
    net.ccrit_cap = 1.2
    net.coh_relax_gain = 0.6

    stress_times = [30.0, 70.0, 110.0, 150.0]
    stress_duration = 4.0

    out = run_multi_stress(p, net, stress_times, stress_duration)
    rec = compute_recovery_times(out["t"], out["mean_s"], stress_times, net.dt, net.recovery_threshold)

    # save plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(out["t"], out["mean_s"], label="mean S")
    for st in stress_times:
        axes[0].axvline(st, color="gray", alpha=0.3, linestyle="--")
    axes[0].set_ylabel("mean S")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(out["t"], out["phase_dispersion"], label="phase dispersion")
    for st in stress_times:
        axes[1].axvline(st, color="gray", alpha=0.3, linestyle="--")
    axes[1].set_ylabel("phase dispersion")
    axes[1].set_xlabel("t")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = r"E:\MyProject\meta-stable-architectures\outputs\qrc_multi_stress_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # save table
    table_path = r"E:\MyProject\meta-stable-architectures\outputs\qrc_multi_stress_table.csv"
    with open(table_path, "w", newline="") as f:
        f.write("episode,stress_time,recovery_time\n")
        for i, (st, rt) in enumerate(zip(stress_times, rec), start=1):
            f.write(f"{i},{st},{rt}\n")

    print("Wrote:", plot_path)
    print("Wrote:", table_path)
    print("Recovery times:", rec)


if __name__ == "__main__":
    main()
