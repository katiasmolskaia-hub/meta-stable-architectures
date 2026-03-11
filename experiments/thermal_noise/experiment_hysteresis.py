"""
Hysteresis experiment for Langevin noise on a(t).
Sigma path: high -> low (e.g., 0.6 -> 0.2) at a switch time.
Tracks I(t), C(t), and recovery time after the switch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from simulation_noise import NoiseParams, grad_v, sat, theta_sigmoid


def _simulate_langevin_with_sigma_schedule(
    t_end: float,
    dt: float,
    p: NoiseParams,
    sigma_high: float,
    sigma_low: float,
    t_switch: float,
    seed: int = 42,
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
    v = np.zeros(n)
    sigma_series = np.zeros(n)

    # Initial state
    a[0] = 1.15
    phi[0] = 0.25
    k_struct[0] = 0.0
    i_iso[0] = 0.05

    window = max(1, int(p.tau / dt))
    sqrt_dt = math.sqrt(dt)

    for idx in range(n - 1):
        t_now = t[idx]
        sigma_a = sigma_high if t_now < t_switch else sigma_low
        sigma_series[idx] = sigma_a

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

        noise = sigma_a * sqrt_dt * rng.normal()
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
    sigma_series[-1] = sigma_low

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
        "sigma_a": sigma_series,
    }


def _recovery_time(det: dict[str, np.ndarray], t_switch: float, threshold: float, window_steps: int) -> float:
    t = det["t"]
    i_iso = det["I"]
    idx0 = int(np.searchsorted(t, t_switch))
    n = len(t)
    for idx in range(idx0, n - window_steps):
        if i_iso[idx] <= threshold:
            if float(np.mean(i_iso[idx : idx + window_steps])) <= threshold:
                return float(t[idx] - t_switch)
    return float("nan")


def _save_series_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    keys = list(data.keys())
    matrix = np.column_stack([np.asarray(data[k]) for k in keys])
    header = ",".join(keys)
    np.savetxt(path, matrix, delimiter=",", header=header, comments="")


def _plot_timeseries(out_file: Path, det: dict[str, np.ndarray], t_switch: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = det["t"]
    c = det["C"]
    i_iso = det["I"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, c, label="C(t)")
    axes[0].plot(t, det["Ccrit"], label="Ccrit(t)", alpha=0.7)
    axes[0].axvline(t_switch, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("C")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, i_iso, label="I(t)")
    axes[1].axvline(t_switch, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("I")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)

def _update_experiment_log(readme_path: Path, run_id: str, purpose: str, sigma_desc: str, eta_desc: str) -> None:
    if not readme_path.exists():
        return

    lines = readme_path.read_text(encoding="utf-8").splitlines()
    row = f"| {run_id} | {purpose} | {sigma_desc} | {eta_desc} |"
    if row in lines:
        return

    insert_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Add new runs here"):
            insert_idx = idx
            break

    if insert_idx is None:
        lines.append(row)
    else:
        lines.insert(insert_idx, row)

    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_root = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
    run_tag = "hysteresis_sigma_0.6_to_0.2"
    run_dir = out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    t_end = 160.0
    dt = 0.01
    seed = 42

    sigma_high = 0.60
    sigma_low = 0.20
    t_switch = 80.0

    eta_list = [0.70, 0.75, 0.80]

    recovery_threshold = 0.10
    recovery_window_steps = 200

    metrics_rows: list[dict[str, float]] = []

    for eta in eta_list:
        params = replace(NoiseParams(), eta=eta)
        det = _simulate_langevin_with_sigma_schedule(
            t_end=t_end,
            dt=dt,
            p=params,
            sigma_high=sigma_high,
            sigma_low=sigma_low,
            t_switch=t_switch,
            seed=seed,
        )

        crisis_share_total = float(np.mean(det["C"] > det["Ccrit"]))
        idx_switch = int(np.searchsorted(det["t"], t_switch))
        crisis_share_after = float(np.mean(det["C"][idx_switch:] > det["Ccrit"][idx_switch:]))
        recovery_time = _recovery_time(det, t_switch, recovery_threshold, recovery_window_steps)

        metrics = {
            "eta": eta,
            "sigma_high": sigma_high,
            "sigma_low": sigma_low,
            "t_switch": t_switch,
            "I_final": float(det["I"][-1]),
            "I_max": float(np.max(det["I"])),
            "crisis_share_total": crisis_share_total,
            "crisis_share_after": crisis_share_after,
            "recovery_time": recovery_time,
        }
        metrics_rows.append(metrics)

        _save_series_csv(run_dir / f"timeseries_eta{eta:.2f}.csv", det)
        _plot_timeseries(run_dir / f"timeseries_eta{eta:.2f}.png", det, t_switch)

    metrics_path = run_dir / "hysteresis_metrics.csv"
    header = list(metrics_rows[0].keys())
    with metrics_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in metrics_rows:
            fh.write(",".join(f"{row[k]:.8f}" for k in header) + "\n")

    notes_path = run_dir / "RUN_NOTES.txt"
    with notes_path.open("w", encoding="utf-8") as fh:
        fh.write("Hysteresis experiment: sigma high -> low\n")
        fh.write(f"run_tag={run_tag}\n")
        fh.write(f"t_end={t_end}\n")
        fh.write(f"dt={dt}\n")
        fh.write(f"seed={seed}\n")
        fh.write(f"sigma_high={sigma_high}\n")
        fh.write(f"sigma_low={sigma_low}\n")
        fh.write(f"t_switch={t_switch}\n")
        fh.write(f"eta_list={eta_list}\n")


        fh.write(f"recovery_threshold={recovery_threshold}\n")
        fh.write(f"recovery_window_steps={recovery_window_steps}\n")

    readme_path = Path(__file__).with_name("README.md")
    _update_experiment_log(
        readme_path,
        run_id=run_dir.name,
        purpose=f"Hysteresis: sigma {sigma_high} -> {sigma_low}",
        sigma_desc=f"schedule {sigma_high} -> {sigma_low}",
        eta_desc=str(eta_list),
    )

    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
