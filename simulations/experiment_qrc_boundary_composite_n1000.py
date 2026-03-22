from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

from simulations.simulation_network_v1 import MasterParams, NetworkParams, simulate_network


def _configure_master() -> MasterParams:
    base = MasterParams()
    return MasterParams(
        **{
            **asdict(base),
            "sigma_h": base.sigma_h * 1.35,
            "delta_h": base.delta_h * 0.75,
            "h_crit": base.h_crit * 0.85,
            "eps_s": base.eps_s * 0.6,
            "eps_s2": 0.25,
            "c_crit": 0.6,
            "sigma_noise": 8.0,
        }
    )


def _configure_network(seed: int) -> NetworkParams:
    net = NetworkParams(
        n_agents=1000,
        t_end=180.0,
        dt=0.025,
        coupling=0.15,
        ring_k=2,
        seed=seed,
        topology="small_world",
        sw_rewire=0.1,
        delay_mode="grouped",
        delay_group_fracs=(0.6, 0.3, 0.1),
        delay_group_steps=(2, 6, 10),
        stress_time=30.0,
        stress_amp=3.0,
        stress_frac=0.24,
        stress_duration=100.0,
        stress_y_amp=1.0,
    )

    # RC working settings: same bounded phase-coordination layer as the published preset.
    net.qrc_enabled = True
    net.phi_kappa = 1.0
    net.phi_gain = 0.5
    net.phi_gain_boost = 8.0
    net.qrc_g_min = 0.4
    net.recog_threshold = 0.7
    net.recog_gain = 1.2
    net.coh_relax_gain = 0.6
    net.wake_disp_threshold = 0.3
    net.wake_time_required = 4.0
    net.wake_relax_gain = 0.8
    net.phi_iso_threshold = 0.2

    return net


def _energy_proxy(y: np.ndarray, h: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 0.2 + 0.6 * (y**2) + 0.4 * h + 0.3 * s


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "qrc_boundary_composite_n1000.csv"
    out_plot = out_dir / "qrc_boundary_composite_n1000_plot.png"
    out_summary = out_dir / "qrc_boundary_composite_n1000_summary.txt"

    p = _configure_master()
    net = _configure_network(seed=7)
    out = simulate_network(p=p, net=net)

    t = out["t"]
    y = out["y"]
    h = out["H"]
    s = out["S"]
    k = out["K"]
    mean_s = out["mean_s"]
    phase_disp = out["phase_dispersion"]
    rec = float(out["recovery_time"])

    e = _energy_proxy(y, h, s)
    e_eff = e / (1.0 + 0.9 * k)

    stress_end = net.stress_time + net.stress_duration
    mask = t >= stress_end
    e_mean = np.mean(e[mask], axis=0)
    e_eff_mean = np.mean(e_eff[mask], axis=0)

    summary_lines = [
        "Boundary-composite run (N=1000)",
        f"sigma_noise={p.sigma_noise}",
        f"stress_duration={net.stress_duration}",
        f"delay_mode={net.delay_mode}",
        f"delay_group_steps={net.delay_group_steps}",
        f"recovery_time={rec}",
        f"K_peak={float(np.max(np.mean(k, axis=1))):.4f}",
        f"mean_S_final={float(mean_s[-1]):.6f}",
        f"phase_disp_final={float(phase_disp[-1]):.6f}",
        f"E_tail_p95={float(np.percentile(e_mean, 95)):.4f}",
        f"E_eff_tail_p95={float(np.percentile(e_eff_mean, 95)):.4f}",
    ]
    out_summary.write_text("\n".join(summary_lines), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for line in summary_lines[1:]:
            key, value = line.split("=", 1)
            writer.writerow([key, value])

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(t, mean_s, label="mean S")
        axes[0, 0].set_title("Mean isolation")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(t, phase_disp, label="phase dispersion")
        axes[0, 1].set_title("Phase dispersion")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(t, np.mean(k, axis=1), label="K mean")
        axes[1, 0].set_title("K mean")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(t, np.mean(e_eff, axis=1), label="E_eff mean")
        axes[1, 1].set_title("Effective energy")
        axes[1, 1].grid(True, alpha=0.3)
        for ax in axes.flat:
            ax.set_xlabel("time")
        fig.tight_layout()
        fig.savefig(out_plot, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
