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


def _configure_master(sigma_noise: float) -> MasterParams:
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
            "sigma_noise": sigma_noise,
        }
    )


def _configure_network(seed: int) -> NetworkParams:
    net = NetworkParams(
        n_agents=1000,
        t_end=120.0,
        dt=0.02,
        coupling=0.15,
        ring_k=2,
        seed=seed,
        topology="small_world",
        sw_rewire=0.1,
        stress_time=30.0,
        stress_amp=1.8,
        stress_frac=0.24,
        stress_duration=2.0,
        stress_y_amp=1.0,
    )
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
    out_csv = out_dir / "qrc_per_agent_energy_n1000.csv"
    out_plot = out_dir / "qrc_per_agent_energy_n1000_plot.png"
    out_summary = out_dir / "qrc_per_agent_energy_n1000_summary.txt"

    p = _configure_master(0.22)
    net = _configure_network(7)
    out = simulate_network(p=p, net=net)

    t = out["t"]
    y = out["y"]
    h = out["H"]
    s = out["S"]
    k = out["K"]
    e = _energy_proxy(y, h, s)
    e_eff = e / (1.0 + 0.9 * k)

    stress_end = net.stress_time + net.stress_duration
    mask = t >= stress_end
    e_mean = np.mean(e[mask], axis=0)
    e_eff_mean = np.mean(e_eff[mask], axis=0)
    ratio = e_eff_mean / np.maximum(e_mean, 1e-9)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "E_mean", "E_eff_mean", "E_eff_over_E"])
        for i in range(net.n_agents):
            writer.writerow([i, f"{e_mean[i]:.6f}", f"{e_eff_mean[i]:.6f}", f"{ratio[i]:.6f}"])

    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q))

    tail_drop_p95 = 100.0 * (1.0 - pct(e_eff_mean, 95) / max(pct(e_mean, 95), 1e-9))
    tail_drop_p99 = 100.0 * (1.0 - pct(e_eff_mean, 99) / max(pct(e_mean, 99), 1e-9))

    out_summary.write_text(
        "\n".join(
            [
                "Per-agent energy (post-stress window)",
                f"N={net.n_agents}, sigma_noise={p.sigma_noise}, stress_end={stress_end}",
                "Energy proxy: E = 0.2 + 0.6*y^2 + 0.4*H + 0.3*S",
                "E_eff = E / (1 + 0.9*K)",
                f"E_mean p50={pct(e_mean, 50):.4f} p90={pct(e_mean, 90):.4f} p95={pct(e_mean, 95):.4f} p99={pct(e_mean, 99):.4f}",
                f"E_eff_mean p50={pct(e_eff_mean, 50):.4f} p90={pct(e_eff_mean, 90):.4f} p95={pct(e_eff_mean, 95):.4f} p99={pct(e_eff_mean, 99):.4f}",
                f"Tail ratio (p95): {pct(e_eff_mean, 95) / max(pct(e_mean, 95), 1e-9):.4f}",
                f"Tail ratio (p99): {pct(e_eff_mean, 99) / max(pct(e_mean, 99), 1e-9):.4f}",
                f"Tail drop (p95): {tail_drop_p95:.2f}%",
                f"Tail drop (p99): {tail_drop_p99:.2f}%",
            ]
        ),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt

        order = np.argsort(e_mean)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].plot(e_mean[order], label="E_mean (sorted)")
        axes[0].plot(e_eff_mean[order], label="E_eff_mean (sorted)")
        axes[0].set_title("Per-agent energy (sorted)")
        axes[0].set_xlabel("agent rank")
        axes[0].set_ylabel("energy")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].hist(e_mean, bins=24, alpha=0.6, label="E_mean")
        axes[1].hist(e_eff_mean, bins=24, alpha=0.6, label="E_eff_mean")
        axes[1].set_title("Energy distribution (post-stress)")
        axes[1].set_xlabel("energy")
        axes[1].set_ylabel("count")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
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
