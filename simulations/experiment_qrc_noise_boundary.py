from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

from simulations.simulation_network_v1 import MasterParams, NetworkParams, simulate_network


def _frange(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def _configure_master(base: MasterParams, sigma_noise: float) -> MasterParams:
    # Vol IV soft modifications on top of Vol III params
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
        n_agents=100,
        t_end=60.0,
        dt=0.025,
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

    # QRC working settings
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


def _is_recovered(recovery_time: float) -> bool:
    return not np.isnan(recovery_time)


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "qrc_noise_boundary.csv"
    out_plot = out_dir / "qrc_noise_boundary_plot.png"

    # Fast pass: coarse grid to locate the boundary quickly.
    # Increase resolution once a rough boundary is found.
    # Boundary refinement near the failure region.
    sigmas = _frange(7.4, 8.1, 0.1)
    seeds = [7, 13, 23, 31, 43]

    rows: list[dict[str, float]] = []
    for sigma in sigmas:
        rec_times: list[float] = []
        success = 0
        for seed in seeds:
            p = _configure_master(MasterParams(), sigma)
            net = _configure_network(seed)
            out = simulate_network(p=p, net=net)
            rt = float(out["recovery_time"])
            rec_times.append(rt)
            if _is_recovered(rt):
                success += 1
        success_rate = success / len(seeds)
        finite_rts = [rt for rt in rec_times if not np.isnan(rt)]
        mean_rt = float(np.mean(finite_rts)) if finite_rts else float("nan")
        rows.append(
            {
                "sigma_noise": float(sigma),
                "success_rate": float(success_rate),
                "mean_recovery_time": mean_rt,
                "runs": float(len(seeds)),
            }
        )
        print(f"sigma={sigma:.2f} success_rate={success_rate:.2f} mean_rt={mean_rt}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma_noise", "success_rate", "mean_recovery_time", "runs"])
        writer.writeheader()
        writer.writerows(rows)

    # Plot
    try:
        import matplotlib.pyplot as plt

        sig = [r["sigma_noise"] for r in rows]
        succ = [r["success_rate"] for r in rows]
        rt = [r["mean_recovery_time"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
        ax1.plot(sig, succ, marker="o", label="Success rate")
        ax1.set_xlabel("sigma_noise")
        ax1.set_ylabel("Success rate")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(sig, rt, marker="s", color="tab:orange", label="Mean recovery time")
        ax2.set_ylabel("Mean recovery time")

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_plot, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    print(f"Wrote {out_csv}")
    print(f"Plot {out_plot}")


if __name__ == "__main__":
    main()
