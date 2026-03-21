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
        }
    )


def _configure_network(stress_duration: float, seed: int) -> NetworkParams:
    net = NetworkParams(
        n_agents=1000,
        t_end=180.0,
        dt=0.02,
        coupling=0.15,
        ring_k=2,
        seed=seed,
        topology="small_world",
        sw_rewire=0.1,
        stress_time=30.0,
        stress_amp=3.0,
        stress_frac=0.24,
        stress_duration=stress_duration,
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


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "qrc_k_saturation_n1000.csv"
    out_plot = out_dir / "qrc_k_saturation_n1000_plot.png"

    durations = [20, 60, 100]
    seeds = [7]

    rows: list[dict[str, float]] = []

    for dur in durations:
        k_maxes = []
        k_final = []
        e_eff = []
        rec = []
        success = 0

        for seed in seeds:
            p = _configure_master()
            net = _configure_network(dur, seed)
            out = simulate_network(p=p, net=net)

            k = out["K"]
            mean_k = np.mean(k, axis=1)
            k_maxes.append(float(np.max(mean_k)))
            k_final.append(float(mean_k[-1]))

            # simple global energy proxy like in per-agent, but averaged
            e = 0.2 + 0.6 * (out["y"] ** 2) + 0.4 * out["H"] + 0.3 * out["S"]
            e_eff_t = e / (1.0 + 0.9 * out["K"])
            e_eff.append(float(np.mean(e_eff_t)))

            rt = float(out["recovery_time"])
            rec.append(rt)
            if not np.isnan(rt):
                success += 1

        rows.append(
            {
                "stress_duration": float(dur),
                "success_rate": success / len(seeds),
                "recovery_time_mean": float(np.nanmean(rec)),
                "K_mean_peak": float(np.mean(k_maxes)),
                "K_mean_final": float(np.mean(k_final)),
                "E_eff_mean": float(np.mean(e_eff)),
            }
        )

        print(
            f"dur={dur} success={success}/{len(seeds)} "
            f"K_peak={rows[-1]['K_mean_peak']:.3f} K_final={rows[-1]['K_mean_final']:.3f} "
            f"E_eff_mean={rows[-1]['E_eff_mean']:.3f} rt={rows[-1]['recovery_time_mean']:.3f}"
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stress_duration",
                "success_rate",
                "recovery_time_mean",
                "K_mean_peak",
                "K_mean_final",
                "E_eff_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        x = [r["stress_duration"] for r in rows]
        k_peak = [r["K_mean_peak"] for r in rows]
        k_final = [r["K_mean_final"] for r in rows]
        rt = [r["recovery_time_mean"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
        ax1.plot(x, k_peak, marker="o", label="K_mean_peak")
        ax1.plot(x, k_final, marker="s", label="K_mean_final")
        ax1.set_xlabel("stress_duration")
        ax1.set_ylabel("K (mean)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, rt, marker="^", color="tab:orange", label="recovery_time")
        ax2.set_ylabel("recovery_time")

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_plot, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
