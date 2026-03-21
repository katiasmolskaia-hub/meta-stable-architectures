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


def _configure_network(seed: int) -> NetworkParams:
    net = NetworkParams(
        n_agents=100,
        t_end=80.0,
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


def _is_recovered(rt: float) -> bool:
    return not np.isnan(rt)


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "qrc_delay_boundary.csv"
    out_plot = out_dir / "qrc_delay_boundary_plot.png"

    p = _configure_master()
    seeds = [7, 13]

    fixed_steps = [0, 2, 4, 6, 8, 10, 14, 18]
    grouped_steps = [
        (1, 4, 8),
        (2, 6, 10),
        (4, 8, 12),
    ]
    grouped_fracs = (0.6, 0.3, 0.1)

    rows: list[dict[str, float | str]] = []

    # Fixed delays
    for d in fixed_steps:
        success = 0
        rts: list[float] = []
        for seed in seeds:
            net = _configure_network(seed)
            net.delay_mode = "fixed"
            net.delay_steps = d
            out = simulate_network(p=p, net=net)
            rt = float(out["recovery_time"])
            rts.append(rt)
            if _is_recovered(rt):
                success += 1
        rows.append(
            {
                "mode": "fixed",
                "delay_steps": d,
                "delay_group_steps": "",
                "success_rate": success / len(seeds),
                "recovery_time_mean": float(np.nanmean(rts)),
            }
        )
        print(f"fixed d={d} success={success}/{len(seeds)} rt={rows[-1]['recovery_time_mean']}")

    # Grouped delays
    for steps in grouped_steps:
        success = 0
        rts = []
        for seed in seeds:
            net = _configure_network(seed)
            net.delay_mode = "grouped"
            net.delay_group_fracs = grouped_fracs
            net.delay_group_steps = steps
            out = simulate_network(p=p, net=net)
            rt = float(out["recovery_time"])
            rts.append(rt)
            if _is_recovered(rt):
                success += 1
        rows.append(
            {
                "mode": "grouped",
                "delay_steps": float(np.max(steps)),
                "delay_group_steps": "/".join(str(x) for x in steps),
                "success_rate": success / len(seeds),
                "recovery_time_mean": float(np.nanmean(rts)),
            }
        )
        print(
            f"grouped steps={steps} success={success}/{len(seeds)} rt={rows[-1]['recovery_time_mean']}"
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "delay_steps",
                "delay_group_steps",
                "success_rate",
                "recovery_time_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Plot: fixed delays only for readability
    try:
        import matplotlib.pyplot as plt

        fixed = [r for r in rows if r["mode"] == "fixed"]
        x = [r["delay_steps"] for r in fixed]
        succ = [r["success_rate"] for r in fixed]
        rt = [r["recovery_time_mean"] for r in fixed]

        fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
        ax1.plot(x, succ, marker="o", label="Success rate")
        ax1.set_xlabel("delay_steps")
        ax1.set_ylabel("success_rate")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, rt, marker="s", color="tab:orange", label="Mean recovery time")
        ax2.set_ylabel("recovery_time")

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_plot, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
