"""Thaw (recovery) experiment for ODE model with controlled recovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation_noise import NoiseParams, simulate_master_langevin


def main() -> None:
    out_dir = Path('outputs/experiments/20260314_thaw')
    out_dir.mkdir(parents=True, exist_ok=True)

    params = replace(NoiseParams(), eta=0.85, sigma_a=0.6, rec_steps=5, delta_rec=0.05)
    t_end = 140.0
    dt = 0.01
    t_switch = 70.0

    def schedule(t: float) -> float:
        return params.sigma_a if t < t_switch else 0.0

    det = simulate_master_langevin(t_end=t_end, dt=dt, p=params, seed=42, sigma_schedule=schedule)

    # Save CSV
    keys = ["t", "C", "Ccrit", "I"]
    data = np.column_stack([det[k] for k in keys])
    np.savetxt(out_dir / "thaw_traces.csv", data, delimiter=",", header=",".join(keys), comments="")

    # Plot C/Ccrit and I
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
    fig.savefig(out_dir / "thaw_traces.png", dpi=160)
    plt.close(fig)

    with (out_dir / "RUN_NOTES.txt").open("w", encoding="utf-8") as fh:
        fh.write("Thaw (controlled recovery) test\n")
        fh.write(f"eta={params.eta}\n")
        fh.write(f"sigma_high={params.sigma_a}\n")
        fh.write(f"sigma_low=0.0\n")
        fh.write(f"t_switch={t_switch}\n")
        fh.write(f"rec_steps={params.rec_steps}\n")
        fh.write(f"delta_rec={params.delta_rec}\n")
        fh.write(f"t_end={t_end}\n")
        fh.write(f"dt={dt}\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
