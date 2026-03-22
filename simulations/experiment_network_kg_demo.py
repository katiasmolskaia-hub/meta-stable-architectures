from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

from simulations.simulation_network_v1 import MasterParams, NetworkParams, simulate_network


def summarize(out: dict[str, object], label: str) -> dict[str, float | str]:
    return {
        "label": label,
        "recovery_time": float(out["recovery_time"]),
        "peak_dispersion": float(out["phase_dispersion"].max()),
        "final_dispersion": float(out["phase_dispersion"][-1]),
        "peak_iso": float(out["fraction_isolated"].max()),
        "final_iso": float(out["fraction_isolated"][-1]),
        "final_mean_h": float(out["mean_h"][-1]),
        "final_mean_k": float(out["K"][-1].mean()),
        "final_kg": float(out["group_memory"][-1]),
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = MasterParams()
    base_net = NetworkParams(
        n_agents=100,
        t_end=220.0,
        dt=0.01,
        coupling=0.15,
        topology="small_world",
        ring_k=2,
        sw_rewire=0.15,
        delay_mode="grouped",
        delay_group_fracs=(0.5, 0.3, 0.2),
        delay_group_steps=(0, 4, 8),
        stress_time=30.0,
        stress_amp=3.0,
        stress_frac=0.24,
        stress_duration=8.0,
        stress_y_amp=1.0,
        qrc_enabled=True,
        phi_kappa=1.0,
        phi_gain=0.5,
        phi_gain_boost=8.0,
        qrc_g_min=0.4,
        recog_threshold=0.7,
        recog_gain=1.2,
        wake_disp_threshold=0.3,
        wake_time_required=4.0,
        wake_relax_gain=0.8,
        coh_relax_gain=0.6,
    )
    kg_net = NetworkParams(
        **{**base_net.__dict__, "kg_enabled": True, "kg_lambda": 0.06, "kg_decay": 0.01, "kg_phi_boost": 2.0, "kg_wake_boost": 1.5}
    )

    base_out = simulate_network(p, base_net)
    kg_out = simulate_network(p, kg_net)

    summaries = [summarize(base_out, "qrc_no_kg"), summarize(kg_out, "qrc_with_kg")]
    for row in summaries:
        print(
            f"{row['label']}: recovery={row['recovery_time']:.3f}, peak_disp={row['peak_dispersion']:.3f}, "
            f"final_disp={row['final_dispersion']:.3f}, peak_iso={row['peak_iso']:.3f}, final_kg={row['final_kg']:.3f}"
        )

    csv_path = out_dir / "network_kg_demo_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        import csv

        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(base_out["t"], base_out["phase_dispersion"], label="no kg")
        axes[0].plot(kg_out["t"], kg_out["phase_dispersion"], label="with kg")
        axes[0].set_ylabel("phase dispersion")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(base_out["t"], base_out["fraction_isolated"], label="no kg")
        axes[1].plot(kg_out["t"], kg_out["fraction_isolated"], label="with kg")
        axes[1].set_ylabel("fraction isolated")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(base_out["t"], base_out["mean_h"], label="no kg")
        axes[2].plot(kg_out["t"], kg_out["mean_h"], label="with kg")
        axes[2].set_ylabel("mean H")
        axes[2].set_xlabel("time")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "network_kg_demo_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
