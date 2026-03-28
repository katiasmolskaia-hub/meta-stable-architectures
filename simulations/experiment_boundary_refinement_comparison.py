from __future__ import annotations

from pathlib import Path
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulations.experiment_anticipatory_pause_boundary import (
    configure_master,
    full_model_network,
    make_grouped_values,
    run_metrics,
)
from simulations.simulation_network_v1 import MasterParams, simulate_network


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [7.8, 8.0, 8.2]
    seeds = [7, 11]
    topology = "spatial_local"
    variants = {
        "full_model": (False, False, False),
        "full_model_with_pause": (True, False, False),
        "full_model_with_pause_attention": (True, True, False),
        "full_model_with_pause_attention_recognition": (True, True, True),
    }

    rows = []
    for sigma in sigmas:
        alpha = make_grouped_values(180, (0.78, 0.86, 0.95))
        sigma_arr = make_grouped_values(180, (0.30, 0.22, 0.14))
        h_crit = make_grouped_values(180, (0.62, 0.69, 0.77))
        base_master = MasterParams(alpha=alpha, sigma_noise=sigma_arr, h_crit=h_crit)
        p = configure_master(base_master, sigma)

        for variant, flags in variants.items():
            with_pause, with_attention, with_recognition = flags
            for seed in seeds:
                net = full_model_network(topology, seed, with_pause, with_attention, with_recognition)
                out = simulate_network(p=p, net=net)
                metrics = run_metrics(out)
                rows.append(
                    {
                        "sigma_noise": sigma,
                        "variant": variant,
                        "seed": seed,
                        **metrics,
                    }
                )

    summary_rows = []
    for sigma in sigmas:
        for variant in variants:
            sub = [row for row in rows if row["sigma_noise"] == sigma and row["variant"] == variant]
            finite_tail = [float(r["tail_span"]) for r in sub if not np.isnan(float(r["tail_span"]))]
            summary_rows.append(
                {
                    "sigma_noise": sigma,
                    "variant": variant,
                    "success_rate": float(np.mean([float(r["success"]) for r in sub])),
                    "mean_tail_span": float(np.mean(finite_tail)) if finite_tail else float("nan"),
                    "mean_peak_mean_s": float(np.mean([float(r["peak_mean_s"]) for r in sub])),
                    "mean_peak_error": float(np.mean([float(r["peak_mean_error"]) for r in sub])),
                    "mean_peak_turbulence": float(np.mean([float(r["peak_mean_turbulence"]) for r in sub])),
                    "mean_peak_cascade_fraction": float(np.mean([float(r["peak_cascade_fraction"]) for r in sub])),
                    "mean_final_attunement": float(np.mean([float(r["final_attunement"]) for r in sub])),
                    "mean_pause": float(np.mean([float(r["mean_pause"]) for r in sub])),
                    "mean_recognition": float(np.mean([float(r["mean_recognition"]) for r in sub])),
                }
            )

    csv_path = out_dir / "boundary_refinement_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "boundary_refinement_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
