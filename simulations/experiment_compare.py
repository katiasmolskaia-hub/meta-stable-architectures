from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np

from simulation_v2 import MasterParams, simulate_master_deterministic, simulate_memory_sde


def _save_series_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    keys = list(data.keys())
    matrix = np.column_stack([np.asarray(data[k]) for k in keys])
    header = ",".join(keys)
    np.savetxt(path, matrix, delimiter=",", header=header, comments="")


def _compute_metrics(det: dict[str, np.ndarray], sde: dict[str, np.ndarray]) -> dict[str, float]:
    c = det["C"]
    c_crit = det["Ccrit"]
    crisis_mask = c > c_crit
    metrics = {
        "det_a_final": float(det["a"][-1]),
        "det_phi_final": float(det["phi"][-1]),
        "det_K_final": float(det["K"][-1]),
        "det_I_final": float(det["I"][-1]),
        "det_crisis_share": float(np.mean(crisis_mask)),
        "det_crisis_peak_margin": float(np.max(c - c_crit)),
        "det_g_min": float(np.min(det["g"])),
        "det_g_max": float(np.max(det["g"])),
        "sde_x_final": float(sde["x"][-1]),
        "sde_y_final": float(sde["y"][-1]),
        "sde_H_final": float(sde["H"][-1]),
        "sde_K_final": float(sde["K"][-1]),
        "sde_mu_final": float(sde["mu"][-1]),
        "sde_S_final": float(sde["S"][-1]),
        "sde_y_min": float(np.min(sde["y"])),
        "sde_y_max": float(np.max(sde["y"])),
        "sde_H_max": float(np.max(sde["H"])),
        "sde_K_max": float(np.max(sde["K"])),
        "sde_S_max": float(np.max(sde["S"])),
    }
    return metrics


def _save_metrics_csv(path: Path, rows: dict[str, dict[str, float]]) -> None:
    all_metric_names: list[str] = sorted({k for r in rows.values() for k in r.keys()})
    with path.open("w", encoding="utf-8") as fh:
        fh.write("scenario," + ",".join(all_metric_names) + "\n")
        for scenario, values in rows.items():
            line = [scenario]
            for metric in all_metric_names:
                line.append(f"{values.get(metric, np.nan):.8f}")
            fh.write(",".join(line) + "\n")


def _print_comparison_table(rows: dict[str, dict[str, float]]) -> None:
    baseline = rows["baseline"]
    selected = [
        "det_crisis_share",
        "det_crisis_peak_margin",
        "det_I_final",
        "sde_H_max",
        "sde_S_max",
        "sde_K_max",
    ]

    for scenario in rows:
        if scenario == "baseline":
            continue
        print(f"\nComparison (baseline vs {scenario}):")
        print(f"{'metric':26} {'baseline':>12} {scenario:>12} {'delta':>12}")
        for metric in selected:
            b = baseline[metric]
            s = rows[scenario][metric]
            d = s - b
            print(f"{metric:26} {b:12.6f} {s:12.6f} {d:12.6f}")


def _plot_comparison(
    out_file: Path,
    det_runs: dict[str, dict[str, np.ndarray]],
    sde_runs: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "baseline": "-",
        "stress_1": "--",
        "stress_2": ":",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for name, det in det_runs.items():
        ls = styles.get(name, "-")
        axes[0, 0].plot(det["t"], det["C"], linestyle=ls, label=f"C {name}")
        axes[0, 0].plot(det["t"], det["Ccrit"], linestyle=ls, alpha=0.7, label=f"Ccrit {name}")
    axes[0, 0].set_title("Crisis intensity vs threshold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for name, det in det_runs.items():
        ls = styles.get(name, "-")
        axes[0, 1].plot(det["t"], det["I"], linestyle=ls, label=f"I {name}")
    axes[0, 1].set_title("Isolation variable I(t)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for name, sde in sde_runs.items():
        ls = styles.get(name, "-")
        axes[1, 0].plot(sde["t"], sde["H"], linestyle=ls, label=f"H {name}")
        axes[1, 0].plot(sde["t"], sde["S"], linestyle=ls, alpha=0.8, label=f"S {name}")
    axes[1, 0].set_title("Memory and suppression buffer")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    for name, sde in sde_runs.items():
        ls = styles.get(name, "-")
        axes[1, 1].plot(sde["t"], sde["y"], linestyle=ls, label=f"y {name}")
    axes[1, 1].set_title("Fast stochastic signal y(t)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def main() -> None:
    out_root = Path(__file__).resolve().parents[1] / "outputs" / "simulations"
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_params = MasterParams()
    stress_1_params = replace(
        baseline_params,
        c0=0.02,
        zeta=0.15,
        eta=0.80,
        lam_c=1.20,
        sigma_noise=0.35,
        sigma_h=0.90,
        h_crit=0.08,
        eps_s=0.90,
        xi=1.10,
    )
    stress_2_params = replace(
        baseline_params,
        c0=0.039,
        zeta=0.42,
        eta=0.72190,
        lam_c=1.08766,
        sigma_noise=0.50,
        sigma_h=1.20,
        h_crit=0.02,
        eps_s=1.10,
        xi=1.49920,
        nu=0.38378,
        mu_k=0.30200,
        lam=0.35470,
        c_s=0.62194,
        kappa=1.15066,
        u_max=0.78189,
    )

    scenarios: dict[str, MasterParams] = {
        "baseline": baseline_params,
        "stress_1": stress_1_params,
        "stress_2": stress_2_params,
    }

    det_runs: dict[str, dict[str, np.ndarray]] = {}
    sde_runs: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}

    for name, params in scenarios.items():
        det = simulate_master_deterministic(p=params)
        sde = simulate_memory_sde(p=params, seed=42)

        det_runs[name] = det
        sde_runs[name] = sde
        metrics[name] = _compute_metrics(det, sde)

        _save_series_csv(run_dir / f"{name}_deterministic.csv", det)
        _save_series_csv(run_dir / f"{name}_stochastic.csv", sde)

        with (run_dir / f"{name}_params_used.txt").open("w", encoding="utf-8") as fh:
            for key, value in asdict(params).items():
                fh.write(f"{key}={value}\n")

    _save_metrics_csv(run_dir / "comparison_metrics.csv", metrics)
    _plot_comparison(run_dir / "comparison_plot.png", det_runs, sde_runs)

    latest_dir = out_root / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    print(f"Saved results to: {run_dir}")
    print(f"Latest snapshot: {latest_dir}")
    _print_comparison_table(metrics)


if __name__ == "__main__":
    main()

