"""
Phase map experiment for Langevin noise on a(t).
Scans sigma_a and eta, computes crisis metrics, and saves CSV + heatmap.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from simulation_noise import NoiseParams, simulate_master_langevin


def _compute_metrics(det: dict[str, np.ndarray]) -> dict[str, float]:
    c = det["C"]
    c_crit = det["Ccrit"]
    crisis_mask = c > c_crit
    metrics = {
        "crisis_share": float(np.mean(crisis_mask)),
        "crisis_mean": float(np.mean(c)),
        "crisis_peak_margin": float(np.max(c - c_crit)),
        "I_final": float(det["I"][-1]),
        "g_min": float(np.min(det["g"])),
        "g_max": float(np.max(det["g"])),
    }
    return metrics


def _save_metrics_csv(path: Path, rows: list[dict[str, float]], header: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(f"{row.get(k, np.nan):.8f}" for k in header) + "\n")


def _plot_heatmap(out_file: Path, matrix: np.ndarray, x_vals: list[float], y_vals: list[float], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(matrix, origin="lower", aspect="auto")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in x_vals], rotation=45)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in y_vals])

    ax.set_xlabel("sigma_a")
    ax.set_ylabel("eta")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("value")

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
    run_tag = "noiseGrid_refine"
    run_dir = out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    t_end = 120.0
    dt = 0.01
    seed = 42

    sigma_a_list = [0.45, 0.50, 0.55, 0.60, 0.65]
    eta_list = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    rows: list[dict[str, float]] = []

    crisis_share_map = np.zeros((len(eta_list), len(sigma_a_list)))
    crisis_peak_map = np.zeros((len(eta_list), len(sigma_a_list)))
    i_final_map = np.zeros((len(eta_list), len(sigma_a_list)))

    for i_eta, eta in enumerate(eta_list):
        for j_sig, sigma_a in enumerate(sigma_a_list):
            params = replace(NoiseParams(), eta=eta, sigma_a=sigma_a)
            det = simulate_master_langevin(p=params, t_end=t_end, dt=dt, seed=seed)
            metrics = _compute_metrics(det)

            row = {
                "eta": eta,
                "sigma_a": sigma_a,
                **metrics,
            }
            rows.append(row)

            crisis_share_map[i_eta, j_sig] = metrics["crisis_share"]
            crisis_peak_map[i_eta, j_sig] = metrics["crisis_peak_margin"]
            i_final_map[i_eta, j_sig] = metrics["I_final"]

            params_path = run_dir / f"params_eta{eta:.2f}_sigma{sigma_a:.2f}.txt"
            with params_path.open("w", encoding="utf-8") as fh:
                for key, value in asdict(params).items():
                    fh.write(f"{key}={value}\n")

    header = ["eta", "sigma_a", "crisis_share", "crisis_mean", "crisis_peak_margin", "I_final", "g_min", "g_max"]
    _save_metrics_csv(run_dir / "phase_map_metrics.csv", rows, header)

    notes_path = run_dir / "RUN_NOTES.txt"
    with notes_path.open("w", encoding="utf-8") as fh:
        fh.write("Thermal noise phase map (Langevin on a)\n")
        fh.write(f"run_tag={run_tag}\n")
        fh.write(f"t_end={t_end}\n")
        fh.write(f"dt={dt}\n")
        fh.write(f"seed={seed}\n")
        fh.write(f"sigma_a_list={sigma_a_list}\n")
        fh.write(f"eta_list={eta_list}\n")

    readme_path = Path(__file__).with_name("README.md")
    _update_experiment_log(
        readme_path,
        run_id=run_dir.name,
        purpose=f"Phase map ({run_tag})",
        sigma_desc=str(sigma_a_list),
        eta_desc=str(eta_list),
    )


    _plot_heatmap(run_dir / "phase_map_crisis_share.png", crisis_share_map, sigma_a_list, eta_list, "Crisis share")
    _plot_heatmap(run_dir / "phase_map_crisis_peak.png", crisis_peak_map, sigma_a_list, eta_list, "Crisis peak margin")
    _plot_heatmap(run_dir / "phase_map_I_final.png", i_final_map, sigma_a_list, eta_list, "Final isolation I")

    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()




