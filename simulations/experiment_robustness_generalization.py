from __future__ import annotations

from pathlib import Path
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from simulations.simulation_network_v1 import MasterParams, NetworkParams, simulate_network


def lesson_metrics(out: dict[str, np.ndarray], stress_start: float, stress_duration: float, threshold: float = 0.2) -> dict[str, float]:
    t = out["t"]
    mean_s = out["mean_s"]
    s = out["S"]
    stress_end = stress_start + stress_duration
    start_idx = int(np.searchsorted(t, stress_end, side="left"))

    recovery_time = float("nan")
    for i in range(start_idx, len(t)):
        if mean_s[i] <= threshold:
            recovery_time = float(t[i] - stress_end)
            break

    per_agent = []
    for j in range(s.shape[1]):
        rt = float("nan")
        for i in range(start_idx, len(t)):
            if s[i, j] <= threshold:
                rt = float(t[i] - stress_end)
                break
        if not np.isnan(rt):
            per_agent.append(rt)
    per = np.array(per_agent, dtype=float)

    stress_mask = (t >= stress_start) & (t <= stress_end)
    return {
        "recovery_time": recovery_time,
        "tail_span": float(np.max(per) - np.min(per)) if per.size else float("nan"),
        "peak_mean_error": float(np.max(out["mean_error"])),
        "peak_mean_turbulence": float(np.max(out["mean_turbulence"])),
        "peak_cascade_fraction": float(np.max(out["cascade_fraction"])),
        "final_attunement": float(out["mean_attunement"][-1]),
        "final_group_memory": float(out["group_memory"][-1]),
        "final_demper_load": float(out["mean_demper_load"][-1]),
        "stress_channel_instructor": float(np.mean(out["mean_channel_instructor"][stress_mask])),
        "stress_channel_anchor": float(np.mean(out["mean_channel_anchor"][stress_mask])),
    }


def block_mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def curriculum_lessons() -> list[dict[str, object]]:
    lesson_cycle = [
        {"name": "warm_noise", "lesson_type": "noise", "stress_amp": 2.8, "stress_frac": 0.24, "stress_y_amp": 0.90, "sigma_scale": 1.20, "delay_steps": (0, 3, 7), "contagion_weight": 0.75},
        {"name": "warm_lag", "lesson_type": "lag", "stress_amp": 3.0, "stress_frac": 0.26, "stress_y_amp": 1.00, "sigma_scale": 1.00, "delay_steps": (0, 5, 11), "contagion_weight": 0.75},
        {"name": "warm_contagion", "lesson_type": "contagion", "stress_amp": 3.2, "stress_frac": 0.34, "stress_y_amp": 1.15, "sigma_scale": 1.10, "delay_steps": (0, 3, 7), "contagion_weight": 1.20},
        {"name": "warm_mixed", "lesson_type": "contagion", "stress_amp": 3.1, "stress_frac": 0.30, "stress_y_amp": 1.05, "sigma_scale": 1.05, "delay_steps": (0, 4, 9), "contagion_weight": 1.00},
    ]
    lessons = []
    for block in range(2):
        for spec in lesson_cycle:
            lessons.append({"name": f"block_{block + 1}_{spec['name']}", **spec})
    return lessons


def evaluation_scenarios() -> list[dict[str, object]]:
    return [
        {
            "scenario": "baseline_mix",
            "lesson_type": "contagion",
            "stress_amp": 3.1,
            "stress_frac": 0.30,
            "stress_y_amp": 1.05,
            "sigma_scale": 1.05,
            "delay_steps": (0, 4, 9),
            "contagion_weight": 1.00,
        },
        {
            "scenario": "shifted_lag_transfer",
            "lesson_type": "lag",
            "stress_amp": 3.35,
            "stress_frac": 0.31,
            "stress_y_amp": 1.10,
            "sigma_scale": 1.08,
            "delay_steps": (1, 7, 14),
            "contagion_weight": 0.82,
        },
        {
            "scenario": "shifted_contagion_transfer",
            "lesson_type": "contagion",
            "stress_amp": 3.45,
            "stress_frac": 0.37,
            "stress_y_amp": 1.22,
            "sigma_scale": 1.15,
            "delay_steps": (0, 5, 11),
            "contagion_weight": 1.35,
        },
    ]


def base_common_params(n_agents: int) -> dict[str, object]:
    return dict(
        n_agents=n_agents,
        t_end=18.0,
        dt=0.02,
        topology="spatial_local",
        spatial_radius=0.34,
        spatial_falloff=0.16,
        coupling=0.17,
        delay_mode="grouped",
        delay_group_fracs=(0.34, 0.33, 0.33),
        stress_time=7.0,
        stress_duration=2.5,
        qrc_enabled=True,
        phi_kappa=1.0,
        phi_gain=0.42,
        phi_gain_boost=8.0,
        ccrit_gain=0.6,
        recog_threshold=0.7,
        recog_gain=1.2,
        wake_time_required=4.0,
        wake_relax_gain=0.8,
        coh_relax_gain=0.6,
        instructor_enabled=True,
        instructor_error_weight=0.75,
        instructor_anchor_weight=0.85,
        instructor_template_weight=0.70,
        attunement_enabled=True,
        attunement_gain=0.08,
        attunement_decay=0.010,
        kg_enabled=True,
        kg_lambda=0.08,
        kg_decay=0.02,
        kg_crisis_threshold=0.45,
        kg_phi_boost=1.6,
        kg_wake_boost=1.2,
        three_channel_enabled=True,
        three_channel_adaptive=True,
        channel_peer_base=0.58,
        channel_instructor_base=0.25,
        channel_anchor_base=0.17,
        channel_turbulence_gain=0.80,
        channel_attunement_gain=0.55,
        channel_anchor_gain=0.55,
        template_multimode_enabled=True,
        template_focus_gain=1.35,
        template_recovery_gain=0.82,
        template_stabilize_gain=0.92,
        template_turbulence_trigger=0.20,
        template_recovery_trigger=0.10,
        demper_enabled=True,
        demper_load_gain=0.95,
        demper_decay=0.30,
        demper_relax_gain=0.90,
        demper_trigger=0.16,
    )


def make_base_master_params() -> MasterParams:
    alpha = np.concatenate([np.full(12, 0.78), np.full(12, 0.86), np.full(12, 0.95)])
    sigma_noise = np.concatenate([np.full(12, 0.30), np.full(12, 0.22), np.full(12, 0.14)])
    h_crit = np.concatenate([np.full(12, 0.62), np.full(12, 0.69), np.full(12, 0.77)])
    return MasterParams(alpha=alpha, sigma_noise=sigma_noise, h_crit=h_crit)


def initial_attunement_state() -> np.ndarray:
    return np.concatenate([np.full(12, 0.02), np.full(12, 0.06), np.full(12, 0.10)])


def apply_lesson(master: MasterParams, base_common: dict[str, object], lesson: dict[str, object], kg_memory: float, phi_init: float, attunement: np.ndarray, overrides: dict[str, object] | None = None) -> tuple[dict[str, np.ndarray], NetworkParams]:
    p_lesson = MasterParams(
        **{
            **master.__dict__,
            "sigma_noise": master.sigma_noise * float(lesson["sigma_scale"]),
        }
    )
    net_kwargs = {
        **base_common,
        "delay_group_steps": lesson["delay_steps"],
        "stress_amp": lesson["stress_amp"],
        "stress_frac": lesson["stress_frac"],
        "stress_y_amp": lesson["stress_y_amp"],
        "instructor_response_mode": lesson["lesson_type"],
        "instructor_contagion_weight": lesson["contagion_weight"],
        "initial_group_memory": kg_memory,
        "initial_phi": phi_init,
        "initial_attunement": attunement,
    }
    if overrides:
        net_kwargs.update(overrides)
    net = NetworkParams(**net_kwargs)
    out = simulate_network(p_lesson, net)
    return out, net


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    master = make_base_master_params()
    base_common = base_common_params(n_agents=36)
    warmup_lessons = curriculum_lessons()
    scenarios = evaluation_scenarios()
    robustness_grid = [
        {"robustness_case": "nominal", "demper_load_gain": 0.95, "demper_relax_gain": 0.90, "attunement_gain": 0.08},
        {"robustness_case": "weaker_release", "demper_load_gain": 0.78, "demper_relax_gain": 0.72, "attunement_gain": 0.08},
        {"robustness_case": "stronger_release", "demper_load_gain": 1.08, "demper_relax_gain": 1.00, "attunement_gain": 0.08},
        {"robustness_case": "slower_attunement", "demper_load_gain": 0.95, "demper_relax_gain": 0.90, "attunement_gain": 0.05},
        {"robustness_case": "faster_attunement", "demper_load_gain": 0.95, "demper_relax_gain": 0.90, "attunement_gain": 0.11},
    ]

    warmup_rows: list[dict[str, float | str | int]] = []
    scenario_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for case in robustness_grid:
        kg_memory = 0.0
        phi_init = 0.0
        attune_state = initial_attunement_state()
        case_overrides = {
            "demper_load_gain": case["demper_load_gain"],
            "demper_relax_gain": case["demper_relax_gain"],
            "attunement_gain": case["attunement_gain"],
        }

        case_warmup: list[dict[str, float]] = []
        for lesson_idx, lesson in enumerate(warmup_lessons, 1):
            out, net = apply_lesson(master, base_common, lesson, kg_memory, phi_init, attune_state, case_overrides)
            metrics = lesson_metrics(out, net.stress_time, net.stress_duration)
            row = {
                "phase": "warmup",
                "robustness_case": case["robustness_case"],
                "lesson": lesson_idx,
                "scenario": str(lesson["name"]),
                **metrics,
            }
            warmup_rows.append(row)
            case_warmup.append(metrics)
            kg_memory = float(out["group_memory"][-1])
            phi_init = float(out["phi"][-1])
            attune_state = out["attunement"][-1].copy()

        for scenario in scenarios:
            out, net = apply_lesson(master, base_common, scenario, kg_memory, phi_init, attune_state, case_overrides)
            metrics = lesson_metrics(out, net.stress_time, net.stress_duration)
            row = {
                "phase": "evaluation",
                "robustness_case": case["robustness_case"],
                "scenario": str(scenario["scenario"]),
                **metrics,
            }
            scenario_rows.append(row)
            print(
                f"{case['robustness_case']} {scenario['scenario']}: "
                f"rt={metrics['recovery_time']:.3f}, tail={metrics['tail_span']:.3f}, "
                f"err={metrics['peak_mean_error']:.3f}, turb={metrics['peak_mean_turbulence']:.3f}, "
                f"att={metrics['final_attunement']:.3f}"
            )

        summary_rows.append(
            {
                "robustness_case": case["robustness_case"],
                "warmup_mean_recovery_time": block_mean(case_warmup, "recovery_time"),
                "warmup_mean_tail_span": block_mean(case_warmup, "tail_span"),
                "warmup_mean_peak_error": block_mean(case_warmup, "peak_mean_error"),
                "warmup_mean_peak_turbulence": block_mean(case_warmup, "peak_mean_turbulence"),
                "eval_mean_recovery_time": block_mean(
                    [row for row in scenario_rows if row["robustness_case"] == case["robustness_case"]], "recovery_time"
                ),
                "eval_mean_tail_span": block_mean(
                    [row for row in scenario_rows if row["robustness_case"] == case["robustness_case"]], "tail_span"
                ),
                "eval_mean_peak_error": block_mean(
                    [row for row in scenario_rows if row["robustness_case"] == case["robustness_case"]], "peak_mean_error"
                ),
                "eval_mean_peak_turbulence": block_mean(
                    [row for row in scenario_rows if row["robustness_case"] == case["robustness_case"]], "peak_mean_turbulence"
                ),
                "eval_mean_final_attunement": block_mean(
                    [row for row in scenario_rows if row["robustness_case"] == case["robustness_case"]], "final_attunement"
                ),
            }
        )

    warmup_csv = out_dir / "robustness_generalization_warmup.csv"
    with warmup_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(warmup_rows[0].keys()))
        writer.writeheader()
        writer.writerows(warmup_rows)

    eval_csv = out_dir / "robustness_generalization_eval.csv"
    with eval_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)

    summary_csv = out_dir / "robustness_generalization_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    x = np.arange(len(robustness_grid))
    labels = [str(case["robustness_case"]) for case in robustness_grid]
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    axes[0].plot(x, [row["eval_mean_recovery_time"] for row in summary_rows], marker="o")
    axes[1].plot(x, [row["eval_mean_tail_span"] for row in summary_rows], marker="o")
    axes[2].plot(x, [row["eval_mean_peak_error"] for row in summary_rows], marker="o", label="peak error")
    axes[2].plot(x, [row["eval_mean_peak_turbulence"] for row in summary_rows], marker="x", linestyle="--", label="peak turbulence")
    axes[3].plot(x, [row["eval_mean_final_attunement"] for row in summary_rows], marker="o")

    axes[0].set_ylabel("recovery")
    axes[0].set_title("Generalization recovery across robustness cases")
    axes[1].set_ylabel("tail")
    axes[1].set_title("Generalization tail across robustness cases")
    axes[2].set_ylabel("field")
    axes[2].set_title("Generalization field quality across robustness cases")
    axes[3].set_ylabel("attunement")
    axes[3].set_title("Final attunement after shifted lessons")
    axes[3].set_xlabel("robustness case")
    axes[3].set_xticks(x, labels, rotation=15)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    plot_path = out_dir / "robustness_generalization_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {warmup_csv}")
    print(f"Wrote {eval_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
