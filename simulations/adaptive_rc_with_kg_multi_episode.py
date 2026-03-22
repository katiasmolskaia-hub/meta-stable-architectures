from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

from simulations.adaptive_rc_hetero_demo import HeteroDemoParams, recovery_times_per_agent
from simulations.adaptive_rc_skeleton import clamp, normalize_weights


DEFAULT_WINDOWS = [
    (20.0, 6.0),
    (55.0, 8.0),
    (95.0, 10.0),
    (140.0, 12.0),
]


def observe(theta: np.ndarray, s: np.ndarray, prev_s: np.ndarray) -> dict[str, np.ndarray | float]:
    mean_vec = np.mean(np.exp(1j * theta))
    dispersion = float(1.0 - abs(mean_vec))
    lag = float(np.mean(s))
    contagion = float(np.mean(prev_s))
    anchors = 1.0 - s
    return {"D": dispersion, "L": lag, "C": contagion, "A": anchors}


def episode_summary(out: dict[str, np.ndarray], stress_start: float, stress_duration: float) -> dict[str, float]:
    stress_end = stress_start + stress_duration
    t = out["t"]
    mean_s = np.mean(out["S"], axis=1)
    D = out["D"]

    idx_stress = (t >= stress_start) & (t <= stress_end)
    peak_s = float(np.max(mean_s[idx_stress])) if np.any(idx_stress) else float(np.max(mean_s))
    peak_d = float(np.max(D[idx_stress])) if np.any(idx_stress) else float(np.max(D))

    recovery_time = float("nan")
    for i, tt in enumerate(t):
        if tt > stress_end and mean_s[i] <= 0.2:
            recovery_time = float(tt - stress_start)
            break

    per_agent = recovery_times_per_agent(out["S"], t, stress_end)
    ok = per_agent[~np.isnan(per_agent)]
    tail_span = float(np.max(ok) - np.min(ok)) if ok.size else float("nan")
    return {
        "peak_s": peak_s,
        "peak_d": peak_d,
        "recovery_time": recovery_time,
        "tail_span": tail_span,
        "agent_recovery_mean": float(np.mean(ok)) if ok.size else float("nan"),
        "agent_recovery_std": float(np.std(ok)) if ok.size else float("nan"),
        "agent_recovery_min": float(np.min(ok)) if ok.size else float("nan"),
        "agent_recovery_max": float(np.max(ok)) if ok.size else float("nan"),
    }


def crisis_score(peak_s: float, recovery_time: float, tail_span: float, peak_d: float) -> float:
    # Bounded score in [0, 1]-like range for episode significance.
    s = np.clip(peak_s / 1.0, 0.0, 1.0)
    r = np.clip(recovery_time / 20.0, 0.0, 1.0)
    t = np.clip(tail_span / 8.0, 0.0, 1.0)
    d = np.clip(peak_d / 1.0, 0.0, 1.0)
    return float(0.35 * s + 0.30 * r + 0.20 * t + 0.15 * d)


def run_with_kg(
    params: HeteroDemoParams,
    seed: int,
    use_kg: bool = True,
    *,
    windows: list[tuple[float, float]] | None = None,
    kg_strength: tuple[float, float, float] = (0.35, 0.25, 0.15),
    kg_threshold: float = 0.48,
    lambda_g: float = 0.15,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = params.n_agents
    steps = int(params.t_end / params.dt) + 1
    t = np.linspace(0.0, params.t_end, steps)

    x = rng.normal(0.0, 1.0, size=n)
    y = rng.normal(0.0, 1.0, size=n)
    s = np.clip(rng.normal(0.15, 0.05, size=n), 0.0, 1.0)
    k = np.zeros(n)
    Phi = 0.0
    prev_s = s.copy()

    stress_sensitivity = rng.lognormal(mean=0.0, sigma=params.hetero_sigma, size=n)
    recog_threshold = np.clip(rng.normal(0.70, 0.06, size=n), 0.55, 0.90)
    noise_scale = params.noise_scale * rng.lognormal(mean=0.0, sigma=params.hetero_sigma, size=n)

    X = np.zeros((steps, n))
    Y = np.zeros((steps, n))
    S = np.zeros((steps, n))
    K = np.zeros((steps, n))
    Phi_hist = np.zeros(steps)
    D_hist = np.zeros(steps)
    phi_gain_hist = np.zeros(steps)
    kappa_hist = np.zeros(steps)
    Kg_hist = np.zeros(steps)

    kg = 0.0
    Ccrit = kg_threshold
    a_k, b_k, lambda_g = kg_strength[0], kg_strength[1], lambda_g

    episode_idx = 0
    episode_stats: list[dict[str, float]] = []
    active_windows = windows if windows is not None else DEFAULT_WINDOWS
    current_window = active_windows[episode_idx] if episode_idx < len(active_windows) else None
    stress_start, stress_duration = current_window if current_window is not None else (params.stress_time, params.stress_duration)
    stress_end = stress_start + stress_duration

    for idx, tt in enumerate(t):
        if current_window is not None and tt > stress_end and episode_idx < len(active_windows):
            # Close the just-finished episode before moving to the next window.
            row = episode_summary(
                {
                    "t": t[: idx + 1],
                    "S": S[: idx + 1],
                    "D": D_hist[: idx + 1],
                },
                stress_start,
                stress_duration,
            )
            score = crisis_score(row["peak_s"], row["recovery_time"], row["tail_span"], row["peak_d"])
            gate = 1.0 if score > Ccrit else 0.0
            if use_kg:
                kg = (1.0 - lambda_g) * kg + lambda_g * (score * gate)
            episode_stats.append({
                "episode": float(episode_idx + 1),
                "score": score,
                "gate": gate,
                "kg": kg,
                **row,
            })
            episode_idx += 1
            current_window = active_windows[episode_idx] if episode_idx < len(active_windows) else None
            if current_window is not None:
                stress_start, stress_duration = current_window
                stress_end = stress_start + stress_duration

        X[idx] = x
        Y[idx] = y
        S[idx] = s
        K[idx] = k
        Phi_hist[idx] = Phi
        Kg_hist[idx] = kg

        theta = np.arctan2(y, x)
        obs = observe(theta, s, prev_s)
        D = float(obs["D"])
        L = float(obs["L"])
        C = float(obs["C"])
        A = np.asarray(obs["A"], dtype=float)
        D_hist[idx] = D

        # K_g gently shifts the adaptive gains; it never takes over control.
        phi_gain = clamp(
            params.phi0 * (1.0 + params.a1 * D + params.a2 * L + params.a3 * C + a_k * kg),
            params.phi_min,
            params.phi_max,
        )
        kappa = clamp(
            params.kappa0 * (1.0 + params.b1 * D + params.b2 * L + b_k * kg),
            params.kappa_min,
            params.kappa_max,
        )
        phi_gain_hist[idx] = phi_gain
        kappa_hist[idx] = kappa

        if idx == steps - 1:
            break

        mean_phase = float(np.angle(np.mean(np.exp(1j * theta))))
        Phi = Phi + params.dt * (-kappa * Phi + mean_phase)

        w = normalize_weights(A)
        neighbor_signal = np.dot(w, y)
        anti_contagion = (1.0 - s) * (1.0 - np.mean(s))

        dx = -0.8 * x - 0.25 * y
        dy = 0.9 * x - 0.85 * (y**3)
        coupling_term = params.coupling * anti_contagion * (neighbor_signal - y)
        coordinator_term = phi_gain * (Phi - y) * (1.0 - s)

        x = x + params.dt * dx
        y = y + params.dt * (dy + coupling_term + coordinator_term)
        y = y + np.sqrt(params.dt) * noise_scale * rng.normal(size=n)

        in_stress = current_window is not None and (stress_start <= tt < stress_end)
        if in_stress:
            s = np.clip(s + 0.08 * params.stress_amp * stress_sensitivity, 0.0, 1.0)
            y = y + 0.06 * params.stress_y_amp * stress_sensitivity

        match = np.cos(theta - Phi)
        recog = np.maximum(0.0, match - recog_threshold)
        dS = params.base_suppression * (0.4 - D) * s * (1.0 - s)
        dS -= params.recog_gain * recog * s * (1.0 - s)
        dS -= params.coh_gain * (1.0 - D) * s
        dS = dS * stress_sensitivity
        s = np.clip(s + params.dt * dS, 0.0, 1.0)

        k = np.maximum(0.0, k + params.dt * (0.25 * y**2 - 0.3 * k + 0.2 * s))
        prev_s = s.copy()

    # Close the last episode if needed.
    while episode_idx < len(active_windows):
        start, duration = active_windows[episode_idx]
        row = episode_summary(
            {
                "t": t,
                "S": S,
                "D": D_hist,
            },
            start,
            duration,
        )
        score = crisis_score(row["peak_s"], row["recovery_time"], row["tail_span"], row["peak_d"])
        gate = 1.0 if score > Ccrit else 0.0
        if use_kg:
            kg = (1.0 - lambda_g) * kg + lambda_g * (score * gate)
        episode_stats.append({
            "episode": float(episode_idx + 1),
            "score": score,
            "gate": gate,
            "kg": kg,
            **row,
        })
        episode_idx += 1

    return {
        "t": t,
        "X": X,
        "Y": Y,
        "S": S,
        "K": K,
        "Phi": Phi_hist,
        "D": D_hist,
        "phi_gain": phi_gain_hist,
        "kappa": kappa_hist,
        "Kg": Kg_hist,
        "episode_stats": episode_stats,  # type: ignore[dict-item]
    }


def main() -> None:
    out_dir = Path(ROOT) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = HeteroDemoParams(
        t_end=200.0,
        stress_time=20.0,
        stress_duration=6.0,
        stress_amp=2.7,
        stress_y_amp=0.9,
        hetero_sigma=0.35,
        noise_scale=0.04,
        phi0=0.42,
        a1=0.55,
        a2=0.35,
        a3=0.25,
        kappa0=0.90,
        coh_gain=0.45,
        recog_gain=1.00,
    )

    kg_run = run_with_kg(params, seed=11, use_kg=True)
    strong_kg_run = run_with_kg(params, seed=11, use_kg=True, kg_strength=(0.50, 0.40, 0.20), kg_threshold=0.48, lambda_g=0.20)
    base_run = run_with_kg(replace(params), seed=11, use_kg=False)

    def summarize(run: dict[str, np.ndarray], label: str) -> dict[str, float | str]:
        rows = run["episode_stats"]  # type: ignore[index]
        return {
            "label": label,
            "final_mean_s": float(np.mean(run["S"][-1])),
            "final_dispersion": float(run["D"][-1]),
            "final_mean_k": float(np.mean(run["K"][-1])),
            "final_kg": float(run["Kg"][-1]),
            "episode1_score": float(rows[0]["score"]),
            "episode2_score": float(rows[1]["score"]),
            "episode3_score": float(rows[2]["score"]),
            "episode4_score": float(rows[3]["score"]),
            "episode1_gate": float(rows[0]["gate"]),
            "episode2_gate": float(rows[1]["gate"]),
            "episode3_gate": float(rows[2]["gate"]),
            "episode4_gate": float(rows[3]["gate"]),
        }

    summary_rows = [
        summarize(base_run, "spread_friendly_no_kg"),
        summarize(kg_run, "spread_friendly_with_kg"),
        summarize(strong_kg_run, "spread_friendly_stronger_kg"),
    ]

    csv_path = out_dir / "adaptive_rc_with_kg_multi_episode.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            f"{row['label']}: final_mean_s={row['final_mean_s']:.4f}, final_dispersion={row['final_dispersion']:.4f}, "
            f"final_mean_k={row['final_mean_k']:.4f}, final_kg={row['final_kg']:.4f}"
        )
        print(
            f"  episode scores: {row['episode1_score']:.3f}, {row['episode2_score']:.3f}, "
            f"{row['episode3_score']:.3f}, {row['episode4_score']:.3f}"
        )

    print(f"Wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(kg_run["t"], np.mean(kg_run["S"], axis=1), label="with kg")
        axes[0, 0].plot(strong_kg_run["t"], np.mean(strong_kg_run["S"], axis=1), label="strong kg")
        axes[0, 0].plot(base_run["t"], np.mean(base_run["S"], axis=1), label="no kg", linestyle="--")
        axes[0, 0].set_title("Mean suppression")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(kg_run["t"], kg_run["Kg"], label="K_g")
        axes[0, 1].plot(strong_kg_run["t"], strong_kg_run["Kg"], label="strong K_g", linestyle="--")
        axes[0, 1].set_title("Group memory")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(kg_run["t"], kg_run["phi_gain"], label="with kg")
        axes[1, 0].plot(strong_kg_run["t"], strong_kg_run["phi_gain"], label="strong kg")
        axes[1, 0].plot(base_run["t"], base_run["phi_gain"], label="no kg", linestyle="--")
        axes[1, 0].set_title("Phi gain")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(kg_run["t"], kg_run["kappa"], label="with kg")
        axes[1, 1].plot(strong_kg_run["t"], strong_kg_run["kappa"], label="strong kg")
        axes[1, 1].plot(base_run["t"], base_run["kappa"], label="no kg", linestyle="--")
        axes[1, 1].set_title("Kappa")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "adaptive_rc_with_kg_multi_episode_plot.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
