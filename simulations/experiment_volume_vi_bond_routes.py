from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


@dataclass
class BondRouteParams:
    n_agents: int = 14
    t_end: float = 24.0
    dt: float = 0.05
    coupling_radius: int = 2
    seed: int = 7
    n_groups: int = 2

    alpha_s: float = 0.28
    alpha_f: float = 0.24
    alpha_g: float = 0.28
    alpha_d: float = 0.44
    alpha_r: float = 0.18

    beta_open: float = 0.50
    beta_sf: float = 0.22
    beta_df: float = 0.30

    gamma_b: float = 0.34
    gamma_g: float = 0.26
    gamma_f2: float = 0.22
    gamma_d: float = 0.34

    delta_r: float = 0.72
    delta_c: float = 0.25
    delta_s: float = 0.30
    delta_b: float = 0.22

    lambda_s: float = 0.70
    lambda_f: float = 0.48
    lambda_d: float = 0.95

    route_center: float = 10.0
    route_width: float = 3.0
    intervention_enabled: bool = False
    relapse_enabled: bool = True
    relapse_center_offset: float = 4.0
    relapse_width: float = 1.6
    relapse_gain: float = 0.18


def build_ring_edges(n_agents: int, radius: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(n_agents):
        for d in range(1, radius + 1):
            j = (i + d) % n_agents
            edges.append((i, j) if i < j else (j, i))
    return sorted(set(edges))


def assign_groups(n_agents: int, n_groups: int) -> np.ndarray:
    group_size = n_agents // n_groups
    groups = np.zeros(n_agents, dtype=int)
    for i in range(n_agents):
        groups[i] = min(i // group_size, n_groups - 1)
    return groups


def build_group_structured_edges(n_agents: int, radius: int, groups: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(n_agents):
        same_group = [j for j in range(n_agents) if groups[j] == groups[i] and j != i]
        same_group_sorted = sorted(same_group, key=lambda j: min((j - i) % n_agents, (i - j) % n_agents))
        for j in same_group_sorted[: radius + 1]:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    # Add a small number of explicit inter-group bridges.
    group_ids = sorted(set(int(g) for g in groups))
    for g in group_ids:
        h = (g + 1) % len(group_ids)
        g_nodes = np.where(groups == g)[0]
        h_nodes = np.where(groups == h)[0]
        a = int(g_nodes[-1])
        b = int(h_nodes[0])
        edges.add((a, b) if a < b else (b, a))
        if len(g_nodes) > 1 and len(h_nodes) > 1:
            a2 = int(g_nodes[len(g_nodes) // 2])
            b2 = int(h_nodes[len(h_nodes) // 2])
            edges.add((a2, b2) if a2 < b2 else (b2, a2))
    return sorted(edges)


def assign_bond_classes(edges: list[tuple[int, int]], groups: np.ndarray) -> np.ndarray:
    classes = []
    for i, j in edges:
        if groups[i] != groups[j]:
            classes.append("bridge")
            continue
        span = abs(j - i)
        if span <= 2:
            classes.append("local")
        else:
            classes.append("aux")
    return np.array(classes)


def class_mean(values: np.ndarray, classes: np.ndarray, target: str) -> float:
    mask = classes == target
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def route_profile(t: np.ndarray, route: str, center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    base = np.exp(-0.5 * ((t - center) / width) ** 2)
    base /= np.max(base)

    if route == "drift":
        route_pressure = 0.65 * base + 0.25 * np.clip((t - 4.0) / (center + width - 4.0), 0.0, 1.0)
        contagion = 0.25 + 0.20 * base
    elif route == "shock":
        route_pressure = 1.05 * np.exp(-0.5 * ((t - center) / (0.45 * width)) ** 2)
        route_pressure /= np.max(route_pressure)
        contagion = 0.20 + 0.10 * base
    elif route == "distortion":
        route_pressure = 0.80 * base + 0.15
        route_pressure = np.clip(route_pressure, 0.0, 1.0)
        contagion = 0.55 + 0.25 * base
    else:
        raise ValueError(f"unknown route: {route}")
    return route_pressure, np.clip(contagion, 0.0, 1.0)


def classify_states(
    viability: np.ndarray,
    freedom: np.ndarray,
    stability: np.ndarray,
    distortion: np.ndarray,
    reentry: np.ndarray,
    crisis_active: bool,
) -> dict[str, int]:
    counts = {
        "latent_weak": 0,
        "sustained_weak": 0,
        "functionally_empty": 0,
        "restored_weak": 0,
        "stabilizing": 0,
        "crisis_support": 0,
        "distorted": 0,
        "rigid": 0,
    }

    for b, f, s, d, e in zip(viability, freedom, stability, distortion, reentry):
        if d > 0.62 and e < 0.08:
            counts["distorted"] += 1
        elif s > 0.72 and f < 0.18:
            counts["rigid"] += 1
        elif b < 0.22 and e < 0.10:
            counts["functionally_empty"] += 1
        elif b > 0.68 and s > 0.62 and f > 0.50 and e > 0.30:
            if crisis_active and s > 0.78:
                counts["crisis_support"] += 1
            else:
                counts["stabilizing"] += 1
        elif b > 0.42 and d < 0.35 and f > 0.18 and s > 0.28 and e > 0.12:
            counts["restored_weak"] += 1 if b > 0.56 else 0
            counts["sustained_weak"] += 1 if b <= 0.56 else 0
        else:
            counts["latent_weak"] += 1
    return counts


def dominant_state_name(
    viability: float,
    freedom: float,
    stability: float,
    distortion: float,
    reentry: float,
    crisis_active: bool,
) -> str:
    if distortion > 0.62 and reentry < 0.08:
        return "distorted"
    if stability > 0.72 and freedom < 0.18:
        return "rigid"
    if viability < 0.22 and reentry < 0.10:
        return "functionally_empty"
    if viability > 0.68 and stability > 0.62 and freedom > 0.50 and reentry > 0.30:
        return "crisis_support" if crisis_active and stability > 0.78 else "stabilizing"
    if viability > 0.42 and distortion < 0.35 and freedom > 0.18 and stability > 0.28 and reentry > 0.12:
        return "restored_weak" if viability > 0.56 else "sustained_weak"
    return "latent_weak"


def simulate_route(route: str, params: BondRouteParams) -> dict[str, np.ndarray | list[dict[str, float]]]:
    rng = np.random.default_rng(params.seed)
    groups = assign_groups(params.n_agents, params.n_groups)
    edges = build_group_structured_edges(params.n_agents, params.coupling_radius, groups)
    n_edges = len(edges)
    bond_classes = assign_bond_classes(edges, groups)
    t = np.arange(0.0, params.t_end + params.dt, params.dt)

    route_pressure, contagion = route_profile(t, route, params.route_center, params.route_width)

    class_viability_bias = np.where(bond_classes == "local", 0.05, np.where(bond_classes == "bridge", -0.03, 0.0))
    class_freedom_bias = np.where(bond_classes == "local", -0.04, np.where(bond_classes == "bridge", 0.07, 0.0))
    class_stability_bias = np.where(bond_classes == "local", 0.08, np.where(bond_classes == "bridge", -0.05, 0.0))
    class_distortion_bias = np.where(bond_classes == "local", -0.02, np.where(bond_classes == "bridge", 0.04, 0.0))
    class_open_bias = np.where(bond_classes == "local", -0.02, np.where(bond_classes == "bridge", 0.06, 0.0))

    viability = np.clip(0.56 + class_viability_bias + 0.04 * rng.normal(size=n_edges), 0.20, 0.95)
    freedom = np.clip(0.48 + class_freedom_bias + 0.05 * rng.normal(size=n_edges), 0.15, 0.95)
    stability = np.clip(0.44 + class_stability_bias + 0.05 * rng.normal(size=n_edges), 0.10, 0.95)
    distortion = np.clip(0.12 + class_distortion_bias + 0.03 * rng.normal(size=n_edges), 0.00, 0.50)
    adaptive_open = np.clip(0.56 + class_open_bias + 0.04 * rng.normal(size=n_edges), 0.15, 1.0)
    shock_support_charge = np.zeros(n_edges, dtype=float)
    drift_erosion = np.zeros(n_edges, dtype=float)

    records: list[dict[str, float]] = []
    dominant_state_sequence: list[tuple[float, str]] = []
    route_factor = {
        "drift": {
            "f_loss": 0.10,
            "s_loss": 0.08,
            "d_gain": 0.00,
            "d_relief": 0.04,
            "b_support": -0.02,
            "late_v_drag": 0.04,
            "late_f_drag": 0.03,
            "late_s_drag": 0.05,
        },
        "shock": {
            "f_loss": 0.04,
            "s_loss": 0.22,
            "d_gain": 0.04,
            "d_relief": 0.08,
            "b_support": 0.12,
            "late_v_drag": 0.00,
            "late_f_drag": 0.00,
            "late_s_drag": 0.00,
        },
        "distortion": {
            "f_loss": 0.08,
            "s_loss": 0.12,
            "d_gain": 0.28,
            "d_relief": 0.00,
            "b_support": -0.08,
            "late_v_drag": 0.04,
            "late_f_drag": 0.02,
            "late_s_drag": 0.06,
        },
    }[route]

    for idx, time in enumerate(t):
        reentry = viability * (
            params.lambda_s * stability + params.lambda_f * freedom - params.lambda_d * distortion
        )
        gatherability = float(np.clip(np.mean(np.maximum(reentry, 0.0)), 0.0, 1.0))
        current_route = route_pressure[idx]
        current_contagion = contagion[idx]
        class_route_mult = np.where(
            bond_classes == "local",
            0.82 if route == "shock" else 0.92,
            np.where(
                bond_classes == "bridge",
                1.20 if route in {"drift", "distortion"} else 1.08,
                1.0,
            ),
        )
        effective_route = np.clip(current_route * class_route_mult, 0.0, 1.4)
        late_phase = 1.0 / (1.0 + np.exp(-(time - (params.route_center + 1.4)) / 1.2))
        relapse_wave = 0.0
        if params.relapse_enabled and route in {"drift", "distortion"}:
            relapse_center = params.route_center + params.relapse_center_offset
            relapse_wave = params.relapse_gain * np.exp(
                -0.5 * ((time - relapse_center) / params.relapse_width) ** 2
            )
        intervention_gain = 1.0 if params.intervention_enabled else 0.0
        i_shock = intervention_gain * current_route if route == "shock" else 0.0
        i_bridge = intervention_gain * current_route if route == "drift" else 0.0
        i_block = intervention_gain * (0.65 + 0.35 * current_route) if route == "distortion" else 0.0
        local_mask = (bond_classes == "local").astype(float)
        bridge_mask = (bond_classes == "bridge").astype(float)
        if route == "shock":
            shock_support_charge = np.clip(
                shock_support_charge
                + params.dt
                * (0.90 * effective_route * (bond_classes != "bridge") - 0.24 * shock_support_charge),
                0.0,
                1.0,
            )
        elif route == "drift":
            drift_erosion = np.clip(
                drift_erosion
                + params.dt
                * (0.18 * effective_route + 0.05 * late_phase - 0.10 * gatherability),
                0.0,
                1.0,
            )
        states = classify_states(
            viability,
            freedom,
            stability,
            distortion,
            reentry,
            crisis_active=bool(np.mean(effective_route) > 0.45),
        )
        dominant_state = dominant_state_name(
            float(np.mean(viability)),
            float(np.mean(freedom)),
            float(np.mean(stability)),
            float(np.mean(distortion)),
            float(np.mean(reentry)),
            crisis_active=bool(np.mean(effective_route) > 0.45),
        )
        if not dominant_state_sequence or dominant_state_sequence[-1][1] != dominant_state:
            dominant_state_sequence.append((float(time), dominant_state))

        records.append(
            {
                "time": float(time),
                "route_pressure": float(route_pressure[idx]),
                "effective_route_pressure": float(np.mean(effective_route)),
                "relapse_wave": float(relapse_wave),
                "contagion": float(contagion[idx]),
                "mean_viability": float(np.mean(viability)),
                "local_mean_viability": class_mean(viability, bond_classes, "local"),
                "bridge_mean_viability": class_mean(viability, bond_classes, "bridge"),
                "mean_freedom": float(np.mean(freedom)),
                "local_mean_freedom": class_mean(freedom, bond_classes, "local"),
                "bridge_mean_freedom": class_mean(freedom, bond_classes, "bridge"),
                "mean_stability": float(np.mean(stability)),
                "local_mean_stability": class_mean(stability, bond_classes, "local"),
                "bridge_mean_stability": class_mean(stability, bond_classes, "bridge"),
                "mean_distortion": float(np.mean(distortion)),
                "local_mean_distortion": class_mean(distortion, bond_classes, "local"),
                "bridge_mean_distortion": class_mean(distortion, bond_classes, "bridge"),
                "mean_reentry": float(np.mean(reentry)),
                "local_mean_reentry": class_mean(reentry, bond_classes, "local"),
                "bridge_mean_reentry": class_mean(reentry, bond_classes, "bridge"),
                "bridge_loss_index": float(
                    np.clip(
                        class_mean(distortion, bond_classes, "bridge")
                        + (1.0 - class_mean(viability, bond_classes, "bridge")),
                        0.0,
                        2.0,
                    )
                ),
                "local_preservation_index": float(
                    np.clip(
                        0.5 * class_mean(viability, bond_classes, "local")
                        + 0.5 * np.maximum(class_mean(reentry, bond_classes, "local"), 0.0),
                        0.0,
                        1.0,
                    )
                ),
                "gatherability": gatherability,
                **{k: float(v) for k, v in states.items()},
            }
        )

        d_viability = (
            params.alpha_s * stability
            + params.alpha_f * freedom
            + params.alpha_g * gatherability
            - params.alpha_d * distortion
            - params.alpha_r * effective_route
            + route_factor["b_support"] * effective_route
            - route_factor["late_v_drag"] * late_phase
            - 0.12 * drift_erosion
            + 0.16 * shock_support_charge
            + 0.18 * i_shock * local_mask
            + 0.22 * i_bridge * bridge_mask
            + 0.10 * i_block * local_mask
            + 0.10 * i_block * bridge_mask * (1.0 - distortion)
            - relapse_wave * (0.20 * bridge_mask + 0.08 * local_mask)
        )
        d_freedom = (
            params.beta_open * (1.0 - freedom) * adaptive_open
            - params.beta_sf * stability * freedom
            - params.beta_df * distortion * freedom
            - route_factor["f_loss"] * effective_route * freedom
            - route_factor["late_f_drag"] * late_phase * freedom
            - 0.05 * drift_erosion * freedom
            + 0.12 * i_bridge * bridge_mask * (1.0 - freedom)
            + 0.06 * i_block * bridge_mask * (1.0 - freedom)
            - relapse_wave * 0.12 * freedom
        )
        d_stability = (
            params.gamma_b * viability
            + params.gamma_g * gatherability
            - params.gamma_f2 * freedom**2
            - params.gamma_d * distortion
            - route_factor["s_loss"] * effective_route
            - route_factor["late_s_drag"] * late_phase
            - 0.08 * drift_erosion
            + 0.26 * shock_support_charge
            + 0.22 * i_shock * local_mask
            + 0.08 * i_bridge * bridge_mask
            + 0.12 * i_block * local_mask
            - relapse_wave * (0.14 * bridge_mask + 0.06 * local_mask)
        )
        d_distortion = (
            params.delta_r * effective_route
            + params.delta_c * current_contagion
            - params.delta_s * stability
            - params.delta_b * viability
            + route_factor["d_gain"] * effective_route
            - route_factor["d_relief"] * gatherability
            + 0.02 * drift_erosion
            - 0.10 * shock_support_charge
            - 0.40 * i_block
            - 0.08 * i_bridge * bridge_mask
            - 0.08 * i_block * local_mask
            + relapse_wave * (0.16 * bridge_mask + 0.06 * local_mask)
        )

        viability = np.clip(viability + params.dt * d_viability, 0.0, 1.0)
        freedom = np.clip(freedom + params.dt * d_freedom, 0.0, 1.0)
        stability = np.clip(stability + params.dt * d_stability, 0.0, 1.0)
        distortion = np.clip(distortion + params.dt * d_distortion, 0.0, 1.0)

    final = records[-1]
    peak_idx = int(np.argmax(route_pressure))
    peak = records[peak_idx]
    trough = min(records, key=lambda row: row["gatherability"])
    return {
        "timeseries": records,
        "summary": np.array(
            [
                final["mean_viability"],
                final["local_mean_viability"],
                final["bridge_mean_viability"],
                final["mean_freedom"],
                final["local_mean_freedom"],
                final["bridge_mean_freedom"],
                final["mean_stability"],
                final["local_mean_stability"],
                final["bridge_mean_stability"],
                final["mean_distortion"],
                final["local_mean_distortion"],
                final["bridge_mean_distortion"],
                final["mean_reentry"],
                final["local_mean_reentry"],
                final["bridge_mean_reentry"],
                final["bridge_loss_index"],
                final["local_preservation_index"],
                final["gatherability"],
                peak["route_pressure"],
                trough["gatherability"],
            ],
            dtype=float,
        ),
        "summary_labels": np.array(
            [
                "final_mean_viability",
                "final_local_mean_viability",
                "final_bridge_mean_viability",
                "final_mean_freedom",
                "final_local_mean_freedom",
                "final_bridge_mean_freedom",
                "final_mean_stability",
                "final_local_mean_stability",
                "final_bridge_mean_stability",
                "final_mean_distortion",
                "final_local_mean_distortion",
                "final_bridge_mean_distortion",
                "final_mean_reentry",
                "final_local_mean_reentry",
                "final_bridge_mean_reentry",
                "final_bridge_loss_index",
                "final_local_preservation_index",
                "final_gatherability",
                "peak_route_pressure",
                "min_gatherability",
            ]
        ),
        "final_states": classify_states(
            viability,
            freedom,
            stability,
            distortion,
            viability * (params.lambda_s * stability + params.lambda_f * freedom - params.lambda_d * distortion),
            crisis_active=False,
        ),
        "peak_states": {
            key: int(value)
            for key, value in classify_states(
                np.full(n_edges, float(records[peak_idx]["mean_viability"])),
                np.full(n_edges, float(records[peak_idx]["mean_freedom"])),
                np.full(n_edges, float(records[peak_idx]["mean_stability"])),
                np.full(n_edges, float(records[peak_idx]["mean_distortion"])),
                np.full(n_edges, float(records[peak_idx]["mean_reentry"])),
                crisis_active=True,
            ).items()
        },
        "t": t,
        "route_pressure": route_pressure,
        "mean_viability": np.array([row["mean_viability"] for row in records], dtype=float),
        "mean_freedom": np.array([row["mean_freedom"] for row in records], dtype=float),
        "mean_stability": np.array([row["mean_stability"] for row in records], dtype=float),
        "mean_distortion": np.array([row["mean_distortion"] for row in records], dtype=float),
        "mean_reentry": np.array([row["mean_reentry"] for row in records], dtype=float),
        "gatherability": np.array([row["gatherability"] for row in records], dtype=float),
        "dominant_state_sequence": dominant_state_sequence,
    }


def write_timeseries_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_state_sequence_csv(path: Path, seq: list[tuple[float, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "dominant_state"])
        writer.writeheader()
        for time, state in seq:
            writer.writerow({"time": time, "dominant_state": state})


def compute_transition_metrics(
    seq: list[tuple[float, str]],
    t_end: float,
) -> dict[str, float]:
    states = [
        "latent_weak",
        "sustained_weak",
        "functionally_empty",
        "restored_weak",
        "stabilizing",
        "crisis_support",
        "distorted",
        "rigid",
    ]
    dwell = {state: 0.0 for state in states}
    transitions = {state: 0.0 for state in states}

    if not seq:
        return {f"dwell_{state}": 0.0 for state in states} | {f"entries_{state}": 0.0 for state in states} | {
            "n_transitions": 0.0
        }

    for idx, (time, state) in enumerate(seq):
        next_time = seq[idx + 1][0] if idx + 1 < len(seq) else t_end
        dwell[state] += float(next_time - time)
        transitions[state] += 1.0

    metrics: dict[str, float] = {"n_transitions": float(max(len(seq) - 1, 0))}
    for state in states:
        metrics[f"dwell_{state}"] = float(dwell[state])
        metrics[f"entries_{state}"] = float(transitions[state])
    return metrics


def compute_tail_metrics(rows: list[dict[str, float]], threshold: float = 0.92) -> dict[str, float]:
    times = np.array([row["time"] for row in rows], dtype=float)
    gatherability = np.array([row["gatherability"] for row in rows], dtype=float)
    reentry = np.array([row["mean_reentry"] for row in rows], dtype=float)
    route_pressure = np.array([row["route_pressure"] for row in rows], dtype=float)
    restored_weak = np.array([row["restored_weak"] for row in rows], dtype=float)
    stabilizing = np.array([row["stabilizing"] for row in rows], dtype=float)

    peak_idx = int(np.argmax(route_pressure))
    peak_time = float(times[peak_idx])
    post_mask = times >= peak_time
    post_times = times[post_mask]
    post_gatherability = gatherability[post_mask]
    post_reentry = reentry[post_mask]
    post_restored = restored_weak[post_mask]
    post_stabilizing = stabilizing[post_mask]

    tail_duration = float(post_times[-1] - peak_time)
    below = post_gatherability < threshold
    weak_only = (post_restored > 0.0) & (post_stabilizing <= 0.0)

    recovery_time = float("nan")
    for t, g in zip(post_times, post_gatherability):
        if g >= threshold:
            recovery_time = float(t - peak_time)
            break

    time_in_weak_regime = float(np.sum(weak_only) * (post_times[1] - post_times[0])) if len(post_times) > 1 else 0.0
    if len(post_times) > 1:
        dt = float(post_times[1] - post_times[0])
        tail_area = float(np.sum(np.maximum(threshold - post_gatherability, 0.0)) * dt)
        reentry_tail_area = float(np.sum(np.maximum(threshold - post_reentry, 0.0)) * dt)
    else:
        tail_area = 0.0
        reentry_tail_area = 0.0
    below_duration = float(np.sum(below) * (post_times[1] - post_times[0])) if len(post_times) > 1 else 0.0

    return {
        "tail_threshold": float(threshold),
        "peak_time": peak_time,
        "tail_duration_window": tail_duration,
        "recovery_time_to_threshold": recovery_time,
        "time_below_threshold": below_duration,
        "tail_area": tail_area,
        "reentry_tail_area": reentry_tail_area,
        "time_in_restored_weak_only": time_in_weak_regime,
    }


def main() -> None:
    base_params = BondRouteParams()
    out_dir = Path(ROOT) / "outputs" / "volume_vi_bond_routes"
    out_dir.mkdir(parents=True, exist_ok=True)

    route_names = ["drift", "shock", "distortion"]
    seeds = [7, 31]
    sizes = [100]
    variants = [
        ("base", {}),
        ("base_intervention", {"intervention_enabled": True}),
    ]
    per_seed_rows: list[dict[str, float | str]] = []
    aggregate_rows: list[dict[str, float | str]] = []
    canonical_results: dict[str, dict[str, np.ndarray | list[dict[str, float]]]] = {}

    summary_rows: list[dict[str, float | str]] = []
    for n_agents in sizes:
        for variant_name, overrides in variants:
            for seed in seeds:
                params = BondRouteParams(seed=seed, n_agents=n_agents, **overrides)
                results = {route: simulate_route(route, params) for route in route_names}
                if n_agents == sizes[0] and variant_name == "base" and seed == seeds[0]:
                    canonical_results = results

                for route, out in results.items():
                    rows = out["timeseries"]
                    row: dict[str, float | str] = {
                        "route": route,
                        "seed": float(seed),
                        "variant": variant_name,
                        "n_agents": float(n_agents),
                    }
                    for label, value in zip(out["summary_labels"], out["summary"]):
                        row[str(label)] = float(value)
                    for key, value in out["final_states"].items():
                        row[f"final_{key}"] = float(value)
                    for key, value in out["peak_states"].items():
                        row[f"peak_{key}"] = float(value)
                    for key, value in compute_tail_metrics(rows).items():
                        row[key] = float(value)
                    for key, value in compute_transition_metrics(out["dominant_state_sequence"], params.t_end).items():
                        row[key] = float(value)
                    per_seed_rows.append(row)

                    print(
                        f"n={n_agents} variant={variant_name} seed={seed} {route}: "
                        f"gatherability_final={row['final_gatherability']:.3f}, "
                        f"tail_area={row['tail_area']:.3f}, n_transitions={row['n_transitions']:.0f}, "
                        f"final_distorted={row['final_distorted']:.0f}"
                    )

    for n_agents in sizes:
        for variant_name, _ in variants:
            for route in route_names:
                route_rows = [
                    row
                    for row in per_seed_rows
                    if row["route"] == route and row["variant"] == variant_name and row["n_agents"] == float(n_agents)
                ]
                numeric_keys = [key for key, value in route_rows[0].items() if isinstance(value, float)]
                agg: dict[str, float | str] = {"route": route, "variant": variant_name, "n_agents": float(n_agents)}
                for key in numeric_keys:
                    values = np.array([float(row[key]) for row in route_rows], dtype=float)
                    finite = values[np.isfinite(values)]
                    agg[f"mean_{key}"] = float(np.mean(finite)) if finite.size else float("nan")
                    agg[f"std_{key}"] = float(np.std(finite)) if finite.size else float("nan")
                aggregate_rows.append(agg)
    for n_agents in sizes:
        for route in route_names:
            route_rows = [
                row
                for row in per_seed_rows
                if row["route"] == route and row["variant"] == "base" and row["n_agents"] == float(n_agents)
            ]
            numeric_keys = [key for key, value in route_rows[0].items() if isinstance(value, float)]
            agg: dict[str, float | str] = {"route": route, "n_agents": float(n_agents)}
            for key in numeric_keys:
                values = np.array([float(row[key]) for row in route_rows], dtype=float)
                finite = values[np.isfinite(values)]
                agg[f"mean_{key}"] = float(np.mean(finite)) if finite.size else float("nan")
                agg[f"std_{key}"] = float(np.std(finite)) if finite.size else float("nan")
            aggregate_rows.append({"variant": "base_only", **agg})

    results = canonical_results
    for route, out in results.items():
        rows = out["timeseries"]
        write_timeseries_csv(out_dir / f"{route}_timeseries.csv", rows)
        write_state_sequence_csv(out_dir / f"{route}_state_sequence.csv", out["dominant_state_sequence"])

    summary_rows = per_seed_rows

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate_path = out_dir / "aggregate_summary.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    if plt is not None:
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        colors = {"drift": "#2e8b57", "shock": "#c44e52", "distortion": "#4c72b0"}
        for route, out in results.items():
            color = colors[route]
            axes[0].plot(out["t"], out["route_pressure"], label=route, color=color)
            axes[1].plot(out["t"], out["gatherability"], label=route, color=color)
            axes[2].plot(out["t"], out["mean_reentry"], label=route, color=color)
            axes[3].plot(out["t"], out["mean_distortion"], label=route, color=color)

        axes[0].set_ylabel("route pressure")
        axes[1].set_ylabel("gatherability")
        axes[2].set_ylabel("mean re-entry")
        axes[3].set_ylabel("mean distortion")
        axes[3].set_xlabel("time")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        plot_path = out_dir / "routes_overview.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {plot_path}")

    print(f"Wrote {summary_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
