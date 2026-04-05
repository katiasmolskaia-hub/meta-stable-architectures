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
    mixed_drift_delay: float = 2.4
    uneven_group_offset: float = 0.9
    uneven_group_skew: float = 0.28
    mixed_shock_weight: float = 0.95
    mixed_drift_weight: float = 0.72
    mixed_hard_mode: bool = False
    field_memory_decay: float = 0.16
    route_memory_gain: float = 0.52
    route_memory_bridge_bias: float = 0.30
    rebond_gain: float = 0.42
    rebond_bridge_gain: float = 0.34
    local_rebond_gain: float = 0.54
    bridge_rebond_gain: float = 0.18
    rebond_delay_center: float = 13.2
    rebond_delay_width: float = 1.6
    fallback_gain: float = 0.22
    bond_adaptation_gain: float = 0.34
    bond_tension_gain: float = 0.30
    bond_adaptation_decay: float = 0.12
    bond_fatigue_gain: float = 0.24
    bond_fatigue_decay: float = 0.10
    local_adaptation_bias: float = 0.18
    bridge_adaptation_penalty: float = 0.22
    bridge_fatigue_bias: float = 0.26
    bond_adaptation_threshold: float = 0.72
    bond_adaptation_cap: float = 0.88
    rc_trigger_gatherability: float = 0.84
    rc_trigger_duration: float = 1.8
    bridge_drive_gain: float = 0.95
    bridge_fatigue_gain: float = 0.22
    bridge_fatigue_decay: float = 0.10
    bridge_difference_soft: float = 0.18
    bridge_difference_hard: float = 0.42
    bridge_collapse_threshold: float = 0.34
    bridge_difference_window: float = 0.12
    bridge_attempt_gain: float = 0.42
    bridge_attempt_decay: float = 0.10
    bridge_failure_gain: float = 0.55
    bridge_failure_decay: float = 0.08
    interface_field_gain: float = 0.62
    interface_field_decay: float = 0.12
    interface_bridge_support: float = 0.26
    interface_rc_bias: float = 0.18
    interface_overload_penalty: float = 0.34
    interface_rate_scale: float = 0.45
    bridge_interface_rate_scale: float = 0.55
    compatibility_gain: float = 0.58
    compatibility_decay: float = 0.16
    compatibility_rebond_support: float = 0.36
    compatibility_memory_gain: float = 0.34
    compatibility_memory_decay: float = 0.10
    compatibility_late_carry: float = 0.42


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


def compute_bridge_window(
    group_gap: np.ndarray,
    params: BondRouteParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    soft = max(params.bridge_difference_soft, 1e-6)
    hard = max(params.bridge_difference_hard, soft + 1e-6)
    window = max(params.bridge_difference_window, 1e-6)

    quiet_gap = np.clip((soft - group_gap) / soft, 0.0, 1.0)
    moderate_center = 0.5 * (soft + hard)
    moderate_width = max(0.5 * (hard - soft), window)
    moderate_gap = np.exp(-0.5 * ((group_gap - moderate_center) / moderate_width) ** 2)
    moderate_gap *= (group_gap >= soft).astype(float) * (group_gap <= hard).astype(float)
    overload_gap = np.clip((group_gap - hard) / window, 0.0, 1.0)
    return quiet_gap, moderate_gap, overload_gap


def shock_wave(t: np.ndarray, center: float, width: float) -> np.ndarray:
    wave = np.exp(-0.5 * ((t - center) / (0.45 * width)) ** 2)
    return wave / np.max(wave)


def drift_wave(t: np.ndarray, center: float, width: float) -> np.ndarray:
    base = np.exp(-0.5 * ((t - center) / width) ** 2)
    base /= np.max(base)
    ramp_start = max(center - 0.9 * width - 2.0, 0.0)
    ramp_end = center + width
    ramp = np.clip((t - ramp_start) / max(ramp_end - ramp_start, 1e-6), 0.0, 1.0)
    return np.clip(0.65 * base + 0.25 * ramp, 0.0, 1.0)


def route_profile(
    t: np.ndarray,
    route: str,
    center: float,
    width: float,
    params: BondRouteParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    base = np.exp(-0.5 * ((t - center) / width) ** 2)
    base /= np.max(base)

    if route == "drift":
        drift = drift_wave(t, center, width)
        route_pressure = drift
        contagion = 0.25 + 0.20 * base
        components = {
            "shock": np.zeros_like(route_pressure),
            "drift": drift,
            "distortion": np.zeros_like(route_pressure),
        }
    elif route == "shock":
        shock = shock_wave(t, center, width)
        route_pressure = shock
        contagion = 0.20 + 0.10 * base
        components = {
            "shock": shock,
            "drift": np.zeros_like(route_pressure),
            "distortion": np.zeros_like(route_pressure),
        }
    elif route == "distortion":
        route_pressure = 0.80 * base + 0.15
        route_pressure = np.clip(route_pressure, 0.0, 1.0)
        contagion = 0.55 + 0.25 * base
        components = {
            "shock": np.zeros_like(route_pressure),
            "drift": np.zeros_like(route_pressure),
            "distortion": route_pressure,
        }
    elif route == "shock_drift":
        shock = shock_wave(t, center, width)
        drift = drift_wave(t, center + params.mixed_drift_delay, 1.15 * width)
        route_pressure = np.clip(
            params.mixed_shock_weight * shock + params.mixed_drift_weight * drift,
            0.0,
            1.35,
        )
        route_pressure /= np.max(route_pressure)
        contagion = np.clip(0.22 + 0.10 * shock + 0.16 * drift, 0.0, 1.0)
        components = {
            "shock": np.clip(params.mixed_shock_weight * shock, 0.0, 1.0),
            "drift": np.clip(params.mixed_drift_weight * drift, 0.0, 1.0),
            "distortion": np.zeros_like(route_pressure),
        }
    elif route == "full_process":
        early_drift = 0.45 * drift_wave(t, center - 1.9 * width, 1.45 * width)
        early_distortion = 0.22 * np.exp(-0.5 * ((t - (center - 0.9 * width)) / (1.15 * width)) ** 2)
        shock = 0.92 * shock_wave(t, center + 0.45 * width, 0.90 * width)
        late_drift = 0.72 * drift_wave(t, center + params.mixed_drift_delay + 1.35 * width, 1.55 * width)
        late_distortion = 0.26 * np.exp(-0.5 * ((t - (center + 2.25 * width)) / (1.30 * width)) ** 2)
        route_pressure = np.clip(early_drift + early_distortion + shock + late_drift + late_distortion, 0.0, 1.6)
        route_pressure /= np.max(route_pressure)
        contagion = np.clip(
            0.18
            + 0.08 * early_drift
            + 0.12 * shock
            + 0.14 * late_drift
            + 0.18 * (early_distortion + late_distortion),
            0.0,
            1.0,
        )
        components = {
            "shock": np.clip(shock, 0.0, 1.0),
            "drift": np.clip(early_drift + late_drift, 0.0, 1.0),
            "distortion": np.clip(early_distortion + late_distortion, 0.0, 1.0),
        }
    else:
        raise ValueError(f"unknown route: {route}")
    return route_pressure, np.clip(contagion, 0.0, 1.0), components


def build_group_route_profiles(
    t: np.ndarray,
    route: str,
    params: BondRouteParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    route_pressure, contagion, components = route_profile(t, route, params.route_center, params.route_width, params)
    n_groups = params.n_groups
    if route not in {"shock_drift", "full_process"}:
        expanded_pressure = np.repeat(route_pressure[:, None], n_groups, axis=1)
        expanded_contagion = np.repeat(contagion[:, None], n_groups, axis=1)
        expanded_components = {
            key: np.repeat(value[:, None], n_groups, axis=1)
            for key, value in components.items()
        }
        return expanded_pressure, expanded_contagion, expanded_components

    offset_scale = 1.65 if params.mixed_hard_mode else 1.0
    skew_scale = 1.55 if params.mixed_hard_mode else 1.0
    drift_delay = params.mixed_drift_delay + (1.1 if params.mixed_hard_mode else 0.0)
    drift_width = 1.30 * params.route_width if params.mixed_hard_mode else 1.15 * params.route_width

    offsets = np.linspace(
        -params.uneven_group_offset * offset_scale,
        params.uneven_group_offset * offset_scale,
        n_groups,
    )
    shock_gains = np.linspace(
        1.0 + params.uneven_group_skew * skew_scale,
        1.0 - params.uneven_group_skew * skew_scale,
        n_groups,
    )
    drift_gains = np.linspace(
        1.0 - params.uneven_group_skew * skew_scale,
        1.0 + params.uneven_group_skew * skew_scale,
        n_groups,
    )

    group_route_pressure = np.zeros((len(t), n_groups), dtype=float)
    group_contagion = np.zeros((len(t), n_groups), dtype=float)
    group_components = {
        "shock": np.zeros((len(t), n_groups), dtype=float),
        "drift": np.zeros((len(t), n_groups), dtype=float),
        "distortion": np.zeros((len(t), n_groups), dtype=float),
    }

    for group_idx in range(n_groups):
        shock = shock_gains[group_idx] * shock_wave(
            t,
            params.route_center + offsets[group_idx],
            params.route_width,
        )
        drift = drift_gains[group_idx] * drift_wave(
            t,
            params.route_center + drift_delay - 0.55 * offsets[group_idx],
            drift_width,
        )
        if route == "shock_drift":
            group_route_pressure[:, group_idx] = np.clip(
                params.mixed_shock_weight * shock + params.mixed_drift_weight * drift,
                0.0,
                1.4,
            )
            group_contagion[:, group_idx] = np.clip(0.22 + 0.10 * shock + 0.16 * drift, 0.0, 1.0)
            group_components["shock"][:, group_idx] = np.clip(params.mixed_shock_weight * shock, 0.0, 1.0)
            group_components["drift"][:, group_idx] = np.clip(params.mixed_drift_weight * drift, 0.0, 1.0)
            continue

        early_drift = 0.45 * drift_wave(
            t,
            params.route_center - 1.9 * params.route_width + 0.35 * offsets[group_idx],
            1.45 * params.route_width,
        )
        early_distortion = 0.22 * np.exp(
            -0.5
            * (
                (t - (params.route_center - 0.9 * params.route_width + 0.25 * offsets[group_idx]))
                / (1.15 * params.route_width)
            )
            ** 2
        )
        late_drift = 0.72 * drift_wave(
            t,
            params.route_center + drift_delay + 1.35 * params.route_width - 0.45 * offsets[group_idx],
            1.55 * params.route_width,
        )
        late_distortion = 0.26 * np.exp(
            -0.5
            * (
                (t - (params.route_center + 2.25 * params.route_width - 0.30 * offsets[group_idx]))
                / (1.30 * params.route_width)
            )
            ** 2
        )
        full_shock = np.clip(0.92 * params.mixed_shock_weight * shock, 0.0, 1.0)
        full_drift = np.clip(early_drift + late_drift, 0.0, 1.0)
        full_distortion = np.clip(early_distortion + late_distortion, 0.0, 1.0)
        group_route_pressure[:, group_idx] = np.clip(full_shock + full_drift + full_distortion, 0.0, 1.6)
        group_contagion[:, group_idx] = np.clip(
            0.18 + 0.08 * early_drift + 0.12 * full_shock + 0.14 * late_drift + 0.18 * full_distortion,
            0.0,
            1.0,
        )
        group_components["shock"][:, group_idx] = full_shock
        group_components["drift"][:, group_idx] = full_drift
        group_components["distortion"][:, group_idx] = full_distortion

    peak = np.max(group_route_pressure)
    if peak > 0.0:
        group_route_pressure /= peak
    return group_route_pressure, group_contagion, group_components


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


def process_phase_name(
    time: float,
    route_pressure: float,
    route_memory: float,
    field_memory: float,
    local_rebond: float,
    bridge_rebond: float,
    central_fallback: float,
    gatherability: float,
) -> str:
    rebond_total = local_rebond + bridge_rebond + central_fallback
    if route_pressure > 0.58:
        return "acute_crisis"
    if gatherability < 0.82 and (route_memory + field_memory) > 0.40:
        return "memory_drag"
    if rebond_total > 0.16 and time > 0.0:
        return "rebonding"
    if route_memory > 0.22 or field_memory > 0.18:
        return "route_imprint"
    return "background"


def process_lead_name(
    route_pressure: float,
    route_memory: float,
    field_memory: float,
    local_rebond: float,
    bridge_rebond: float,
    central_fallback: float,
    gatherability: float,
) -> str:
    scores = {
        "field": field_memory + max(0.0, 0.9 - gatherability),
        "route_memory": route_memory,
        "local_rebond": local_rebond,
        "bridge_rebond": bridge_rebond,
        "rc_fallback": central_fallback,
        "crisis_drive": route_pressure,
    }
    return max(scores.items(), key=lambda item: item[1])[0]


def simulate_route(route: str, params: BondRouteParams) -> dict[str, np.ndarray | list[dict[str, float]]]:
    rng = np.random.default_rng(params.seed)
    groups = assign_groups(params.n_agents, params.n_groups)
    edges = build_group_structured_edges(params.n_agents, params.coupling_radius, groups)
    n_edges = len(edges)
    bond_classes = assign_bond_classes(edges, groups)
    t = np.arange(0.0, params.t_end + params.dt, params.dt)

    route_pressure_by_group, contagion_by_group, route_components_by_group = build_group_route_profiles(t, route, params)
    route_pressure = np.mean(route_pressure_by_group, axis=1)
    contagion = np.mean(contagion_by_group, axis=1)
    edge_group_i = np.array([groups[i] for i, _ in edges], dtype=int)
    edge_group_j = np.array([groups[j] for _, j in edges], dtype=int)

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
    route_memory = np.zeros(n_edges, dtype=float)
    field_memory = np.zeros(n_edges, dtype=float)
    local_rebond = np.zeros(n_edges, dtype=float)
    bridge_rebond = np.zeros(n_edges, dtype=float)
    central_fallback = np.zeros(n_edges, dtype=float)
    bond_adaptation = np.clip(0.42 + 0.03 * rng.normal(size=n_edges), 0.10, 0.85)
    bond_fatigue = np.zeros(n_edges, dtype=float)
    unresolved_stress_time = np.zeros(n_edges, dtype=float)
    bridge_drive = np.zeros(n_edges, dtype=float)
    bridge_fatigue = np.zeros(n_edges, dtype=float)
    bridge_collapse = np.zeros(n_edges, dtype=float)
    bridge_attempt_memory = np.zeros(n_edges, dtype=float)
    bridge_failure_trace = np.zeros(n_edges, dtype=float)
    interface_field = 0.0
    compatibility_window = np.zeros(n_edges, dtype=float)
    compatibility_memory = np.zeros(n_edges, dtype=float)

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
        "shock_drift": {
            "f_loss": 0.08,
            "s_loss": 0.16,
            "d_gain": 0.06,
            "d_relief": 0.05,
            "b_support": 0.04,
            "late_v_drag": 0.02,
            "late_f_drag": 0.02,
            "late_s_drag": 0.03,
        },
        "full_process": {
            "f_loss": 0.09,
            "s_loss": 0.14,
            "d_gain": 0.09,
            "d_relief": 0.05,
            "b_support": 0.02,
            "late_v_drag": 0.03,
            "late_f_drag": 0.03,
            "late_s_drag": 0.04,
        },
    }[route]

    for idx, time in enumerate(t):
        interface_dt = params.dt * params.interface_rate_scale
        bridge_interface_dt = params.dt * params.bridge_interface_rate_scale
        reentry = viability * (
            params.lambda_s * stability + params.lambda_f * freedom - params.lambda_d * distortion
        )
        gatherability = float(np.clip(np.mean(np.maximum(reentry, 0.0)), 0.0, 1.0))
        current_group_route = route_pressure_by_group[idx]
        current_group_contagion = contagion_by_group[idx]
        current_route = float(np.mean(current_group_route))
        current_contagion = float(np.mean(current_group_contagion))
        post_peak_gate = 1.0 / (1.0 + np.exp(-(time - (params.route_center + 1.15 * params.route_width)) / 1.8))
        route_shock = 0.5 * (
            route_components_by_group["shock"][idx, edge_group_i]
            + route_components_by_group["shock"][idx, edge_group_j]
        )
        route_drift = 0.5 * (
            route_components_by_group["drift"][idx, edge_group_i]
            + route_components_by_group["drift"][idx, edge_group_j]
        )
        route_distortion = 0.5 * (
            route_components_by_group["distortion"][idx, edge_group_i]
            + route_components_by_group["distortion"][idx, edge_group_j]
        )
        edge_route = 0.5 * (current_group_route[edge_group_i] + current_group_route[edge_group_j])
        edge_contagion = 0.5 * (current_group_contagion[edge_group_i] + current_group_contagion[edge_group_j])
        class_route_mult = np.where(
            bond_classes == "local",
            0.82 if route == "shock" else 0.92,
            np.where(
                bond_classes == "bridge",
                1.20 if route in {"drift", "distortion", "shock_drift"} else 1.08,
                1.0,
            ),
        )
        effective_route = np.clip(edge_route * class_route_mult, 0.0, 1.4)
        late_phase = 1.0 / (1.0 + np.exp(-(time - (params.route_center + 1.4)) / 1.2))
        relapse_wave = 0.0
        if params.relapse_enabled and route in {"drift", "distortion", "shock_drift", "full_process"}:
            relapse_center = params.route_center + params.relapse_center_offset
            relapse_wave = params.relapse_gain * np.exp(
                -0.5 * ((time - relapse_center) / params.relapse_width) ** 2
            )
        intervention_gain = 1.0 if params.intervention_enabled else 0.0
        i_shock = intervention_gain * route_shock
        i_bridge = intervention_gain * route_drift
        i_block = intervention_gain * (0.65 * (route_distortion > 0.0) + 0.35 * route_distortion)
        local_mask = (bond_classes == "local").astype(float)
        bridge_mask = (bond_classes == "bridge").astype(float)
        group_gap = np.abs(current_group_route[edge_group_i] - current_group_route[edge_group_j])
        agreement_signal = np.clip(
            0.40 * np.maximum(reentry, 0.0)
            + 0.25 * stability
            + 0.20 * viability
            + 0.15 * gatherability,
            0.0,
            1.5,
        )
        tension_signal = np.clip(
            params.bond_tension_gain * effective_route
            + 0.26 * distortion
            + 0.18 * group_gap
            + 0.12 * np.maximum(0.0, 1.0 - viability),
            0.0,
            1.5,
        )
        adaptation_gain_mask = np.clip(
            1.0 + params.local_adaptation_bias * local_mask - params.bridge_adaptation_penalty * bridge_mask,
            0.25,
            1.4,
        )
        fatigue_gain_mask = np.clip(
            1.0 + params.bridge_fatigue_bias * bridge_mask,
            1.0,
            1.6,
        )
        adaptation_gate = (agreement_signal > params.bond_adaptation_threshold).astype(float)
        need_for_sync = np.maximum(0.0, params.rc_trigger_gatherability - gatherability)
        reentry_mismatch = np.abs(
            reentry[edge_group_i] - reentry[edge_group_j]
        )
        quiet_gap, moderate_gap, overload_gap = compute_bridge_window(group_gap, params)
        bridge_presence = max(float(np.mean(bridge_mask)), 1e-6)
        interface_support = (
            0.42 * class_mean(moderate_gap, bond_classes, "bridge")
            + 0.24 * float(np.mean(bridge_attempt_memory))
            + 0.18 * class_mean(route_memory, bond_classes, "bridge")
            + 0.16 * float(np.mean(reentry_mismatch * bridge_mask) / bridge_presence)
        )
        interface_drag = (
            0.42 * class_mean(overload_gap, bond_classes, "bridge")
            + 0.26 * float(np.mean(bridge_collapse))
            + 0.22 * float(np.mean(bridge_failure_trace))
            + 0.18 * max(0.0, 0.82 - gatherability)
        )
        interface_field = float(
            np.clip(
                interface_field
                + interface_dt
                * (
                    params.interface_field_gain * interface_support
                    - params.interface_overload_penalty * interface_drag
                    - params.interface_field_decay * interface_field
                ),
                0.0,
                1.5,
            )
        )
        compatibility_seed = (
            0.34 * moderate_gap
            + 0.24 * np.clip(interface_field / 1.5, 0.0, 1.0)
            + 0.22 * np.clip(reentry_mismatch / 0.08, 0.0, 1.0)
            + 0.16 * np.clip(bridge_attempt_memory / 0.05, 0.0, 1.0)
        )
        compatibility_guard = np.clip(
            1.0
            - 0.55 * overload_gap
            - 0.45 * bridge_collapse
            - 0.38 * bridge_failure_trace,
            0.0,
            1.0,
        )
        compatibility_memory = np.clip(
            compatibility_memory
            + interface_dt
            * (
                params.compatibility_memory_gain
                * bridge_mask
                * (
                    0.45 * moderate_gap
                    + 0.30 * np.clip(interface_field / 1.5, 0.0, 1.0)
                    + 0.25 * np.clip(reentry_mismatch / 0.08, 0.0, 1.0)
                )
                * compatibility_guard
                - (
                    params.compatibility_memory_decay
                    + 0.12 * overload_gap
                    + 0.10 * bridge_collapse
                    + 0.08 * bridge_failure_trace
                )
                * compatibility_memory
            ),
            0.0,
            1.2,
        )
        compatibility_support = (
            params.compatibility_gain
            * bridge_mask
            * (
                compatibility_seed
                + 0.22 * compatibility_memory
                + params.compatibility_late_carry * post_peak_gate * compatibility_memory
            )
            * compatibility_guard
        )
        compatibility_window = np.clip(
            compatibility_window
            + interface_dt
            * (
                compatibility_support
                - (
                    params.compatibility_decay
                    + 0.10 * overload_gap
                    + 0.08 * bridge_collapse
                    + 0.08 * bridge_failure_trace
                )
                * compatibility_window
            ),
            0.0,
            1.2,
        )
        rapid_changes = np.clip(
            0.45 * relapse_wave + 0.30 * np.abs(route_drift - route_shock),
            0.0,
            1.0,
        )
        if route in {"shock", "shock_drift", "full_process"}:
            shock_support_charge = np.clip(
                shock_support_charge
                + params.dt
                * (0.90 * route_shock * (bond_classes != "bridge") - 0.24 * shock_support_charge),
                0.0,
                1.0,
            )
        if route in {"drift", "shock_drift", "full_process"}:
            drift_erosion = np.clip(
                drift_erosion
                + params.dt * (0.18 * route_drift + 0.05 * late_phase - 0.10 * gatherability),
                0.0,
                1.0,
            )
        bond_adaptation = np.clip(
            bond_adaptation
            + params.dt
            * (
                params.bond_adaptation_gain
                * adaptation_gain_mask
                * adaptation_gate
                * agreement_signal
                * np.maximum(0.0, 1.0 - bond_fatigue)
                - fatigue_gain_mask * tension_signal
                - params.bond_adaptation_decay * bond_adaptation
            ),
            0.0,
            params.bond_adaptation_cap,
        )
        bond_fatigue = np.clip(
            bond_fatigue
            + params.dt
            * (
                params.bond_fatigue_gain
                * fatigue_gain_mask
                * (
                    tension_signal
                    + 0.18 * route_memory
                    + 0.14 * field_memory
                    - 0.16 * bond_adaptation
                )
                - params.bond_fatigue_decay * bond_fatigue
            ),
            0.0,
            1.5,
        )
        unresolved_stress_time = np.clip(
            unresolved_stress_time
            + params.dt
            * (
                (
                    float(gatherability < params.rc_trigger_gatherability)
                    + 0.6 * (bond_fatigue > 0.18).astype(float)
                    + 0.5 * bridge_mask * (group_gap > 0.10).astype(float)
                )
                - 0.75 * float(gatherability >= params.rc_trigger_gatherability) * unresolved_stress_time
            ),
            0.0,
            8.0,
        )
        bridge_drive = np.clip(
            bridge_drive
            + bridge_interface_dt
            * (
                params.bridge_drive_gain
                * bridge_mask
                * (
                    moderate_gap
                    * (
                        0.60 * need_for_sync
                        + 0.58 * reentry_mismatch
                        + 0.25 * np.maximum(0.0, 1.0 - gatherability)
                        + 0.22 * bridge_attempt_memory
                        + params.interface_bridge_support * interface_field
                        + 0.18 * compatibility_window
                        + 0.08 * compatibility_memory
                    )
                )
                - (
                    0.08
                    + 0.12 * quiet_gap
                    + 0.42 * overload_gap
                    + 0.22 * bridge_collapse
                    + 0.18 * bridge_failure_trace
                )
                * bridge_drive
            ),
            0.0,
            1.2,
        )
        bridge_attempt_memory = np.clip(
            bridge_attempt_memory
            + bridge_interface_dt
            * (
                params.bridge_attempt_gain
                * bridge_mask
                * moderate_gap
                * (0.56 * need_for_sync + 0.44 * reentry_mismatch + 0.26 * bridge_drive)
                - (
                    params.bridge_attempt_decay
                    + 0.08 * quiet_gap
                    + 0.22 * overload_gap
                    + 0.16 * bridge_failure_trace
                    + 0.12 * bridge_collapse
                )
                * bridge_attempt_memory
            ),
            0.0,
            1.2,
        )
        bridge_fatigue = np.clip(
            bridge_fatigue
            + bridge_interface_dt
            * (
                params.bridge_fatigue_gain
                * bridge_mask
                * (
                    0.04 * quiet_gap
                    + 0.16 * moderate_gap
                    + 0.52 * overload_gap
                    + 0.20 * group_gap
                    + 0.18 * reentry_mismatch
                    + 0.20 * route_memory
                    + 0.18 * distortion
                    + 0.18 * rapid_changes
                    + 0.14 * need_for_sync
                    - 0.16 * bridge_drive
                    + 0.16 * bridge_attempt_memory
                )
                - params.bridge_fatigue_decay * bridge_fatigue
            ),
            0.0,
            1.5,
        )
        bridge_collapse = np.clip(
            bridge_collapse
            + bridge_interface_dt
            * (
                bridge_mask
                * (
                    0.80 * (bridge_fatigue > params.bridge_collapse_threshold).astype(float)
                    + 0.48 * overload_gap
                    - 0.12 * moderate_gap
                )
                - 0.18 * bridge_collapse
            ),
            0.0,
            1.5,
        )
        bridge_failure_trace = np.clip(
            bridge_failure_trace
            + bridge_interface_dt
            * (
                params.bridge_failure_gain
                * bridge_mask
                * (
                    0.70 * bridge_collapse
                    + 0.36 * overload_gap
                    + 0.26 * (bridge_fatigue > params.bridge_collapse_threshold).astype(float)
                )
                - (
                    params.bridge_failure_decay
                    + 0.04 * moderate_gap
                    + 0.08 * bridge_drive
                )
                * bridge_failure_trace
            ),
            0.0,
            1.5,
        )
        if route in {"shock_drift", "full_process"}:
            hard_scale = 1.45 if params.mixed_hard_mode else 1.0
            rebond_gate = 1.0 / (1.0 + np.exp(-(time - params.rebond_delay_center) / params.rebond_delay_width))
            local_viability_gap = np.maximum(0.0, 0.72 - viability)
            bridge_viability_gap = np.maximum(0.0, 0.78 - viability)
            local_support = gatherability * (1.0 - route_shock) * (1.0 - distortion)
            central_need = np.maximum(0.0, 0.82 - gatherability)
            bridge_success_mode = 1.0 if params.bridge_rebond_gain >= 0.20 else 0.0
            stress_gate = np.clip(
                unresolved_stress_time / max(params.rc_trigger_duration, 1e-6),
                0.0,
                1.0,
            )
            route_memory = np.clip(
                route_memory
                + params.dt
                * (
                    hard_scale
                    * (
                        params.route_memory_gain * route_drift
                        + 0.24 * route_shock
                        + params.route_memory_bridge_bias * group_gap * bridge_mask
                    )
                    - params.field_memory_decay * route_memory
                ),
                0.0,
                1.5,
            )
            field_memory = np.clip(
                field_memory
                + params.dt
                * (
                    hard_scale
                    * (
                        0.34 * np.maximum(0.0, 1.0 - gatherability)
                        + 0.26 * relapse_wave
                        + 0.18 * group_gap
                    )
                    - 0.18 * field_memory
                ),
                0.0,
                1.5,
            )
            local_rebond = np.clip(
                local_rebond
                + params.dt
                * (
                    params.local_rebond_gain
                    * rebond_gate
                    * local_mask
                    * (
                        0.45 * local_support
                        + 0.18 * np.maximum(reentry, 0.0)
                        + 0.20 * local_viability_gap
                        + 0.22 * bond_adaptation
                        - 0.16 * bond_fatigue
                    )
                    - (
                        0.30 * route_drift
                        + 0.18 * route_shock
                        + 0.16 * route_memory
                        + 0.10 * group_gap
                        + 0.12 * bond_fatigue
                    )
                    * local_rebond
                ),
                0.0,
                1.5,
            )
            bridge_drive_term = (
                0.12 * gatherability
                + 0.16 * np.maximum(reentry, 0.0)
                + 0.24 * bridge_viability_gap
                + 0.60 * bridge_drive
                + 0.28 * bridge_attempt_memory
                + (0.22 + 0.20 * bridge_success_mode) * moderate_gap
                + params.interface_bridge_support * interface_field
                + params.compatibility_rebond_support * compatibility_window
                + 0.12 * compatibility_memory
                + 0.04 * bond_adaptation
            )
            bridge_fatigue_term = (
                0.20 * bond_fatigue
                + 0.18 * bridge_fatigue
                + 0.08 * quiet_gap
                + (0.24 - 0.08 * bridge_success_mode) * overload_gap
            )
            bridge_collapse_term = (
                (0.60 - 0.10 * bridge_success_mode) * bridge_collapse
                + (0.24 - 0.08 * bridge_success_mode) * bridge_failure_trace
            )
            bridge_rebond_support = np.maximum(
                0.0,
                bridge_drive_term - bridge_fatigue_term - bridge_collapse_term,
            )
            bridge_rebond_decay = (
                0.34 * route_drift
                + 0.12 * route_shock
                + 0.20 * route_memory
                + 0.16 * field_memory
                + 0.06 * quiet_gap
                + 0.14 * bond_fatigue
                + 0.18 * bridge_fatigue
                + (0.22 - 0.08 * bridge_success_mode) * overload_gap
                + (0.40 - 0.08 * bridge_success_mode) * bridge_collapse
                + (0.28 - 0.10 * bridge_success_mode) * bridge_failure_trace
            )
            bridge_rebond = np.clip(
                bridge_rebond
                + params.dt
                * (
                    (params.bridge_rebond_gain + 0.75 * bridge_drive)
                    * rebond_gate
                    * bridge_mask
                    * bridge_rebond_support
                    - bridge_rebond_decay * bridge_rebond
                ),
                0.0,
                1.2,
            )
            central_fallback = np.clip(
                central_fallback
                + params.dt
                * (
                    params.fallback_gain
                    * bridge_mask
                    * stress_gate
                    * (
                        central_need
                        + params.interface_rc_bias * interface_field
                        + 0.35 * bond_fatigue
                        + 0.20 * group_gap
                        + 0.45 * bridge_collapse
                        + 0.20 * bridge_fatigue
                        + 0.24 * bridge_failure_trace
                    )
                    * rebond_gate
                    * (
                        1.0
                        - np.clip(
                            0.70 * local_rebond
                            + 0.65 * bridge_rebond
                            + 0.18 * bond_adaptation
                            + 0.18 * bridge_drive
                            + 0.16 * bridge_attempt_memory,
                            0.0,
                            1.0,
                        )
                    )
                    - (0.26 + 0.18 * route_memory + 0.12 * field_memory) * central_fallback
                ),
                0.0,
                0.9,
            )
        else:
            route_memory *= max(0.0, 1.0 - params.dt * 0.25)
            field_memory *= max(0.0, 1.0 - params.dt * 0.20)
            local_rebond *= max(0.0, 1.0 - params.dt * 0.15)
            bridge_rebond *= max(0.0, 1.0 - params.dt * 0.10)
            central_fallback *= max(0.0, 1.0 - params.dt * 0.08)
            bond_adaptation *= max(0.0, 1.0 - params.dt * 0.12)
            bond_fatigue *= max(0.0, 1.0 - params.dt * 0.10)
            unresolved_stress_time *= max(0.0, 1.0 - params.dt * 0.18)
            bridge_drive *= max(0.0, 1.0 - params.dt * 0.18)
            bridge_fatigue *= max(0.0, 1.0 - params.dt * 0.12)
            bridge_collapse *= max(0.0, 1.0 - params.dt * 0.10)
            bridge_attempt_memory *= max(0.0, 1.0 - params.dt * 0.10)
            bridge_failure_trace *= max(0.0, 1.0 - params.dt * 0.08)
            compatibility_window *= max(0.0, 1.0 - params.dt * 0.08)
            compatibility_memory *= max(0.0, 1.0 - params.dt * 0.06)
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

        phase_name = process_phase_name(
            time=float(time),
            route_pressure=float(np.mean(effective_route)),
            route_memory=float(np.mean(route_memory)),
            field_memory=float(np.mean(field_memory)),
            local_rebond=float(np.mean(local_rebond)),
            bridge_rebond=float(np.mean(bridge_rebond)),
            central_fallback=float(np.mean(central_fallback)),
            gatherability=gatherability,
        )
        process_lead = process_lead_name(
            route_pressure=float(np.mean(effective_route)),
            route_memory=float(np.mean(route_memory)),
            field_memory=float(np.mean(field_memory)),
            local_rebond=float(np.mean(local_rebond)),
            bridge_rebond=float(np.mean(bridge_rebond)),
            central_fallback=float(np.mean(central_fallback)),
            gatherability=gatherability,
        )

        records.append(
            {
                "time": float(time),
                "process_phase": phase_name,
                "process_lead": process_lead,
                "route_pressure": float(route_pressure[idx]),
                "effective_route_pressure": float(np.mean(effective_route)),
                "relapse_wave": float(relapse_wave),
                "contagion": float(contagion[idx]),
                "group_route_std": float(np.std(current_group_route)),
                "bond_adaptation_index": float(np.mean(bond_adaptation)),
                "bond_fatigue_index": float(np.mean(bond_fatigue)),
                "unresolved_stress_index": float(np.mean(unresolved_stress_time)),
                "bridge_drive_index": float(np.mean(bridge_drive)),
                "bridge_fatigue_index": float(np.mean(bridge_fatigue)),
                "bridge_collapse_index": float(np.mean(bridge_collapse)),
                "bridge_attempt_memory_index": float(np.mean(bridge_attempt_memory)),
                "bridge_failure_trace_index": float(np.mean(bridge_failure_trace)),
                "interface_field_index": interface_field,
                "compatibility_window_index": float(np.mean(compatibility_window)),
                "compatibility_memory_index": float(np.mean(compatibility_memory)),
                "bridge_quiet_share": class_mean(quiet_gap, bond_classes, "bridge"),
                "bridge_active_share": class_mean(moderate_gap, bond_classes, "bridge"),
                "bridge_overload_share": class_mean(overload_gap, bond_classes, "bridge"),
                "route_memory_index": float(np.mean(route_memory)),
                "field_memory_index": float(np.mean(field_memory)),
                "local_rebonding_index": float(np.mean(local_rebond)),
                "bridge_rebonding_index": float(np.mean(bridge_rebond)),
                "central_fallback_index": float(np.mean(central_fallback)),
                "collective_rebonding_index": float(np.mean(local_rebond + bridge_rebond + central_fallback)),
                "bridge_strain_index": float(np.mean(group_gap * bridge_mask + route_memory * bridge_mask)),
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
            + 0.16 * bond_adaptation * local_mask
            + 0.04 * bond_adaptation * bridge_mask
            - 0.14 * route_memory * bridge_mask
            - 0.10 * field_memory * bridge_mask
            - 0.12 * bond_fatigue * local_mask
            - 0.22 * bond_fatigue * bridge_mask
            + 0.20 * local_rebond * local_mask
            + 0.12 * bridge_rebond * local_mask
            + 0.16 * bridge_rebond * bridge_mask
            + 0.16 * central_fallback * bridge_mask
            - relapse_wave * (0.20 * bridge_mask + 0.08 * local_mask)
        )
        d_freedom = (
            params.beta_open * (1.0 - freedom) * adaptive_open
            - params.beta_sf * stability * freedom
            - params.beta_df * distortion * freedom
            - route_factor["f_loss"] * effective_route * freedom
            - route_factor["late_f_drag"] * late_phase * freedom
            - 0.05 * drift_erosion * freedom
            + 0.10 * bond_adaptation * local_mask * (1.0 - freedom)
            + 0.03 * bond_adaptation * bridge_mask * (1.0 - freedom)
            + 0.12 * i_bridge * bridge_mask * (1.0 - freedom)
            + 0.06 * i_block * bridge_mask * (1.0 - freedom)
            - 0.08 * route_memory * bridge_mask * freedom
            - 0.08 * bond_fatigue * local_mask * freedom
            - 0.16 * bond_fatigue * bridge_mask * freedom
            + 0.06 * local_rebond * local_mask * (1.0 - freedom)
            + 0.08 * bridge_rebond * bridge_mask * (1.0 - freedom)
            + 0.08 * central_fallback * bridge_mask * (1.0 - freedom)
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
            + 0.08 * bond_adaptation * local_mask
            + 0.02 * bond_adaptation * bridge_mask
            - 0.06 * field_memory * bridge_mask
            - 0.08 * bond_fatigue * local_mask
            - 0.16 * bond_fatigue * bridge_mask
            + 0.08 * local_rebond
            + 0.06 * bridge_rebond
            + 0.10 * central_fallback * bridge_mask
            - relapse_wave * (0.14 * bridge_mask + 0.06 * local_mask)
        )
        d_distortion = (
            params.delta_r * effective_route
            + params.delta_c * edge_contagion
            - params.delta_s * stability
            - params.delta_b * viability
            + route_factor["d_gain"] * effective_route
            - route_factor["d_relief"] * gatherability
            + 0.02 * drift_erosion
            - 0.10 * shock_support_charge
            - 0.40 * i_block
            - 0.10 * bond_adaptation * local_mask * (1.0 - distortion)
            - 0.04 * bond_adaptation * bridge_mask * (1.0 - distortion)
            + 0.12 * route_memory * bridge_mask
            + 0.10 * field_memory * bridge_mask
            + 0.10 * bond_fatigue * local_mask
            + 0.18 * bond_fatigue * bridge_mask
            - 0.08 * i_bridge * bridge_mask
            - 0.08 * i_block * local_mask
            - 0.10 * local_rebond * local_mask * (1.0 - distortion)
            - 0.12 * bridge_rebond * bridge_mask * (1.0 - distortion)
            - 0.10 * central_fallback * bridge_mask * (1.0 - distortion)
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
                max(row["group_route_std"] for row in records),
                final["bond_adaptation_index"],
                max(row["bond_fatigue_index"] for row in records),
                max(row["unresolved_stress_index"] for row in records),
                max(row["bridge_drive_index"] for row in records),
                max(row["bridge_fatigue_index"] for row in records),
                max(row["bridge_collapse_index"] for row in records),
                max(row["bridge_attempt_memory_index"] for row in records),
                max(row["bridge_failure_trace_index"] for row in records),
                max(row["interface_field_index"] for row in records),
                max(row["compatibility_window_index"] for row in records),
                max(row["compatibility_memory_index"] for row in records),
                max(row["bridge_quiet_share"] for row in records),
                max(row["bridge_active_share"] for row in records),
                max(row["bridge_overload_share"] for row in records),
                final["route_memory_index"],
                max(row["field_memory_index"] for row in records),
                max(row["local_rebonding_index"] for row in records),
                max(row["bridge_rebonding_index"] for row in records),
                max(row["central_fallback_index"] for row in records),
                max(row["collective_rebonding_index"] for row in records),
                max(row["bridge_strain_index"] for row in records),
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
                "max_group_route_std",
                "final_bond_adaptation_index",
                "peak_bond_fatigue_index",
                "peak_unresolved_stress_index",
                "peak_bridge_drive_index",
                "peak_bridge_fatigue_index",
                "peak_bridge_collapse_index",
                "peak_bridge_attempt_memory_index",
                "peak_bridge_failure_trace_index",
                "peak_interface_field_index",
                "peak_compatibility_window_index",
                "peak_compatibility_memory_index",
                "peak_bridge_quiet_share",
                "peak_bridge_active_share",
                "peak_bridge_overload_share",
                "final_route_memory_index",
                "peak_field_memory_index",
                "peak_local_rebonding_index",
                "peak_bridge_rebonding_index",
                "peak_central_fallback_index",
                "peak_collective_rebonding_index",
                "peak_bridge_strain_index",
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


def compute_process_mirror_metrics(rows: list[dict[str, float | str]], t_end: float) -> dict[str, float]:
    phase_names = ["background", "acute_crisis", "route_imprint", "memory_drag", "rebonding"]
    lead_names = ["field", "route_memory", "local_rebond", "bridge_rebond", "rc_fallback", "crisis_drive"]
    metrics: dict[str, float] = {}
    if not rows:
        for phase in phase_names:
            metrics[f"dwell_phase_{phase}"] = 0.0
        for lead in lead_names:
            metrics[f"lead_time_{lead}"] = 0.0
        metrics["n_phase_switches"] = 0.0
        return metrics

    phase_dwell = {phase: 0.0 for phase in phase_names}
    lead_dwell = {lead: 0.0 for lead in lead_names}
    phase_switches = 0
    dt = 0.0
    if len(rows) > 1:
        dt = float(rows[1]["time"]) - float(rows[0]["time"])

    prev_phase = None
    for idx, row in enumerate(rows):
        phase = str(row["process_phase"])
        lead = str(row["process_lead"])
        next_time = float(rows[idx + 1]["time"]) if idx + 1 < len(rows) else t_end
        duration = max(0.0, next_time - float(row["time"]))
        if phase in phase_dwell:
            phase_dwell[phase] += duration
        if lead in lead_dwell:
            lead_dwell[lead] += duration
        if prev_phase is not None and phase != prev_phase:
            phase_switches += 1
        prev_phase = phase

    for phase in phase_names:
        metrics[f"dwell_phase_{phase}"] = float(phase_dwell[phase])
    for lead in lead_names:
        metrics[f"lead_time_{lead}"] = float(lead_dwell[lead])
    metrics["n_phase_switches"] = float(phase_switches)
    metrics["process_dt"] = float(dt)
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


def compute_bridge_time_metrics(rows: list[dict[str, float]], t_end: float) -> dict[str, float]:
    if not rows:
        return {}

    times = np.array([row["time"] for row in rows], dtype=float)
    route_pressure = np.array([row["route_pressure"] for row in rows], dtype=float)
    peak_time = float(times[int(np.argmax(route_pressure))])

    post_mask = times >= peak_time
    post_rows = [row for row, keep in zip(rows, post_mask) if keep]
    post_times = times[post_mask]
    if len(post_rows) < 3:
        post_rows = rows
        post_times = times

    span = max(float(post_times[-1] - post_times[0]), 1e-6)
    early_end = float(post_times[0] + 0.33 * span)
    mid_end = float(post_times[0] + 0.66 * span)

    early_rows = [row for row in post_rows if float(row["time"]) <= early_end]
    mid_rows = [row for row in post_rows if early_end < float(row["time"]) <= mid_end]
    late_rows = [row for row in post_rows if float(row["time"]) > mid_end]

    if not early_rows:
        early_rows = post_rows[:1]
    if not mid_rows:
        mid_rows = post_rows[len(post_rows) // 2 : len(post_rows) // 2 + 1]
    if not late_rows:
        late_rows = post_rows[-1:]

    tracked = [
        "interface_field_index",
        "compatibility_window_index",
        "compatibility_memory_index",
        "bridge_quiet_share",
        "bridge_active_share",
        "bridge_overload_share",
        "bridge_drive_index",
        "bridge_fatigue_index",
        "bridge_collapse_index",
        "bridge_rebonding_index",
        "bridge_failure_trace_index",
        "central_fallback_index",
        "gatherability",
    ]

    metrics: dict[str, float] = {
        "bridge_probe_peak_time": peak_time,
        "bridge_probe_post_window": span,
    }

    for key in tracked:
        full = np.array([float(row[key]) for row in post_rows], dtype=float)
        early = np.array([float(row[key]) for row in early_rows], dtype=float)
        mid = np.array([float(row[key]) for row in mid_rows], dtype=float)
        late = np.array([float(row[key]) for row in late_rows], dtype=float)
        metrics[f"timeavg_{key}"] = float(np.mean(full))
        metrics[f"early_{key}"] = float(np.mean(early))
        metrics[f"mid_{key}"] = float(np.mean(mid))
        metrics[f"late_{key}"] = float(np.mean(late))
        metrics[f"delta_{key}_late_minus_early"] = float(np.mean(late) - np.mean(early))

    return metrics


def main() -> None:
    base_params = BondRouteParams()
    out_dir = Path(ROOT) / "outputs" / "volume_vi_bond_routes"
    out_dir.mkdir(parents=True, exist_ok=True)

    route_names = ["drift", "shock", "distortion", "shock_drift", "full_process"]
    long_run_route_names = ["shock_drift"]
    full_process_route_names = ["full_process"]
    seeds = [7, 31]
    sizes = [100]
    variant_filter = os.environ.get("VOL6_VARIANT_FILTER", "").strip()
    variants = [
        ("base", {}),
        ("base_intervention", {"intervention_enabled": True}),
        (
            "mixed_uneven_hard",
            {
                "intervention_enabled": True,
                "mixed_hard_mode": True,
                "relapse_gain": 0.30,
                "mixed_drift_delay": 3.5,
                "uneven_group_offset": 1.5,
                "uneven_group_skew": 0.44,
                "route_memory_gain": 0.66,
                "route_memory_bridge_bias": 0.42,
                "local_rebond_gain": 0.44,
                "bridge_rebond_gain": 0.11,
                "rebond_delay_center": 14.1,
                "rebond_delay_width": 1.9,
                "fallback_gain": 0.12,
            },
        ),
        (
            "bridge_long_run_probe",
            {
                "intervention_enabled": True,
                "mixed_hard_mode": True,
                "t_end": 48.0,
                "relapse_gain": 0.34,
                "mixed_drift_delay": 3.8,
                "uneven_group_offset": 1.6,
                "uneven_group_skew": 0.46,
                "route_memory_gain": 0.68,
                "route_memory_bridge_bias": 0.44,
                "local_rebond_gain": 0.42,
                "bridge_rebond_gain": 0.13,
                "rebond_delay_center": 14.4,
                "rebond_delay_width": 2.0,
                "fallback_gain": 0.14,
            },
        ),
        (
            "bridge_moderate_rebond_probe",
            {
                "intervention_enabled": True,
                "mixed_hard_mode": True,
                "t_end": 48.0,
                "relapse_gain": 0.34,
                "mixed_drift_delay": 3.8,
                "uneven_group_offset": 1.6,
                "uneven_group_skew": 0.46,
                "route_memory_gain": 0.68,
                "route_memory_bridge_bias": 0.44,
                "local_rebond_gain": 0.42,
                "bridge_rebond_gain": 0.22,
                "rebond_delay_center": 14.4,
                "rebond_delay_width": 2.0,
                "fallback_gain": 0.14,
            },
        ),
        (
            "full_process_extended",
            {
                "intervention_enabled": True,
                "mixed_hard_mode": True,
                "t_end": 72.0,
                "route_center": 18.0,
                "route_width": 4.4,
                "relapse_gain": 0.30,
                "mixed_drift_delay": 5.4,
                "uneven_group_offset": 1.4,
                "uneven_group_skew": 0.42,
                "route_memory_gain": 0.70,
                "route_memory_bridge_bias": 0.46,
                "local_rebond_gain": 0.46,
                "bridge_rebond_gain": 0.20,
                "rebond_delay_center": 27.5,
                "rebond_delay_width": 3.0,
                "fallback_gain": 0.16,
            },
        ),
    ]
    per_seed_rows: list[dict[str, float | str]] = []
    aggregate_rows: list[dict[str, float | str]] = []
    canonical_results: dict[str, dict[str, np.ndarray | list[dict[str, float]]]] = {}

    summary_rows: list[dict[str, float | str]] = []
    for n_agents in sizes:
        for variant_name, overrides in variants:
            if variant_filter and variant_name not in {name.strip() for name in variant_filter.split(",") if name.strip()}:
                continue
            for seed in seeds:
                params = BondRouteParams(seed=seed, n_agents=n_agents, **overrides)
                active_route_names = (
                    full_process_route_names
                    if variant_name == "full_process_extended"
                    else long_run_route_names
                    if variant_name in {"bridge_long_run_probe", "bridge_moderate_rebond_probe"}
                    else route_names
                )
                results = {route: simulate_route(route, params) for route in active_route_names}
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
                    for key, value in compute_process_mirror_metrics(rows, params.t_end).items():
                        row[key] = float(value)
                    for key, value in compute_bridge_time_metrics(rows, params.t_end).items():
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
            active_route_names = (
                full_process_route_names
                if variant_name == "full_process_extended"
                else long_run_route_names
                if variant_name in {"bridge_long_run_probe", "bridge_moderate_rebond_probe"}
                else route_names
            )
            for route in active_route_names:
                route_rows = [
                    row
                    for row in per_seed_rows
                    if row["route"] == route and row["variant"] == variant_name and row["n_agents"] == float(n_agents)
                ]
                if not route_rows:
                    continue
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
            if not route_rows:
                continue
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
        colors = {
            "drift": "#2e8b57",
            "shock": "#c44e52",
            "distortion": "#4c72b0",
            "shock_drift": "#dd8452",
            "full_process": "#8c6d31",
        }
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
