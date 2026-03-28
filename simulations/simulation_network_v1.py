"""
Network simulation for Meta-Stable Architectures (multi-agent, no metronome).

Usage:
    python simulations/simulation_network_v1.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import os

import numpy as np

try:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class MasterParams:
    # Volume I / II core (used in thresholds)
    c0: float = 0.45
    mu_k: float = 0.25
    zeta: float = 0.55

    # Volume III memory block
    k: float = 0.8
    lam: float = 0.25
    alpha: float = 0.9
    gamma: float = 0.85
    sigma_h: float = 0.35
    delta_h: float = 0.45
    eta_s: float = 0.55
    beta: float = 0.25
    rho1: float = 0.4
    rho2: float = 0.3
    rho3: float = 0.35
    delta_k: float = 0.3
    eps_s: float = 0.6
    h_crit: float = 0.7
    sigma_noise: float = 0.22
    xi: float = 0.9
    iso_cool: float = 1.2
    iso_noise_scale: float = 0.2
    eps_s2: float = 0.25
    c_crit: float = 0.6


@dataclass
class NetworkParams:
    n_agents: int = 32
    t_end: float = 120.0
    dt: float = 0.01
    coupling: float = 0.2
    ring_k: int = 2  # neighbors on each side
    seed: int = 42
    topology: str = "ring"  # ring | erdos_renyi | small_world | scale_free | spatial_local
    er_p: float = 0.1
    sw_rewire: float = 0.1
    ba_m: int = 2
    spatial_radius: float = 0.35
    spatial_falloff: float = 0.18
    instructor_position: tuple[float, float] = (0.5, 0.5)
    visibility_falloff: float = 0.45
    hearing_falloff: float = 0.60
    delay_mode: str = "fixed"  # fixed | grouped
    delay_steps: int = 0
    delay_group_fracs: tuple[float, ...] = (0.5, 0.5)
    delay_group_steps: tuple[int, ...] = (2, 6)

    stress_time: float = 30.0
    stress_amp: float = 3.0
    stress_frac: float = 0.5
    stress_duration: float = 4.0
    stress_y_amp: float = 1.0
    stress_windows: tuple[tuple[float, float], ...] | None = None
    y_cap: float = 3.5

    iso_threshold: float = 0.8
    recovery_threshold: float = 0.2
    metro_amp: float = 0.0
    metro_freq: float = 1.1
    metro_phase: float = 0.0
    # QRC / reflexive layer
    qrc_enabled: bool = False
    phi_kappa: float = 1.0
    phi_gain: float = 0.15
    qrc_eta: float = 0.6
    qrc_g_min: float = 0.3
    qrc_g_max: float = 1.2
    phi_gain_boost: float = 4.0
    phi_listen_isolated: bool = False
    phi_iso_threshold: float = 0.5
    recog_threshold: float = 0.7
    recog_gain: float = 1.2
    wake_disp_threshold: float = 0.3
    wake_time_required: float = 4.0
    wake_relax_gain: float = 0.8
    coh_relax_gain: float = 0.6
    instructor_enabled: bool = False
    instructor_response_mode: str = "generic"  # generic | noise | lag | contagion
    instructor_error_weight: float = 0.7
    instructor_anchor_weight: float = 1.0
    instructor_contagion_weight: float = 1.0
    instructor_template_weight: float = 0.7
    template_multimode_enabled: bool = False
    template_focus_gain: float = 1.25
    template_recovery_gain: float = 0.85
    template_stabilize_gain: float = 0.92
    template_turbulence_trigger: float = 0.28
    template_recovery_trigger: float = 0.12
    instructor_phi_scale_noise: float = 0.9
    instructor_phi_scale_lag: float = 1.1
    instructor_phi_scale_contagion: float = 0.95
    instructor_anchor_scale_noise: float = 0.8
    instructor_anchor_scale_lag: float = 1.35
    instructor_anchor_scale_contagion: float = 1.0
    instructor_contagion_scale_noise: float = 0.9
    instructor_contagion_scale_lag: float = 0.9
    instructor_contagion_scale_contagion: float = 1.6
    focus_lock_enabled: bool = False
    focus_lock_trigger: float = 0.35
    focus_lock_strength: float = 0.65
    focus_lock_anchor_boost: float = 0.9
    turbulence_error_weight: float = 0.55
    cascade_threshold: float = 0.35
    reroute_enabled: bool = False
    reroute_strength: float = 0.9
    reroute_instructor_weight: float = 1.0
    reroute_anchor_weight: float = 1.0
    three_channel_enabled: bool = False
    three_channel_adaptive: bool = False
    channel_peer_base: float = 0.55
    channel_instructor_base: float = 0.30
    channel_anchor_base: float = 0.15
    channel_turbulence_gain: float = 0.60
    channel_attunement_gain: float = 0.45
    channel_anchor_gain: float = 0.40
    demper_enabled: bool = False
    demper_load_gain: float = 0.90
    demper_decay: float = 0.35
    demper_relax_gain: float = 0.75
    demper_trigger: float = 0.18
    forewarning_enabled: bool = False
    forewarning_gain: float = 0.80
    forewarning_decay: float = 0.28
    forewarning_trigger: float = 0.34
    forewarning_phase_floor: float = 0.18
    forewarning_phase_ceiling: float = 0.62
    forewarning_s_ceiling: float = 0.68
    pause_peer_suppress: float = 0.55
    pause_instructor_boost: float = 0.45
    pause_anchor_boost: float = 0.30
    pause_slow_gain: float = 0.22
    pause_relax_gain: float = 0.28
    pause_attention_mode: bool = False
    pause_attention_focus_gain: float = 0.55
    pause_attention_template_gain: float = 1.18
    pause_attention_phi_gain: float = 0.35
    instructor_recognition_enabled: bool = False
    recognition_rate: float = 0.30
    recognition_decay: float = 0.06
    recognition_similarity_gain: float = 7.5
    recognition_threshold: float = 0.48
    recognition_attention_boost: float = 0.40
    attunement_enabled: bool = False
    attunement_gain: float = 0.03
    attunement_decay: float = 0.005
    attunement_mode: str = "scalar"  # scalar | multi
    initial_attunement: float | np.ndarray = 0.0
    kg_enabled: bool = False
    kg_lambda: float = 0.05
    kg_decay: float = 0.02
    kg_decay_min: float | None = None
    kg_decay_max: float | None = None
    kg_decay_stateful: bool = False
    kg_phi_boost: float = 1.5
    kg_wake_boost: float = 1.0
    kg_tail_boost: float = 1.0
    kg_lag_boost: float = 1.0
    kg_floor: float = 0.0
    kg_cap: float = 1.0
    kg_crisis_threshold: float = 0.6
    initial_group_memory: float = 0.0
    initial_phi: float = 0.0
    ccrit_gain: float = 0.6
    ccrit_floor: float = 0.2
    ccrit_cap: float = 1.2


def _build_ring_adjacency(n: int, k: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        for offset in range(1, k + 1):
            a[i, (i + offset) % n] = 1.0
            a[i, (i - offset) % n] = 1.0
    return a


def _build_erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    a = rng.random((n, n)) < p
    np.fill_diagonal(a, 0)
    a = np.triu(a, 1)
    a = a + a.T
    return a.astype(float)


def _build_small_world(n: int, k: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    a = _build_ring_adjacency(n, k)
    for i in range(n):
        for offset in range(1, k + 1):
            j = (i + offset) % n
            if rng.random() < beta:
                a[i, j] = 0.0
                a[j, i] = 0.0
                candidates = [x for x in range(n) if x != i and a[i, x] == 0.0]
                if candidates:
                    new_j = rng.choice(candidates)
                    a[i, new_j] = 1.0
                    a[new_j, i] = 1.0
                else:
                    a[i, j] = 1.0
                    a[j, i] = 1.0
    return a


def _build_scale_free(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    m = max(1, min(m, n - 1))
    a = np.zeros((n, n), dtype=float)
    core = m + 1
    for i in range(core):
        for j in range(i + 1, core):
            a[i, j] = 1.0
            a[j, i] = 1.0
    degrees = a.sum(axis=1)
    for new in range(core, n):
        probs = degrees[:new]
        if probs.sum() == 0:
            targets = rng.choice(new, size=m, replace=False)
        else:
            probs = probs / probs.sum()
            targets = rng.choice(new, size=m, replace=False, p=probs)
        for t in targets:
            a[new, t] = 1.0
            a[t, new] = 1.0
        degrees = a.sum(axis=1)
    return a


def _build_spatial_local(n: int, radius: float, falloff: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positions = rng.random((n, 2))
    dists = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2))
    a = np.exp(-dists / max(falloff, 1e-6)) * (dists <= radius)
    np.fill_diagonal(a, 0.0)
    for i in range(n):
        if np.all(a[i] == 0.0):
            nearest = np.argsort(dists[i])[1]
            w = math.exp(-float(dists[i, nearest]) / max(falloff, 1e-6))
            a[i, nearest] = w
            a[nearest, i] = max(a[nearest, i], w)
    return a, positions


def _circular_variance(theta: np.ndarray) -> float:
    mean_vec = np.mean(np.exp(1j * theta))
    return float(1.0 - np.abs(mean_vec))


def _response_scales(net: NetworkParams) -> tuple[float, float, float]:
    if net.instructor_response_mode == "noise":
        return (
            net.instructor_phi_scale_noise,
            net.instructor_anchor_scale_noise,
            net.instructor_contagion_scale_noise,
        )
    if net.instructor_response_mode == "lag":
        return (
            net.instructor_phi_scale_lag,
            net.instructor_anchor_scale_lag,
            net.instructor_contagion_scale_lag,
        )
    if net.instructor_response_mode == "contagion":
        return (
            net.instructor_phi_scale_contagion,
            net.instructor_anchor_scale_contagion,
            net.instructor_contagion_scale_contagion,
        )
    return (1.0, 1.0, 1.0)


def simulate_network(
    p: MasterParams | None = None,
    net: NetworkParams | None = None,
) -> dict[str, np.ndarray]:
    if p is None:
        p = MasterParams()
    if net is None:
        net = NetworkParams()

    rng = np.random.default_rng(net.seed)
    n_steps = int(net.t_end / net.dt) + 1
    t = np.linspace(0.0, net.t_end, n_steps)

    n = net.n_agents
    positions = np.zeros((n, 2))
    response_phi_scale, response_anchor_scale, response_contagion_scale = _response_scales(net)
    if net.topology == "ring":
        a = _build_ring_adjacency(n, net.ring_k)
    elif net.topology == "erdos_renyi":
        a = _build_erdos_renyi(n, net.er_p, rng)
    elif net.topology == "small_world":
        a = _build_small_world(n, net.ring_k, net.sw_rewire, rng)
    elif net.topology == "scale_free":
        a = _build_scale_free(n, net.ba_m, rng)
    elif net.topology == "spatial_local":
        a, positions = _build_spatial_local(n, net.spatial_radius, net.spatial_falloff, rng)
    else:
        raise ValueError(f"Unknown topology: {net.topology}")
    deg = a.sum(axis=1)
    deg[deg == 0.0] = 1.0
    instructor_pos = np.array(net.instructor_position, dtype=float)
    coord_dist = np.sqrt(((positions - instructor_pos) ** 2).sum(axis=1))
    visibility_quality = np.exp(-coord_dist / max(net.visibility_falloff, 1e-6))
    hearing_quality = np.exp(-coord_dist / max(net.hearing_falloff, 1e-6))
    access_quality = np.clip(0.5 * (visibility_quality + hearing_quality), 0.0, 1.0)

    # State
    x = np.zeros((n_steps, n))
    y = np.zeros((n_steps, n))
    h = np.zeros((n_steps, n))
    k_struct = np.zeros((n_steps, n))
    mu = np.zeros((n_steps, n))
    s_buf = np.zeros((n_steps, n))
    demper_load = np.zeros((n_steps, n))
    forewarning = np.zeros((n_steps, n))
    recognition_signal = np.zeros(n_steps)
    recognition_signature = np.zeros((n_steps, 6))
    phi = np.zeros(n_steps)
    attunement = np.zeros((n_steps, n))
    attunement_rhythm = np.zeros((n_steps, n))
    attunement_focus = np.zeros((n_steps, n))
    attunement_risk = np.zeros((n_steps, n))

    x[0] = 0.2 + 0.02 * rng.normal(size=n)
    y[0] = 0.1 + 0.02 * rng.normal(size=n)
    h[0] = 0.05
    k_struct[0] = 0.0
    mu[0] = 0.1
    s_buf[0] = 0.05
    phi[0] = net.initial_phi
    if isinstance(net.initial_attunement, np.ndarray):
        attunement[0] = net.initial_attunement.astype(float)
    else:
        attunement[0] = float(net.initial_attunement)
    attunement_rhythm[0] = attunement[0]
    attunement_focus[0] = attunement[0]
    attunement_risk[0] = attunement[0]

    stress_agents = rng.choice(n, size=max(1, int(n * net.stress_frac)), replace=False)
    if net.stress_windows is None or len(net.stress_windows) == 0:
        stress_windows: tuple[tuple[float, float], ...] = ((net.stress_time, net.stress_duration),)
    else:
        stress_windows = net.stress_windows

    sqrt_dt = math.sqrt(net.dt)

    phase_disp = np.zeros(n_steps)
    frac_iso = np.zeros(n_steps)
    mean_h = np.zeros(n_steps)
    calm_time = np.zeros(n_steps)
    group_memory = np.zeros(n_steps)
    mean_s = np.zeros(n_steps)
    phi_gain_hist = np.zeros(n_steps)
    kappa_hist = np.zeros(n_steps)
    error_hist = np.zeros((n_steps, n))
    anchor_hist = np.zeros((n_steps, n))
    mean_error = np.zeros(n_steps)
    mean_anchor = np.zeros(n_steps)
    turbulence_hist = np.zeros((n_steps, n))
    mean_turbulence = np.zeros(n_steps)
    cascade_fraction = np.zeros(n_steps)
    focus_lock_hist = np.zeros((n_steps, n))
    mean_focus_lock = np.zeros(n_steps)
    channel_peer_hist = np.zeros((n_steps, n))
    channel_instructor_hist = np.zeros((n_steps, n))
    channel_anchor_hist = np.zeros((n_steps, n))
    mean_channel_peer = np.zeros(n_steps)
    mean_channel_instructor = np.zeros(n_steps)
    mean_channel_anchor = np.zeros(n_steps)
    template_mode_hist = np.zeros(n_steps)
    mean_demper_load = np.zeros(n_steps)
    mean_forewarning = np.zeros(n_steps)
    mean_pause = np.zeros(n_steps)
    mean_recognition = np.zeros(n_steps)
    mean_access = np.zeros(n_steps)
    mean_attunement = np.zeros(n_steps)
    group_memory[0] = float(np.clip(net.initial_group_memory, net.kg_floor, net.kg_cap))

    # Delay profile (for grouped delays)
    delay_per_agent = np.zeros(n, dtype=int)
    if net.delay_mode == "grouped" and sum(net.delay_group_fracs) > 0:
        fracs = np.array(net.delay_group_fracs, dtype=float)
        fracs = fracs / fracs.sum()
        steps = np.array(net.delay_group_steps, dtype=int)
        groups = rng.choice(len(fracs), size=n, p=fracs)
        delay_per_agent = steps[groups]

    for idx in range(n_steps - 1):
        theta = np.arctan2(y[idx], x[idx])
        phase_disp_now = _circular_variance(theta)
        frac_iso_now = float(np.mean(s_buf[idx] >= net.iso_threshold))
        mean_h_now = float(np.mean(h[idx]))
        mean_s_now = float(np.mean(s_buf[idx]))
        if phase_disp_now < net.wake_disp_threshold:
            calm_now = (calm_time[idx - 1] + net.dt) if idx > 0 else net.dt
        else:
            calm_now = 0.0
        calm_time[idx] = calm_now
        phase_disp[idx] = phase_disp_now
        frac_iso[idx] = frac_iso_now
        mean_h[idx] = mean_h_now
        mean_s[idx] = mean_s_now
        mean_access[idx] = float(np.mean(access_quality))
        mean_attunement[idx] = float(np.mean(attunement[idx]))
        mean_demper_load[idx] = float(np.mean(demper_load[idx]))
        mean_forewarning[idx] = float(np.mean(forewarning[idx]))
        current_pause_level = np.clip((forewarning[idx] - 0.15) / 0.85, 0.0, 1.0) if net.forewarning_enabled else np.zeros(n)
        match_now = 0.5 * (1.0 + np.cos(theta - phi[idx]))
        if net.attunement_enabled:
            if net.attunement_mode == "multi":
                effective_attunement = np.clip(
                    0.45 * attunement_rhythm[idx] + 0.30 * attunement_focus[idx] + 0.25 * attunement_risk[idx],
                    0.0,
                    1.0,
                )
            else:
                effective_attunement = np.clip(attunement[idx], 0.0, 1.0)
        else:
            effective_attunement = np.zeros(n)
        template_scale = 1.0
        template_mode_value = 0.0
        if net.template_multimode_enabled:
            if net.instructor_response_mode == "contagion":
                template_scale = net.template_stabilize_gain
                template_mode_value = 2.0
            elif mean_s_now >= net.template_turbulence_trigger:
                template_scale = net.template_focus_gain
                template_mode_value = 1.0
            elif phase_disp_now <= net.template_recovery_trigger and mean_s_now <= net.template_recovery_trigger:
                template_scale = net.template_recovery_gain
                template_mode_value = -1.0
        if net.forewarning_enabled and net.pause_attention_mode:
            template_scale *= (1.0 + net.pause_attention_template_gain * current_pause_level)
        template_mode_hist[idx] = template_mode_value
        template_y = template_scale * np.sin(phi[idx]) * (1.0 - 0.5 * s_buf[idx])
        template_mismatch = np.abs(y[idx] - template_y) / (0.25 + np.abs(template_y) + np.std(y[idx]))
        template_mismatch = np.clip(template_mismatch, 0.0, 1.0)
        neighbor_ref = (a @ y[idx]) / deg
        local_mismatch = np.abs(y[idx] - neighbor_ref) / (0.25 + np.abs(neighbor_ref) + float(np.std(y[idx])))
        local_mismatch = np.clip(local_mismatch, 0.0, 1.0)
        base_error = (
            net.instructor_template_weight * template_mismatch
            + (1.0 - net.instructor_template_weight)
            * (net.instructor_error_weight * (1.0 - match_now) + (1.0 - net.instructor_error_weight) * local_mismatch)
        )
        if net.attunement_enabled:
            base_error *= (1.0 - 0.55 * effective_attunement)
        error_now = np.clip(0.8 * base_error + 0.2 * s_buf[idx], 0.0, 1.0)
        error_now = np.clip(error_now * (1.15 - 0.45 * access_quality), 0.0, 1.0)
        anchor_now = np.clip(
            access_quality * (1.0 - s_buf[idx]) * (1.0 - error_now) ** 2 * (1.0 + 0.5 * effective_attunement),
            0.0,
            1.0,
        )
        error_hist[idx] = error_now
        anchor_hist[idx] = anchor_now
        mean_error[idx] = float(np.mean(error_now))
        mean_anchor[idx] = float(np.mean(anchor_now))
        neighbor_error = (a @ error_now) / deg
        turbulence_now = np.clip(
            net.turbulence_error_weight * error_now + (1.0 - net.turbulence_error_weight) * neighbor_error,
            0.0,
            1.0,
        )
        focus_lock_now = np.zeros(n)
        if net.focus_lock_enabled and net.instructor_enabled:
            focus_lock_now = np.clip(
                (turbulence_now - net.focus_lock_trigger) / max(1e-6, 1.0 - net.focus_lock_trigger),
                0.0,
                1.0,
            )
        turbulence_hist[idx] = turbulence_now
        mean_turbulence[idx] = float(np.mean(turbulence_now))
        cascade_fraction[idx] = float(np.mean(turbulence_now >= net.cascade_threshold))
        focus_lock_hist[idx] = focus_lock_now
        mean_focus_lock[idx] = float(np.mean(focus_lock_now))
        signature_now = np.array(
            [
                float(np.clip(phase_disp_now, 0.0, 1.0)),
                float(np.clip(np.mean(error_now), 0.0, 1.0)),
                float(np.clip(np.mean(turbulence_now), 0.0, 1.0)),
                float(np.clip(np.mean(demper_load[idx]), 0.0, 1.0)),
                float(np.clip(0.5 - np.mean(attunement[idx] - attunement[max(0, idx - 1)]), 0.0, 1.0)),
                float(np.clip(np.mean(turbulence_now >= net.cascade_threshold), 0.0, 1.0)),
            ],
            dtype=float,
        )
        recognition_signature[idx] = signature_now
        if net.instructor_recognition_enabled:
            motif_strength = 0.5 * signature_now[1] + 0.5 * signature_now[2]
            if idx > 3 and motif_strength >= net.recognition_threshold:
                candidate_pool = recognition_signature[:idx]
                distances = np.linalg.norm(candidate_pool - signature_now[None, :], axis=1)
                best_distance = float(np.min(distances))
                similarity = math.exp(-net.recognition_similarity_gain * best_distance)
            else:
                similarity = 0.0
            d_recognition = net.recognition_rate * similarity * (1.0 - recognition_signal[idx])
            d_recognition -= net.recognition_decay * (1.0 - similarity) * recognition_signal[idx]
            recognition_signal[idx + 1] = float(np.clip(recognition_signal[idx] + net.dt * d_recognition, 0.0, 1.0))
        else:
            recognition_signal[idx + 1] = recognition_signal[idx]
        mean_recognition[idx] = recognition_signal[idx]
        if net.forewarning_enabled:
            phase_window = np.clip(
                (phase_disp_now - net.forewarning_phase_floor) / max(1e-6, net.forewarning_phase_ceiling - net.forewarning_phase_floor),
                0.0,
                1.0,
            )
            suppression_window = np.clip(
                (net.forewarning_s_ceiling - mean_s_now) / max(1e-6, net.forewarning_s_ceiling),
                0.0,
                1.0,
            )
            forewarning_drive = np.clip(
                0.35 * error_now
                + 0.35 * turbulence_now
                + 0.20 * demper_load[idx]
                + 0.10 * (1.0 - effective_attunement)
                - net.forewarning_trigger,
                0.0,
                1.0,
            )
            forewarning_drive *= phase_window * suppression_window
            d_forewarning = net.forewarning_gain * forewarning_drive * (1.0 - forewarning[idx])
            d_forewarning -= net.forewarning_decay * (1.0 - forewarning_drive) * forewarning[idx]
            forewarning[idx + 1] = np.clip(forewarning[idx] + net.dt * d_forewarning, 0.0, 1.0)
            pause_level = np.clip((forewarning[idx] - 0.15) / 0.85, 0.0, 1.0)
            if net.instructor_recognition_enabled:
                pause_level = np.clip(pause_level + net.recognition_attention_boost * recognition_signal[idx], 0.0, 1.0)
        else:
            forewarning[idx + 1] = forewarning[idx]
            pause_level = np.zeros(n)
        mean_pause[idx] = float(np.mean(pause_level))

        # Reflexive phase (global mediator)
        if net.qrc_enabled:
            if net.phi_listen_isolated:
                mask = s_buf[idx] > net.phi_iso_threshold
                if np.any(mask):
                    u_global = float(np.arctan2(np.mean(np.sin(theta[mask])), np.mean(np.cos(theta[mask]))))
                else:
                    u_global = float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))
            else:
                u_global = float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))
            phi[idx + 1] = phi[idx] + net.dt * (-net.phi_kappa * phi[idx] + u_global)
            kappa_hist[idx] = net.phi_kappa
            c_local = y[idx] ** 2
            c_mean = float(np.mean(c_local))
            g = max(net.qrc_g_min, min(net.qrc_g_max, 1.0 - net.qrc_eta * c_mean))
            kg_now = group_memory[idx] if net.kg_enabled else 0.0
            phi_gain = net.phi_gain * (1.0 + net.phi_gain_boost * (mean_s_now ** 2) + net.kg_phi_boost * kg_now)
            if net.instructor_enabled:
                phi_gain *= response_phi_scale
            if net.forewarning_enabled and net.pause_attention_mode:
                phi_gain *= (1.0 + net.pause_attention_phi_gain * np.mean(current_pause_level))
        else:
            phi[idx + 1] = phi[idx]
            g = 1.0
            phi_gain = 0.0
            kappa_hist[idx] = 0.0
        phi_gain_hist[idx] = phi_gain

        # Coupling term (isolation-gated) with optional delays
        if net.delay_mode == "fixed" and net.delay_steps > 0:
            src = y[max(0, idx - net.delay_steps)]
        elif net.delay_mode == "grouped" and np.any(delay_per_agent > 0):
            src = y[idx].copy()
            unique_delays = np.unique(delay_per_agent)
            for d in unique_delays:
                if d == 0:
                    continue
                mask = delay_per_agent == d
                src[mask] = y[max(0, idx - d)][mask]
        else:
            src = y[idx]
        if net.instructor_enabled:
            instructor_ref = template_y
            peer_gate = (1.0 - response_contagion_scale * net.instructor_contagion_weight * error_now)
            if net.focus_lock_enabled:
                peer_gate *= (1.0 - net.focus_lock_strength * focus_lock_now)
            if net.forewarning_enabled:
                peer_gate *= (1.0 - net.pause_peer_suppress * pause_level)
            anchor_gate = (1.0 + response_anchor_scale * net.instructor_anchor_weight * anchor_now)
            if net.focus_lock_enabled:
                anchor_gate *= (1.0 + net.focus_lock_anchor_boost * focus_lock_now * anchor_now)
            if net.forewarning_enabled:
                anchor_gate *= (1.0 + net.pause_anchor_boost * pause_level)
            eff_adj = (
                a
                * np.clip(peer_gate, 0.0, 1.0)[None, :] ** 2
                * np.clip(anchor_gate, 0.0, None)[None, :]
            )
            eff_adj = np.clip(eff_adj, 0.0, None)
            eff_deg = eff_adj.sum(axis=1)
            eff_deg[eff_deg == 0.0] = 1.0
            y_neighbor = eff_adj @ src
            anchor_weighted_adj = eff_adj * anchor_now[None, :]
            anchor_weight_sum = anchor_weighted_adj.sum(axis=1)
            safe_anchor_weight_sum = anchor_weight_sum.copy()
            safe_anchor_weight_sum[safe_anchor_weight_sum == 0.0] = 1.0
            anchor_ref_local = (anchor_weighted_adj @ src) / safe_anchor_weight_sum
            if np.any(anchor_weight_sum == 0.0):
                anchor_ref_local[anchor_weight_sum == 0.0] = y_neighbor[anchor_weight_sum == 0.0] / eff_deg[anchor_weight_sum == 0.0]
            peer_term = g * net.coupling * (1.0 - s_buf[idx]) * (y_neighbor - eff_deg * y[idx]) / eff_deg
            if net.three_channel_enabled:
                if net.three_channel_adaptive:
                    w_peer = np.clip(
                        net.channel_peer_base * (1.0 - net.channel_turbulence_gain * turbulence_now),
                        0.05,
                        1.0,
                    )
                    w_instr = np.clip(
                        net.channel_instructor_base
                        + net.channel_turbulence_gain * turbulence_now
                        + net.channel_attunement_gain * effective_attunement,
                        0.05,
                        2.0,
                    )
                    w_anchor = np.clip(
                        net.channel_anchor_base
                        + net.channel_anchor_gain * anchor_now
                        + 0.5 * net.channel_turbulence_gain * turbulence_now,
                        0.05,
                        2.0,
                    )
                    if net.forewarning_enabled:
                        w_peer = np.clip(w_peer * (1.0 - net.pause_peer_suppress * pause_level), 0.02, 1.0)
                        w_instr = np.clip(w_instr + net.pause_instructor_boost * pause_level, 0.05, 2.5)
                        w_anchor = np.clip(w_anchor + net.pause_anchor_boost * pause_level, 0.05, 2.5)
                        if net.pause_attention_mode:
                            w_instr = np.clip(w_instr + net.pause_attention_focus_gain * pause_level, 0.05, 3.0)
                else:
                    w_peer = np.full(n, net.channel_peer_base)
                    w_instr = np.full(n, net.channel_instructor_base)
                    w_anchor = np.full(n, net.channel_anchor_base)
                w_sum = w_peer + w_instr + w_anchor
                w_peer = w_peer / w_sum
                w_instr = w_instr / w_sum
                w_anchor = w_anchor / w_sum
                channel_peer_hist[idx] = w_peer
                channel_instructor_hist[idx] = w_instr
                channel_anchor_hist[idx] = w_anchor
                mean_channel_peer[idx] = float(np.mean(w_peer))
                mean_channel_instructor[idx] = float(np.mean(w_instr))
                mean_channel_anchor[idx] = float(np.mean(w_anchor))
                three_channel_ref = (
                    w_peer * (y_neighbor / eff_deg)
                    + w_instr * instructor_ref
                    + w_anchor * anchor_ref_local
                )
                coupling_term = g * net.coupling * (1.0 - s_buf[idx]) * (three_channel_ref - y[idx])
            elif net.reroute_enabled:
                reroute_level = focus_lock_now if net.focus_lock_enabled else np.clip(
                    (turbulence_now - net.focus_lock_trigger) / max(1e-6, 1.0 - net.focus_lock_trigger),
                    0.0,
                    1.0,
                )
                reroute_term = g * net.coupling * (1.0 - s_buf[idx]) * reroute_level * net.reroute_strength * (
                    net.reroute_instructor_weight * (instructor_ref - y[idx])
                    + net.reroute_anchor_weight * anchor_now * (anchor_ref_local - y[idx])
                )
                coupling_term = (1.0 - reroute_level) * peer_term + reroute_term
            else:
                coupling_term = peer_term
        else:
            y_neighbor = a @ (src * (1.0 - s_buf[idx]))
            coupling_term = g * net.coupling * (1.0 - s_buf[idx]) * (y_neighbor - deg * y[idx]) / deg
        if net.qrc_enabled and phi_gain != 0.0:
            coupling_term += phi_gain * (phi[idx] - y[idx]) * (1.0 - s_buf[idx])

        dx = -p.k * x[idx] - p.lam * y[idx]
        alpha = p.alpha
        if isinstance(alpha, np.ndarray):
            alpha = alpha
        else:
            alpha = np.full(n, float(alpha))
        drift_y = alpha * x[idx] + mu[idx] * y[idx] - p.gamma * y[idx] ** 3
        if net.forewarning_enabled:
            drift_y *= (1.0 - net.pause_slow_gain * pause_level)
        metro = net.metro_amp * math.sin(net.metro_freq * t[idx] + net.metro_phase)
        sigma_noise = p.sigma_noise
        if isinstance(sigma_noise, np.ndarray):
            sigma_noise = sigma_noise
        else:
            sigma_noise = np.full(n, float(sigma_noise))
        noise_scale = sigma_noise * (p.iso_noise_scale + (1.0 - p.iso_noise_scale) * (1.0 - s_buf[idx]))
        if net.forewarning_enabled:
            noise_scale *= (1.0 - 0.5 * net.pause_slow_gain * pause_level)
        noise = noise_scale * sqrt_dt * rng.normal(size=n)

        dH = p.sigma_h * y[idx] ** 2 - (p.delta_h + p.eta_s * s_buf[idx] + p.iso_cool * s_buf[idx]) * h[idx]
        dK = p.beta * h[idx] - p.delta_k * k_struct[idx] + p.xi * s_buf[idx] * h[idx]
        dmu = p.rho1 * h[idx] - p.rho2 * k_struct[idx] - p.rho3 * mu[idx]
        c_local = y[idx] ** 2
        ccrit_eff = p.c_crit
        if net.qrc_enabled:
            ccrit_eff = p.c_crit + net.ccrit_gain * (1.0 - phase_disp_now)
            ccrit_eff = float(np.clip(ccrit_eff, net.ccrit_floor, net.ccrit_cap))
        h_crit = p.h_crit
        if isinstance(h_crit, np.ndarray):
            h_crit = h_crit
        else:
            h_crit = np.full(n, float(h_crit))
        dS = p.eps_s * (h[idx] - h_crit) * s_buf[idx] * (1.0 - s_buf[idx])
        dS += p.eps_s2 * (c_local - ccrit_eff) * s_buf[idx] * (1.0 - s_buf[idx])
        if net.qrc_enabled:
            match = np.cos(theta - phi[idx])
            recog = np.maximum(0.0, match - net.recog_threshold)
            dS -= net.recog_gain * recog * s_buf[idx] * (1.0 - s_buf[idx])
            wake_relax_gain = net.wake_relax_gain
            if net.kg_enabled:
                wake_relax_gain *= (1.0 + net.kg_wake_boost * (group_memory[idx] if net.kg_enabled else 0.0))
            if calm_now >= net.wake_time_required:
                dS -= wake_relax_gain * s_buf[idx] * (1.0 - s_buf[idx])
            dS -= net.coh_relax_gain * (1.0 - phase_disp_now) * s_buf[idx]
        if net.forewarning_enabled:
            dS -= net.pause_relax_gain * pause_level * (1.0 - phase_disp_now) * s_buf[idx]
        if net.demper_enabled:
            demper_drive = np.clip(error_now + 0.5 * turbulence_now + 0.5 * s_buf[idx] - net.demper_trigger, 0.0, 1.0)
            d_dem = net.demper_load_gain * demper_drive * (1.0 - demper_load[idx])
            d_dem -= net.demper_decay * (1.0 - mean_s_now) * demper_load[idx]
            demper_load[idx + 1] = np.clip(demper_load[idx] + net.dt * d_dem, 0.0, 1.0)
            dS -= net.demper_relax_gain * demper_load[idx] * (1.0 - phase_disp_now) * s_buf[idx]
        else:
            demper_load[idx + 1] = demper_load[idx]

        x[idx + 1] = x[idx] + net.dt * dx
        y[idx + 1] = y[idx] + net.dt * (drift_y + coupling_term + metro) + noise
        if net.y_cap is not None:
            y[idx + 1] = np.clip(y[idx + 1], -net.y_cap, net.y_cap)
        h[idx + 1] = np.maximum(0.0, h[idx] + net.dt * dH)
        k_struct[idx + 1] = np.maximum(0.0, k_struct[idx] + net.dt * dK)
        mu[idx + 1] = mu[idx] + net.dt * dmu
        s_buf[idx + 1] = np.clip(s_buf[idx] + net.dt * dS, 0.0, 1.0)
        if net.attunement_enabled:
            if net.attunement_mode == "multi":
                d_rhythm = net.attunement_gain * access_quality * match_now * (1.0 - s_buf[idx]) * (1.0 - attunement_rhythm[idx])
                d_focus = net.attunement_gain * access_quality * (1.0 - error_now) * (1.0 - s_buf[idx]) * (1.0 - attunement_focus[idx])
                d_risk = net.attunement_gain * access_quality * (1.0 - mean_s_now) * (1.0 - error_now) * (1.0 - attunement_risk[idx])
                d_rhythm -= net.attunement_decay * s_buf[idx] * attunement_rhythm[idx]
                d_focus -= net.attunement_decay * s_buf[idx] * attunement_focus[idx]
                d_risk -= 1.5 * net.attunement_decay * s_buf[idx] * attunement_risk[idx]
                attunement_rhythm[idx + 1] = np.clip(attunement_rhythm[idx] + net.dt * d_rhythm, 0.0, 1.0)
                attunement_focus[idx + 1] = np.clip(attunement_focus[idx] + net.dt * d_focus, 0.0, 1.0)
                attunement_risk[idx + 1] = np.clip(attunement_risk[idx] + net.dt * d_risk, 0.0, 1.0)
                attunement[idx + 1] = np.clip(
                    0.45 * attunement_rhythm[idx + 1] + 0.30 * attunement_focus[idx + 1] + 0.25 * attunement_risk[idx + 1],
                    0.0,
                    1.0,
                )
            else:
                d_attune = net.attunement_gain * access_quality * match_now * (1.0 - s_buf[idx]) * (1.0 - attunement[idx])
                d_attune -= net.attunement_decay * s_buf[idx] * attunement[idx]
                attunement[idx + 1] = np.clip(attunement[idx] + net.dt * d_attune, 0.0, 1.0)
                attunement_rhythm[idx + 1] = attunement[idx + 1]
                attunement_focus[idx + 1] = attunement[idx + 1]
                attunement_risk[idx + 1] = attunement[idx + 1]
        else:
            attunement[idx + 1] = attunement[idx]
            attunement_rhythm[idx + 1] = attunement_rhythm[idx]
            attunement_focus[idx + 1] = attunement_focus[idx]
            attunement_risk[idx + 1] = attunement_risk[idx]

        in_stress = False
        for start_t, duration in stress_windows:
            stress_idx = int(start_t / net.dt)
            stress_steps = max(1, int(duration / net.dt))
            if stress_idx <= idx < stress_idx + stress_steps:
                in_stress = True
                break
        if in_stress:
            h[idx + 1, stress_agents] += net.stress_amp
            y[idx + 1, stress_agents] += net.stress_y_amp

        if net.kg_enabled:
            lag_proxy = float(np.clip(calm_now / max(net.wake_time_required, 1e-6), 0.0, 1.0))
            kg_score = (
                0.25 * phase_disp_now
                + 0.25 * frac_iso_now
                + 0.20 * (mean_h_now / (1.0 + mean_h_now))
                + 0.20 * mean_s_now
                + 0.10 * lag_proxy
            )
            kg_score += 0.10 * net.kg_tail_boost * float(np.clip(mean_s_now, 0.0, 1.0))
            kg_score += 0.05 * net.kg_lag_boost * lag_proxy
            kg_decay_eff = net.kg_decay
            if net.kg_decay_stateful:
                decay_min = net.kg_decay_min if net.kg_decay_min is not None else net.kg_decay
                decay_max = net.kg_decay_max if net.kg_decay_max is not None else decay_min
                memory_level = float(np.clip(group_memory[idx] / max(net.kg_cap, 1e-12), 0.0, 1.0))
                kg_decay_eff = float(decay_min + (decay_max - decay_min) * memory_level)
            kg_increment = net.kg_lambda * kg_score if kg_score >= net.kg_crisis_threshold else 0.0
            group_memory[idx + 1] = float(
                np.clip((1.0 - kg_decay_eff) * group_memory[idx] + kg_increment, net.kg_floor, net.kg_cap)
            )
        else:
            group_memory[idx + 1] = group_memory[idx]

    theta = np.arctan2(y[-1], x[-1])
    phase_disp[-1] = _circular_variance(theta)
    frac_iso[-1] = float(np.mean(s_buf[-1] >= net.iso_threshold))
    mean_h[-1] = float(np.mean(h[-1]))
    mean_s[-1] = float(np.mean(s_buf[-1]))
    group_memory[-1] = group_memory[-2] if n_steps > 1 else 0.0
    mean_access[-1] = float(np.mean(access_quality))
    mean_attunement[-1] = float(np.mean(attunement[-1]))
    mean_demper_load[-1] = float(np.mean(demper_load[-1]))
    mean_forewarning[-1] = float(np.mean(forewarning[-1]))
    mean_pause[-1] = mean_pause[-2] if n_steps > 1 else 0.0
    mean_recognition[-1] = recognition_signal[-1]
    match_last = 0.5 * (1.0 + np.cos(theta - phi[-1]))
    template_y_last = np.sin(phi[-1]) * (1.0 - 0.5 * s_buf[-1])
    template_mismatch_last = np.abs(y[-1] - template_y_last) / (0.25 + np.abs(template_y_last) + np.std(y[-1]))
    template_mismatch_last = np.clip(template_mismatch_last, 0.0, 1.0)
    neighbor_ref_last = (a @ y[-1]) / deg
    local_mismatch_last = np.abs(y[-1] - neighbor_ref_last) / (0.25 + np.abs(neighbor_ref_last) + float(np.std(y[-1])))
    local_mismatch_last = np.clip(local_mismatch_last, 0.0, 1.0)
    base_error_last = (
        net.instructor_template_weight * template_mismatch_last
        + (1.0 - net.instructor_template_weight)
        * (net.instructor_error_weight * (1.0 - match_last) + (1.0 - net.instructor_error_weight) * local_mismatch_last)
    )
    if net.attunement_enabled:
        if net.attunement_mode == "multi":
            effective_last = np.clip(
                0.45 * attunement_rhythm[-1] + 0.30 * attunement_focus[-1] + 0.25 * attunement_risk[-1],
                0.0,
                1.0,
            )
        else:
            effective_last = np.clip(attunement[-1], 0.0, 1.0)
        base_error_last *= (1.0 - 0.55 * effective_last)
    else:
        effective_last = np.zeros(n)
    error_hist[-1] = np.clip(0.8 * base_error_last + 0.2 * s_buf[-1], 0.0, 1.0)
    error_hist[-1] = np.clip(error_hist[-1] * (1.15 - 0.45 * access_quality), 0.0, 1.0)
    anchor_hist[-1] = np.clip(
        access_quality * (1.0 - s_buf[-1]) * (1.0 - error_hist[-1]) ** 2 * (1.0 + 0.5 * effective_last),
        0.0,
        1.0,
    )
    mean_error[-1] = float(np.mean(error_hist[-1]))
    mean_anchor[-1] = float(np.mean(anchor_hist[-1]))
    neighbor_error_last = (a @ error_hist[-1]) / deg
    turbulence_hist[-1] = np.clip(
        net.turbulence_error_weight * error_hist[-1] + (1.0 - net.turbulence_error_weight) * neighbor_error_last,
        0.0,
        1.0,
    )
    mean_turbulence[-1] = float(np.mean(turbulence_hist[-1]))
    cascade_fraction[-1] = float(np.mean(turbulence_hist[-1] >= net.cascade_threshold))
    if net.focus_lock_enabled and net.instructor_enabled:
        focus_lock_hist[-1] = np.clip(
            (turbulence_hist[-1] - net.focus_lock_trigger) / max(1e-6, 1.0 - net.focus_lock_trigger),
            0.0,
            1.0,
        )
    mean_focus_lock[-1] = float(np.mean(focus_lock_hist[-1]))
    if n_steps > 1:
        channel_peer_hist[-1] = channel_peer_hist[-2]
        channel_instructor_hist[-1] = channel_instructor_hist[-2]
        channel_anchor_hist[-1] = channel_anchor_hist[-2]
        mean_channel_peer[-1] = mean_channel_peer[-2]
        mean_channel_instructor[-1] = mean_channel_instructor[-2]
        mean_channel_anchor[-1] = mean_channel_anchor[-2]
        template_mode_hist[-1] = template_mode_hist[-2]

    # Recovery time: first time after post-stress peak where mean S <= recovery_threshold
    recovery_time = math.nan
    if stress_windows:
        first_start = int(stress_windows[0][0] / net.dt)
        first_end = int((stress_windows[0][0] + stress_windows[0][1]) / net.dt)
        start_idx = min(first_end, n_steps - 1)
        peak_idx = int(np.argmax(mean_s[start_idx:])) + start_idx
        for idx in range(peak_idx, n_steps):
            if mean_s[idx] <= net.recovery_threshold:
                recovery_time = t[idx] - t[first_start]
                break

    return {
        "t": t,
        "x": x,
        "y": y,
        "H": h,
        "K": k_struct,
        "mu": mu,
        "S": s_buf,
        "demper_load": demper_load,
        "phi": phi,
        "phase_dispersion": phase_disp,
        "fraction_isolated": frac_iso,
        "mean_h": mean_h,
        "group_memory": group_memory,
        "phi_gain": phi_gain_hist,
        "kappa": kappa_hist,
        "mean_s": mean_s,
        "error": error_hist,
        "anchor": anchor_hist,
        "mean_error": mean_error,
        "mean_anchor": mean_anchor,
        "turbulence": turbulence_hist,
        "mean_turbulence": mean_turbulence,
        "cascade_fraction": cascade_fraction,
        "focus_lock": focus_lock_hist,
        "mean_focus_lock": mean_focus_lock,
        "channel_peer": channel_peer_hist,
        "channel_instructor": channel_instructor_hist,
        "channel_anchor": channel_anchor_hist,
        "mean_channel_peer": mean_channel_peer,
        "mean_channel_instructor": mean_channel_instructor,
        "mean_channel_anchor": mean_channel_anchor,
        "template_mode": template_mode_hist,
        "mean_demper_load": mean_demper_load,
        "forewarning": forewarning,
        "mean_forewarning": mean_forewarning,
        "mean_pause": mean_pause,
        "recognition_signal": recognition_signal,
        "recognition_signature": recognition_signature,
        "mean_recognition": mean_recognition,
        "mean_access": mean_access,
        "visibility_quality": visibility_quality,
        "hearing_quality": hearing_quality,
        "access_quality": access_quality,
        "attunement": attunement,
        "attunement_rhythm": attunement_rhythm,
        "attunement_focus": attunement_focus,
        "attunement_risk": attunement_risk,
        "mean_attunement": mean_attunement,
        "recovery_time": recovery_time,
        "stress_agents": stress_agents,
        "stress_windows": np.array(stress_windows, dtype=float),
        "positions": positions,
    }


def _plot_results(out: dict[str, np.ndarray], save_path: str | None = None, show: bool = False) -> None:
    if plt is None:
        print("matplotlib is not available. Numerical run completed without plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(out["t"], out["phase_dispersion"], label="phase dispersion")
    axes[0, 0].set_title("Phase coherence")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(out["t"], out["fraction_isolated"], label="fraction isolated")
    axes[0, 1].set_title("Isolation fraction")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(out["t"], out["mean_h"], label="mean H")
    axes[1, 0].plot(out["t"], out["mean_s"], label="mean S")
    axes[1, 0].set_title("Mean stress and suppression")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(out["t"], out["y"][:, 0], label="y[0]")
    axes[1, 1].plot(out["t"], out["y"][:, 1], label="y[1]")
    axes[1, 1].set_title("Sample fast dynamics")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    out = simulate_network()
    print("Network simulation completed.")
    print(f"Recovery time (mean S <= threshold): {out['recovery_time']}")
    _plot_results(out, save_path="E:\\MyProject\\meta-stable-architectures\\outputs\\network_v1_summary.png")


if __name__ == "__main__":
    main()
