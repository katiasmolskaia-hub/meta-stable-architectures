"""Fork an identical recovered network; compare continued swaps with a pause."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from bond_rewiring_scaled_model import (
    Params, apply_swap, candidate_swaps, graph_metrics, make_world, quality_at, select_move,
)

THRESHOLD = .95
STABLE_WINDOW = 10.
QUIET_DURATION = 30.
SECOND_DURATION = 60.


@dataclass
class State:
    adj: np.ndarray
    stock: np.ndarray
    estimates: np.ndarray
    observation_rng: np.random.Generator
    swaps: int = 0
    cost: float = 0.
    max_mass_error: float = 0.
    max_budget_ratio: float = 0.


@dataclass
class World:
    sources: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    qualities: list
    order: np.ndarray
    seed: int


class RecoveryDetector:
    def __init__(self, dt):
        self.required = round(STABLE_WINDOW / dt)
        self.affected = False
        self.stable_steps = 0

    def observe(self, minimum_service, eligible=True):
        if minimum_service < THRESHOLD - 1e-12:
            self.affected = True
            self.stable_steps = 0
        elif self.affected and eligible:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        return self.affected and self.stable_steps >= self.required


def parameters(n, cost_multiplier, dt=.2, noise=.03):
    return replace(Params(), n=n, sources=n // 4, swaps_per_event=n // 24,
                   swap_cost=.024 * cost_multiplier, dt=dt, observation_noise=noise)


def initialize(seed, scenario, p):
    adj, sources, supply, demand, qualities = make_world(seed, scenario, p)
    state = State(adj, np.full(p.n, p.initial_stock), np.full((p.n, p.n), p.prior),
                  np.random.default_rng(seed + 200000))
    world = World(sources, supply, demand, qualities,
                  np.random.default_rng(seed + 300000).permutation(p.n), seed)
    return state, world


def fingerprint(state):
    h = hashlib.sha256()
    for a in (state.adj, state.stock, state.estimates):
        h.update(a.tobytes())
    h.update(json.dumps(state.observation_rng.bit_generator.state, sort_keys=True).encode())
    h.update(repr((state.swaps, state.cost, state.max_mass_error, state.max_budget_ratio)).encode())
    return h.hexdigest()


def advance(state, world, policy, p, step, quality, allow_swaps=True):
    """Same arithmetic as the previous kernel; proposal clock survives a pause."""
    before = float(state.stock.sum())
    fee = 0.
    count = 0
    rewire_steps = round(p.rewire_interval / p.dt)
    assert abs(rewire_steps * p.dt - p.rewire_interval) < 1e-8
    if allow_swaps and policy != "fixed" and step > 0 and step % rewire_steps == 0:
        for j in range(p.swaps_per_event):
            event = (step // rewire_steps - 1) * p.swaps_per_event + j
            initiator = int(world.order[event % p.n])
            rng = np.random.default_rng(world.seed * 10000 + event)
            proposals = candidate_swaps(state.adj, initiator, rng, p.candidates)
            move = select_move(proposals, state.estimates, policy, int(rng.integers(p.candidates)))
            apply_swap(state.adj, move)
            fee += p.swap_cost
            count += 1
            state.swaps += 1
            state.cost += p.swap_cost
    injection = world.supply * p.dt
    if fee:
        injection *= 1.0 - fee / injection.sum()
    assert np.all(state.stock + injection >= -1e-12), "Sources cannot afford scheduled swap"
    state.stock += injection
    spill = np.maximum(0., state.stock - p.storage)
    state.stock = np.minimum(state.stock, p.storage)
    leakage = state.stock * (1. - np.exp(-p.leakage * p.dt))
    state.stock -= leakage
    flow = p.conductance * np.maximum(state.stock[:, None] - state.stock[None, :], 0.) * state.adj * p.dt
    row_sum = flow.sum(axis=1)
    available = np.minimum(state.stock, p.per_node_budget * p.dt)
    flow *= np.minimum(1., available / np.maximum(row_sum, 1e-15))[:, None]
    budget = p.total_budget_per_node * p.n * p.dt
    if flow.sum() > budget:
        flow *= budget / flow.sum()
    received = flow * quality
    loss = float((flow - received).sum())
    state.stock += received.sum(axis=0) - flow.sum(axis=1)
    spill += np.maximum(0., state.stock - p.storage)
    state.stock = np.minimum(state.stock, p.storage)
    consumption = np.minimum(state.stock, world.demand * p.dt)
    state.stock -= consumption
    noise = state.observation_rng.normal(0, p.observation_noise, (p.n, p.n))
    noise = np.triu(noise, 1)
    noise += noise.T
    observation = np.clip(quality + noise, 0, 1)
    exposure = flow + flow.T
    if policy == "experience_swap":
        alpha = 1 - np.exp(-p.learning_rate * exposure)
        state.estimates += alpha * (observation - state.estimates)
    elif policy == "recent_contact":
        active = exposure > 1e-12
        state.estimates[active] = observation[active]
    mass_error = abs(state.stock.sum() - (before + injection.sum() - spill.sum() - leakage.sum() - loss - consumption.sum()))
    ratio = float(flow.sum() / budget) if budget else 0.
    state.max_mass_error = max(state.max_mass_error, float(mass_error))
    state.max_budget_ratio = max(state.max_budget_ratio, ratio)
    assert mass_error < 1e-8 and np.isfinite(state.stock).all()
    assert state.stock.min() >= -1e-10 and state.stock.max() <= p.storage + 1e-10
    assert ratio <= 1 + 1e-10
    minimum = float(np.min(consumption[~world.sources] / (world.demand[~world.sources] * p.dt)))
    return dict(consumption=consumption, minimum_service=minimum, fee=fee, swaps=count, loss=loss)


def run_phase(state, world, policy, scenario, p, start, duration, allow_swaps,
              second_shock, fork_adj, branch, trace_enabled):
    served = np.zeros(p.n)
    low_steps = swaps = 0
    cost = loss = 0.
    detector = RecoveryDetector(p.dt)
    recovery_delay = None
    traces = []
    steps = round(duration / p.dt)
    for step in range(start, start + steps):
        quality = world.qualities[1] if second_shock else quality_at(step * p.dt, scenario, world.qualities, p)
        r = advance(state, world, policy, p, step, quality, allow_swaps)
        served += r["consumption"]
        low_steps += r["minimum_service"] < THRESHOLD - 1e-12
        cost += r["fee"]
        swaps += r["swaps"]
        loss += r["loss"]
        if detector.observe(r["minimum_service"]) and recovery_delay is None:
            recovery_delay = (step + 1 - start) * p.dt
        if trace_enabled and step % round(1. / p.dt) == 0:
            traces.append(dict(branch=branch, phase="second" if second_shock else "quiet",
                time=(step + 1) * p.dt, minimum_service=r["minimum_service"],
                service=float(r["consumption"].sum() / (world.demand.sum() * p.dt)),
                stock_total=float(state.stock.sum()), swaps_total=state.swaps,
                fork_edge_share=float((state.adj & fork_adj).sum() / fork_adj.sum())))
    consumer = served[~world.sources] / (world.demand[~world.sources] * duration)
    _, reach = graph_metrics(state.adj, world.sources)
    result = dict(service=float(served.sum() / (world.demand.sum() * duration)),
                  p10=float(np.quantile(consumer, .1)), minimum_consumer=float(consumer.min()),
                  low_time_fraction=float(low_steps / steps), relapse=int(low_steps > 0),
                  cost=cost, swaps=swaps, lost=loss, stock_end=float(state.stock.sum()),
                  fork_edge_share=float((state.adj & fork_adj).sum() / fork_adj.sum()),
                  source_reach_end=reach,
                  recovery_status=("unaffected" if not detector.affected else
                                   "recovered" if recovery_delay is not None else "not_recovered"),
                  recovery_delay=recovery_delay,
                  unrecovered=int(detector.affected and recovery_delay is None))
    return result, traces


def run_episode(n, cost_multiplier, scenario, policy, seed, dt=.2, noise=.03, trace_enabled=False):
    p = parameters(n, cost_multiplier, dt, noise)
    state, world = initialize(seed, scenario, p)
    detector = RecoveryDetector(p.dt)
    fork_step = None
    traces = []
    for step in range(round(p.horizon / p.dt)):
        t = step * p.dt
        r = advance(state, world, policy, p, step, quality_at(t, scenario, world.qualities, p))
        if t >= p.crisis_start:
            if detector.observe(r["minimum_service"], eligible=scenario != "temporary_damage" or t >= 70):
                fork_step = step + 1
        if trace_enabled and step % round(1. / p.dt) == 0:
            traces.append(dict(branch="common", phase="first", time=(step + 1) * p.dt,
                minimum_service=r["minimum_service"],
                service=float(r["consumption"].sum() / (world.demand.sum() * p.dt)),
                stock_total=float(state.stock.sum()), swaps_total=state.swaps, fork_edge_share=None))
        if fork_step is not None:
            break
    metadata = dict(n=n, cost_multiplier=cost_multiplier, scenario=scenario, policy=policy,
                    seed=seed, dt=dt, noise=noise)
    episode = dict(**metadata, status="recovered" if fork_step is not None else
                   "not_recovered" if detector.affected else "unaffected",
                   recovery_time=fork_step * p.dt if fork_step is not None else None,
                   max_mass_error=state.max_mass_error, max_budget_ratio=state.max_budget_ratio,
                   fork_fingerprint=fingerprint(state) if fork_step is not None else None)
    rows = []
    if fork_step is not None:
        fork_adj = state.adj.copy()
        for branch in ("continue", "pause"):
            branch_state = copy.deepcopy(state)
            assert fingerprint(branch_state) == episode["fork_fingerprint"]
            quiet, qt = run_phase(branch_state, world, policy, scenario, p, fork_step, QUIET_DURATION,
                                 branch == "continue", False, fork_adj, branch, trace_enabled)
            if branch == "pause":
                np.testing.assert_array_equal(branch_state.adj, fork_adj)
            second_start = fork_step + round(QUIET_DURATION / p.dt)
            second, st = run_phase(branch_state, world, policy, scenario, p, second_start, SECOND_DURATION,
                                   True, True, fork_adj, branch, trace_enabled)
            rows.append(dict(**metadata, branch=branch, recovery_time=fork_step * p.dt,
                second_shock_time=second_start * p.dt,
                **{"quiet_" + k: v for k, v in quiet.items()},
                **{"second_" + k: v for k, v in second.items()},
                max_mass_error=branch_state.max_mass_error, max_budget_ratio=branch_state.max_budget_ratio))
            traces.extend(qt + st)
        assert fingerprint(state) == episode["fork_fingerprint"]
        assert rows[0]["second_swaps"] == rows[1]["second_swaps"] == 30 * (n // 24)
        assert rows[0]["second_cost"] == rows[1]["second_cost"]
        assert rows[0]["quiet_swaps"] == 15 * (n // 24) and rows[1]["quiet_swaps"] == 0
    return episode, rows, [dict(**metadata, **t) for t in traces]
