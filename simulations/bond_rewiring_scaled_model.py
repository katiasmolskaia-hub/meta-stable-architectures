"""Scale/cost kernel fork: proportional event batches, original flow and learning.

At n=24 this reproduces experiment_bond_rewiring. The separate file keeps the
previous experiment unchanged. Use experiment_bond_rewiring_scale_cost to run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bond_rewiring_2026-10-03"
POLICIES = ("fixed", "random_swap", "recent_contact", "experience_swap")
SCENARIOS = ("uniform", "persistent_damage", "temporary_damage", "rapid_change", "separated_resources", "noisy_feedback")


@dataclass(frozen=True)
class Params:
    n: int = 24
    sources: int = 6
    dt: float = 0.2
    horizon: float = 120.0
    crisis_start: float = 40.0
    rewire_interval: float = 2.0
    candidates: int = 12
    swap_cost: float = 0.024
    source_rate: float = 0.55
    demand_rate: float = 0.15
    storage: float = 2.0
    initial_stock: float = 0.7
    conductance: float = 0.55
    per_node_budget: float = 0.65
    total_budget_per_node: float = 0.4
    leakage: float = 0.003
    good_quality: float = 0.98
    bad_quality: float = 0.20
    bad_share: float = 0.35
    cross_quality: float = 0.50
    prior: float = 0.75
    learning_rate: float = 5.0
    observation_noise: float = 0.03
    swaps_per_event: int = 1


def initial_graph(n: int, permutation: np.ndarray) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for k in range(n):
        for distance in (1, 2):
            a, b = permutation[k], permutation[(k + distance) % n]
            adj[a, b] = adj[b, a] = True
    return adj


def assert_graph(adj: np.ndarray) -> None:
    assert np.array_equal(adj, adj.T) and not np.diag(adj).any()
    assert np.all(adj.sum(axis=1) == 4)


def graph_metrics(adj: np.ndarray, sources: np.ndarray) -> tuple[float, float]:
    unseen = set(range(len(adj)))
    components = []
    while unseen:
        start = unseen.pop()
        component, pending = {start}, [start]
        while pending:
            for neighbor in np.flatnonzero(adj[pending.pop()]):
                v = int(neighbor)
                if v in unseen:
                    unseen.remove(v)
                    component.add(v)
                    pending.append(v)
        components.append(component)
    reachable = sum(sum(not sources[v] for v in c) for c in components if any(sources[v] for v in c))
    return max(map(len, components)) / len(adj), reachable / int((~sources).sum())


def candidate_swaps(adj: np.ndarray, initiator: int, rng: np.random.Generator, count: int):
    candidates = []
    a = initiator
    neighbors = np.flatnonzero(adj[a])
    possible = np.flatnonzero(~adj[a] & (np.arange(len(adj)) != a))
    # Each candidate is a feasible exchange proposed by the same initiating node.
    for _ in range(2000):
        b = int(rng.choice(neighbors))
        c = int(rng.choice(possible))
        d_options = [int(v) for v in np.flatnonzero(adj[c]) if v not in (a, b) and not adj[b, v]]
        if d_options:
            candidates.append((a, b, c, int(rng.choice(d_options))))
        if len(candidates) == count:
            return candidates
    raise RuntimeError("Could not construct matched candidate budget")


def apply_swap(adj: np.ndarray, move) -> None:
    a, b, c, d = move
    assert len({a, b, c, d}) == 4
    assert adj[a, b] and adj[c, d] and not adj[a, c] and not adj[b, d]
    adj[a, b] = adj[b, a] = adj[c, d] = adj[d, c] = False
    adj[a, c] = adj[c, a] = adj[b, d] = adj[d, b] = True
    assert_graph(adj)


def select_move(candidates, estimates, policy: str, random_index: int):
    if policy == "random_swap":
        return candidates[random_index]
    scores = [estimates[a, c] + estimates[b, d] - estimates[a, b] - estimates[c, d]
              for a, b, c, d in candidates]
    return candidates[int(np.argmax(scores))]


def make_world(seed: int, scenario: str, p: Params):
    rng = np.random.default_rng(seed)
    adj = initial_graph(p.n, rng.permutation(p.n))
    source_indices = rng.choice(p.n, p.sources, replace=False)
    if scenario == "separated_resources":
        source_indices = np.arange(p.sources)
    sources = np.zeros(p.n, dtype=bool)
    sources[source_indices] = True
    supply = sources * p.source_rate * rng.uniform(.94, 1.06, p.n)
    demand = (~sources) * p.demand_rate * rng.uniform(.92, 1.08, p.n)
    qualities = []
    for _ in range(16):
        bad = rng.random((p.n, p.n)) < p.bad_share
        bad = np.triu(bad, 1)
        bad |= bad.T
        qualities.append(np.where(bad, p.bad_quality, p.good_quality))
    return adj, sources, supply, demand, qualities


def quality_at(t, scenario, qualities, p):
    if scenario == "separated_resources":
        groups = np.arange(p.n) < p.n // 2
        return np.where(groups[:, None] == groups[None, :], p.good_quality, p.cross_quality)
    if scenario == "uniform" or t < p.crisis_start or (scenario == "temporary_damage" and t >= 70):
        return np.full((p.n, p.n), p.good_quality)
    epoch = int((t - p.crisis_start) // 10) if scenario == "rapid_change" else 0
    return qualities[epoch % len(qualities)]


def simulate(policy: str, scenario: str, seed: int, p: Params = Params(), trace_enabled=False):
    if policy not in POLICIES or scenario not in SCENARIOS:
        raise ValueError((policy, scenario))
    adj, sources, supply, demand, qualities = make_world(seed, scenario, p)
    initial = adj.copy()
    estimates = np.full((p.n, p.n), p.prior)
    stock = np.full(p.n, p.initial_stock)
    steps = round(p.horizon / p.dt)
    rewire_steps = round(p.rewire_interval / p.dt)
    if rewire_steps < 1 or abs(rewire_steps * p.dt - p.rewire_interval) > 1e-8:
        raise ValueError("dt must divide rewire_interval")
    observation_rng = np.random.default_rng(seed + 200000)
    events_rng = np.random.default_rng(seed + 300000)
    order = events_rng.permutation(p.n)
    noise_scale = p.observation_noise * (10 if scenario == "noisy_feedback" else 1)
    served = np.zeros(p.n)
    wanted = np.zeros(p.n)
    totals = dict(lost=0.0, leaked=0.0, spilled=0.0, moved=0.0, rewiring_cost=0.0,
                  swaps=0, largest_component_area=0.0, source_reach_area=0.0)
    traces, events = [], []
    max_mass_error = max_budget_ratio = 0.0
    largest, reach = graph_metrics(adj, sources)
    for step in range(steps):
        t = step * p.dt
        before = float(stock.sum())
        fee = 0.0
        if policy != "fixed" and step > 0 and step % rewire_steps == 0:
            for _ in range(p.swaps_per_event):
                event = totals["swaps"]
                initiator = int(order[event % p.n])
                proposal_rng = np.random.default_rng(seed * 10000 + event)
                proposals = candidate_swaps(adj, initiator, proposal_rng, p.candidates)
                random_index = int(proposal_rng.integers(p.candidates))
                move = select_move(proposals, estimates, policy, random_index)
                apply_swap(adj, move)
                fee += p.swap_cost
                totals["swaps"] += 1
                totals["rewiring_cost"] += p.swap_cost
                if trace_enabled:
                    events.append(dict(policy=policy, scenario=scenario, seed=seed, time=t,
                                       a=move[0], b=move[1], c=move[2], d=move[3], fee=p.swap_cost))
            largest, reach = graph_metrics(adj, sources)
        injection = supply * p.dt
        if fee:
            # A fee is an impulse paid by sources in proportion to their supply.
            # If it exceeds this step's production, use their existing stock.
            # Restricting payment to dt * supply makes refinement impossible.
            injection *= 1.0 - fee / injection.sum()
        assert np.all(stock + injection >= -1e-12), "Sources cannot afford scheduled swap"
        stock += injection
        spill = np.maximum(0.0, stock - p.storage)
        stock = np.minimum(stock, p.storage)
        leakage = stock * (1.0 - np.exp(-p.leakage * p.dt))
        stock -= leakage
        quality = quality_at(t, scenario, qualities, p)
        flow = p.conductance * np.maximum(stock[:, None] - stock[None, :], 0.0) * adj * p.dt
        row_sum = flow.sum(axis=1)
        available = np.minimum(stock, p.per_node_budget * p.dt)
        scale = np.minimum(1.0, available / np.maximum(row_sum, 1e-15))
        flow *= scale[:, None]
        budget = p.total_budget_per_node * p.n * p.dt
        if flow.sum() > budget:
            flow *= budget / flow.sum()
        received = flow * quality
        loss = float((flow - received).sum())
        stock += received.sum(axis=0) - flow.sum(axis=1)
        # Storage overflow is counted even when simultaneous incoming traffic fills a node.
        spill += np.maximum(0.0, stock - p.storage)
        stock = np.minimum(stock, p.storage)
        consumption = np.minimum(stock, demand * p.dt)
        stock -= consumption
        # Draw every pair's noise on every step, so policy-dependent contact counts
        # cannot change the exogenous random stream.
        noise = observation_rng.normal(0, noise_scale, (p.n, p.n))
        noise = np.triu(noise, 1)
        noise += noise.T
        observation = np.clip(quality + noise, 0, 1)
        exposure = flow + flow.T
        active = exposure > 1e-12
        if policy == "experience_swap":
            alpha = 1 - np.exp(-p.learning_rate * exposure)
            estimates += alpha * (observation - estimates)
        elif policy == "recent_contact":
            estimates[active] = observation[active]
        mass_error = abs(stock.sum() - (before + injection.sum() - spill.sum() - leakage.sum() - loss - consumption.sum()))
        max_mass_error = max(max_mass_error, float(mass_error))
        max_budget_ratio = max(max_budget_ratio, float(flow.sum() / budget) if budget else 0.0)
        assert mass_error < 1e-8 and np.isfinite(stock).all()
        assert stock.min() >= -1e-10 and stock.max() <= p.storage + 1e-10
        assert flow.sum() <= budget + 1e-10
        if t >= p.crisis_start:
            served += consumption
            wanted += demand * p.dt
            totals["lost"] += loss
            totals["leaked"] += float(leakage.sum())
            totals["spilled"] += float(spill.sum())
            totals["moved"] += float(flow.sum())
            totals["largest_component_area"] += largest * p.dt
            totals["source_reach_area"] += reach * p.dt
        if trace_enabled and step % max(1, round(1.0 / p.dt)) == 0:
            traces.append(dict(policy=policy, scenario=scenario, seed=seed, time=t,
                               service=float(consumption.sum() / (demand.sum() * p.dt)),
                               largest_component=largest, source_reach=reach,
                               old_edge_share=float((adj & initial).sum() / adj.sum()),
                               stock_total=float(stock.sum()), loss_rate=loss / p.dt))
    consumer_service = served[~sources] / wanted[~sources]
    horizon = p.horizon - p.crisis_start
    result = dict(policy=policy, scenario=scenario, seed=seed,
                  service_fraction=float(served.sum() / wanted.sum()),
                  consumer_p10=float(np.quantile(consumer_service, .1)),
                  underserved_share=float(np.mean(consumer_service < .5)),
                  largest_component_mean=totals["largest_component_area"] / horizon,
                  source_reach_mean=totals["source_reach_area"] / horizon,
                  final_old_edge_share=float((adj & initial).sum() / adj.sum()),
                  gross_supply=float(supply.sum() * p.horizon),
                  **totals, max_mass_error=max_mass_error, max_budget_ratio=max_budget_ratio)
    return result, traces, events, estimates


