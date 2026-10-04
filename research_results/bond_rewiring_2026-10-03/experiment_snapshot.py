"""Degree-preserving partner changes, matched rewiring frequency and resource cost."""
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
    for step in range(steps):
        t = step * p.dt
        before = float(stock.sum())
        fee = 0.0
        if policy != "fixed" and step > 0 and step % rewire_steps == 0:
            event = step // rewire_steps - 1
            initiator = int(order[event % p.n])
            proposal_rng = np.random.default_rng(seed * 10000 + event)
            proposals = candidate_swaps(adj, initiator, proposal_rng, p.candidates)
            random_index = int(proposal_rng.integers(p.candidates))
            move = select_move(proposals, estimates, policy, random_index)
            apply_swap(adj, move)
            fee = p.swap_cost
            totals["swaps"] += 1
            totals["rewiring_cost"] += fee
            if trace_enabled:
                events.append(dict(policy=policy, scenario=scenario, seed=seed, time=t,
                                   a=move[0], b=move[1], c=move[2], d=move[3], fee=fee))
        injection = supply * p.dt
        assert fee <= injection.sum() + 1e-12
        if fee:
            injection *= 1.0 - fee / injection.sum()
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
            largest, reach = graph_metrics(adj, sources)
            totals["largest_component_area"] += largest * p.dt
            totals["source_reach_area"] += reach * p.dt
        if trace_enabled and step % max(1, round(1.0 / p.dt)) == 0:
            largest, reach = graph_metrics(adj, sources)
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
                  **totals, max_mass_error=max_mass_error, max_budget_ratio=max_budget_ratio)
    return result, traces, events, estimates


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    results = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            sub = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            r = dict(scenario=scenario, policy=policy, n=len(sub))
            for key in ("service_fraction", "consumer_p10", "underserved_share", "largest_component_mean",
                        "source_reach_mean", "lost", "moved", "rewiring_cost", "swaps", "final_old_edge_share"):
                vals = np.array([r[key] for r in sub])
                r[key + "_mean"] = float(vals.mean())
                r[key + "_sd"] = float(vals.std(ddof=1))
            results.append(r)
    return results


def comparisons(rows):
    rng = np.random.default_rng(651)
    results = []
    for scenario in SCENARIOS:
        for a, b in (("experience_swap", "random_swap"), ("recent_contact", "random_swap"),
                     ("experience_swap", "recent_contact"), ("experience_swap", "fixed")):
            aa = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == a}
            bb = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == b}
            for key in ("service_fraction", "consumer_p10", "underserved_share", "lost"):
                delta = np.array([aa[s][key] - bb[s][key] for s in sorted(aa)])
                boot = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                results.append(dict(scenario=scenario, treatment=a, baseline=b, metric=key,
                                    mean=float(delta.mean()), ci_low=float(lo), ci_high=float(hi)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--step-check", action="store_true")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("Need at least two seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.step_check:
        rows = []
        for scenario in SCENARIOS:
            for policy in POLICIES:
                for dt in (.2, .1, .05):
                    r, _, _, _ = simulate(policy, scenario, 5900, replace(Params(), dt=dt, observation_noise=0))
                    rows.append({"dt": dt, **r})
        write_csv(args.output / "step_refinement.csv", rows)
        print(f"Step check: {len(rows)} runs")
        return
    source = Path(__file__).read_bytes()
    (args.output / "experiment_snapshot.py").write_bytes(source)
    (args.output / "config.json").write_text(json.dumps(dict(params=asdict(Params()), seeds=args.seeds,
        seed_start=args.seed_start, python=platform.python_version(), numpy=np.__version__,
        sha256=hashlib.sha256(source).hexdigest(), policies=POLICIES, scenarios=SCENARIOS), indent=2), encoding="utf-8")
    rows, traces, events = [], [], []
    for scenario in SCENARIOS:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for policy in POLICIES:
                r, trace, event, _ = simulate(policy, scenario, seed, trace_enabled=seed == args.seed_start)
                rows.append(r)
                traces.extend(trace)
                events.extend(event)
        print(f"Completed {scenario}", flush=True)
    summary = aggregate(rows)
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "paired_comparisons.csv", comparisons(rows))
    write_csv(args.output / "first_seed_traces.csv", traces)
    write_csv(args.output / "first_seed_swaps.csv", events)
    for r in summary:
        print(f"{r['scenario']:20} {r['policy']:17} service={r['service_fraction_mean']:.3f} "
              f"p10={r['consumer_p10_mean']:.3f} reach={r['source_reach_mean']:.3f}")


if __name__ == "__main__":
    main()
