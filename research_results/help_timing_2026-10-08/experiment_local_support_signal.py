"""Sparse local support requests. No all-pairs state or global recovery oracle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "local_support_signal_2026-10-04"
POLICIES = ("fixed", "periodic", "local_signal", "shuffled_signal")
SCENARIOS = ("uniform", "persistent_damage", "repeated_damage", "scarcity")
SIZES = (96, 384, 960)


@dataclass(frozen=True)
class Params:
    n: int = 96
    dt: float = .2
    horizon: float = 160.
    radius: int = 6
    storage: float = 2.
    initial_stock: float = .7
    source_rate: float = .55
    demand_rate: float = .15
    conductance: float = .55
    per_node_budget: float = .65
    leakage: float = .003
    good_quality: float = .98
    bad_quality: float = .20
    bad_share: float = .35
    prior: float = .75
    swap_cost: float = .024
    helper_reserve: float = .3
    helper_margin: float = .1
    request_interval: float = 8.
    need_threshold: float = .95
    clear_threshold: float = .99
    need_duration: float = 2.
    clear_duration: float = 5.


def make_world(seed, scenario, p):
    assert p.n > 2 * p.radius and p.radius >= 2 and p.n % 4 == 0
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(p.n), p.radius)
    distance = np.tile(np.arange(1, p.radius + 1), p.n)
    v = (u + distance) % p.n
    adjacency = [set() for _ in range(p.n)]
    discovery = [dict() for _ in range(p.n)]
    for e, (a, b) in enumerate(zip(u, v)):
        a, b = int(a), int(b)
        discovery[a][b] = discovery[b][a] = e
        if distance[e] <= 2:
            adjacency[a].add(b)
            adjacency[b].add(a)
    sources = np.zeros(p.n, dtype=bool)
    sources[rng.choice(p.n, p.n // 4, replace=False)] = True
    rate = .30 if scenario == "scarcity" else p.source_rate
    supply = sources * rate * rng.uniform(.94, 1.06, p.n)
    demand = (~sources) * p.demand_rate * rng.uniform(.92, 1.08, p.n)
    qualities = np.where(rng.random((2, len(u))) < p.bad_share, p.bad_quality, p.good_quality)
    return dict(u=u, v=v, active=distance <= 2, adjacency=adjacency, discovery=discovery,
                sources=sources, supply=supply, demand=demand, qualities=qualities)


def quality_at(t, scenario, world, p):
    if scenario in ("uniform", "scarcity") or t < 40 or (scenario == "repeated_damage" and 70 <= t < 100):
        return np.full(len(world["u"]), p.good_quality)
    return world["qualities"][1 if scenario == "repeated_damage" and t >= 100 else 0]


def flow_step(stock, active, estimates, world, quality, p):
    """Only 6N possible edges; no dense graph or all-pairs observation matrix."""
    before = float(stock.sum())
    injection = world["supply"] * p.dt
    stock += injection
    spill = np.maximum(0., stock - p.storage)
    stock = np.minimum(stock, p.storage)
    leakage = stock * (1. - np.exp(-p.leakage * p.dt))
    stock -= leakage
    u, v = world["u"], world["v"]
    diff = stock[u] - stock[v]
    donor = np.where(diff >= 0, u, v)
    receiver = np.where(diff >= 0, v, u)
    volume = p.conductance * np.abs(diff) * active * p.dt
    outgoing = np.bincount(donor, weights=volume, minlength=p.n)
    scale = np.minimum(1., np.minimum(stock, p.per_node_budget * p.dt) / np.maximum(outgoing, 1e-15))
    volume *= scale[donor]
    received = volume * quality
    stock += np.bincount(receiver, weights=received, minlength=p.n) - np.bincount(donor, weights=volume, minlength=p.n)
    spill += np.maximum(0., stock - p.storage)
    stock = np.minimum(stock, p.storage)
    consumption = np.minimum(stock, world["demand"] * p.dt)
    stock -= consumption
    used = volume > 1e-12
    estimates[used] = quality[used]
    loss = float((volume - received).sum())
    error = abs(stock.sum() - (before + injection.sum() - spill.sum() - leakage.sum() - loss - consumption.sum()))
    outgoing = np.bincount(donor, weights=volume, minlength=p.n)
    ratio = float(outgoing.max() / (p.per_node_budget * p.dt))
    assert error < 1e-8 and stock.min() >= -1e-10 and stock.max() <= p.storage + 1e-10
    assert ratio <= 1 + 1e-10
    return stock, consumption, loss, error, ratio


def update_need(need, below, above, service, consumers, p):
    below[:] = np.where(consumers & (service < p.need_threshold - 1e-12), below + 1, 0)
    above[:] = np.where(consumers & (service >= p.clear_threshold - 1e-12), above + 1, 0)
    old = need.copy()
    need[below >= round(p.need_duration / p.dt)] = True
    need[above >= round(p.clear_duration / p.dt)] = False
    need[~consumers] = False
    return int((need & ~old).sum()), int((~need & old).sum())


def request_help(a, stock, estimates, world, seed, step, p):
    """Requester sees only bounded replies from its discovery neighborhood."""
    adjacency, discovery, active = world["adjacency"], world["discovery"], world["active"]
    rng = np.random.default_rng(seed * 1000000 + step * p.n + a)
    neighbors = sorted(adjacency[a])
    worst = min(estimates[discovery[a][b]] for b in neighbors)
    b = int(rng.choice([b for b in neighbors if estimates[discovery[a][b]] == worst]))
    ab = discovery[a][b]
    best, best_score = None, -np.inf
    for cc in rng.permutation(sorted(discovery[a])):
        c = int(cc)
        if c in adjacency[a] or stock[c] < p.helper_reserve + p.swap_cost or stock[c] <= stock[a] + p.helper_margin:
            continue
        ac = discovery[a][c]
        for d in sorted(adjacency[c]):
            if d in (a, b) or d not in discovery[b] or d in adjacency[b]:
                continue
            cd, bd = discovery[c][d], discovery[b][d]
            score = estimates[ac] + estimates[bd] - estimates[ab] - estimates[cd]
            if score > best_score:
                best, best_score = (c, d, ac, cd, bd), score
    messages = 2 * len(discovery[a])
    if best is None:
        return messages, 0, 0.
    c, d, ac, cd, bd = best
    before = float(stock.sum())
    stock[c] -= p.swap_cost
    assert stock[c] >= p.helper_reserve - 1e-12
    assert abs(stock.sum() - (before - p.swap_cost)) < 1e-8
    for x, y, e in ((a, b, ab), (c, d, cd)):
        adjacency[x].remove(y)
        adjacency[y].remove(x)
        active[e] = False
    for x, y, e in ((a, c, ac), (b, d, bd)):
        adjacency[x].add(y)
        adjacency[y].add(x)
        active[e] = True
    assert all(len(adjacency[x]) == 4 for x in (a, b, c, d))
    return messages + 8, 1, p.swap_cost


def source_reach(world):
    seen = set(map(int, np.flatnonzero(world["sources"])))
    todo = list(seen)
    while todo:
        for node in world["adjacency"][todo.pop()]:
            if node not in seen:
                seen.add(node)
                todo.append(node)
    return sum(not world["sources"][i] for i in seen) / int((~world["sources"]).sum())


def simulate(policy, scenario, seed, p=Params(), replay=None, keep_log=True):
    world = make_world(seed, scenario, p)
    stock = np.full(p.n, p.initial_stock)
    estimates = np.full(len(world["u"]), p.prior)
    consumers = ~world["sources"]
    need = np.zeros(p.n, dtype=bool)
    below = np.zeros(p.n, dtype=int)
    above = np.zeros(p.n, dtype=int)
    last_request = np.full(p.n, -10**9, dtype=int)
    interval = round(p.request_interval / p.dt)
    assert abs(interval * p.dt - p.request_interval) < 1e-9
    phase_rng = np.random.default_rng(seed + 200000)
    phases = np.rint(phase_rng.integers(0, 40, p.n) * .2 / p.dt).astype(int)
    replay_by_step = {}
    if policy == "shuffled_signal":
        assert replay is not None
        ids = np.flatnonzero(consumers)
        mapping = dict(zip(map(int, ids), map(int, np.random.default_rng(seed + 300000).permutation(ids))))
        for step, node in replay:
            replay_by_step.setdefault(step, []).append(mapping[node])
    served = np.zeros((4, p.n))
    messages = np.zeros(4, dtype=int)
    requests = np.zeros(4, dtype=int)
    swaps = np.zeros(4, dtype=int)
    fees = np.zeros(4)
    onsets = np.zeros(4, dtype=int)
    offsets = np.zeros(4, dtype=int)
    log = []
    max_mass = max_budget = 0.
    requests_90_100 = requests_110_140 = 0
    started = time.perf_counter()
    for step in range(round(p.horizon / p.dt)):
        t = step * p.dt
        phase = 0 if t < 40 else 1 if t < 70 else 2 if t < 100 else 3
        if policy == "periodic":
            actors = list(map(int, np.flatnonzero(consumers & ((step % interval) == phases)))) if step else []
        elif policy == "local_signal":
            actors = list(map(int, np.flatnonzero(need & ((step - last_request) >= interval))))
        elif policy == "shuffled_signal":
            actors = replay_by_step.get(step, [])
        else:
            assert policy == "fixed"
            actors = []
        # A reproducible tie-breaking scheduler; physical simultaneity and radio
        # collisions are not modeled. The signal decision has no global input.
        actors = sorted(actors, key=lambda a: ((a * 2654435761 + step * 2246822519 + seed) % 4294967291))
        for a in actors:
            assert step - last_request[a] >= interval
            last_request[a] = step
            msg, swap, fee = request_help(a, stock, estimates, world, seed, step, p)
            requests[phase] += 1
            messages[phase] += msg
            swaps[phase] += swap
            fees[phase] += fee
            if keep_log:
                log.append((step, a))
            requests_90_100 += 90 <= t < 100
            requests_110_140 += 110 <= t < 140
        stock, consumption, _, error, ratio = flow_step(stock, world["active"], estimates, world,
                                                        quality_at(t, scenario, world, p), p)
        max_mass, max_budget = max(max_mass, error), max(max_budget, ratio)
        service = np.ones(p.n)
        service[consumers] = consumption[consumers] / (world["demand"][consumers] * p.dt)
        on, off = update_need(need, below, above, service, consumers, p)
        onsets[phase] += on
        offsets[phase] += off
        served[phase] += consumption
    durations = (40., 30., 30., 60.)
    assert p.horizon == sum(durations)
    individual = served[1:, consumers].sum(axis=0) / (world["demand"][consumers] * 120.)
    result = dict(n=p.n, scenario=scenario, policy=policy, seed=seed, dt=p.dt,
                  service=float(served[1:].sum() / (world["demand"].sum() * 120.)),
                  p10=float(np.quantile(individual, .1)),
                  messages=int(messages.sum()), requests=int(requests.sum()), swaps=int(swaps.sum()),
                  resource_cost=float(fees.sum()), active_need_end=int(need.sum()),
                  request_rate_90_100=float(requests_90_100 / (consumers.sum() * 10.)),
                  request_rate_110_140=float(requests_110_140 / (consumers.sum() * 30.)),
                  source_reach_end=source_reach(world),
                  allowed_edges=len(world["u"]), active_edges=int(world["active"].sum()),
                  max_mass_error=max_mass, max_budget_ratio=max_budget,
                  simulation_seconds=time.perf_counter() - started)
    for phase, duration in enumerate(durations):
        result.update({f"phase{phase}_service": float(served[phase].sum() / (world["demand"].sum() * duration)),
                       f"phase{phase}_requests": int(requests[phase]), f"phase{phase}_messages": int(messages[phase]),
                       f"phase{phase}_need_onsets": int(onsets[phase]), f"phase{phase}_need_offsets": int(offsets[phase])})
    assert all(len(nbrs) == 4 for nbrs in world["adjacency"])
    assert all(len(nbrs) == 2 * p.radius for nbrs in world["discovery"])
    return result, log


def run_group(case):
    n, scenario, seed, dt = case
    p = Params(n=n, dt=dt)
    local, log = simulate("local_signal", scenario, seed, p)
    shuffled, _ = simulate("shuffled_signal", scenario, seed, p, replay=log, keep_log=False)
    assert local["requests"] == shuffled["requests"]
    rows = [simulate(policy, scenario, seed, p, keep_log=False)[0] for policy in ("fixed", "periodic")]
    rows += [local, shuffled]
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--step-check", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "step_" if args.step_check else ""
    if (args.output / (prefix + "runs.csv")).exists():
        parser.error("Refusing to overwrite existing results")
    cases = (list(product((96,), ("repeated_damage",), range(10100, 10106), (.2, .1))) if args.step_check
             else list(product(SIZES, SCENARIOS, range(10000, 10012), (.2,))))
    source = Path(__file__).read_bytes()
    (args.output / (prefix + "experiment_snapshot.py")).write_bytes(source)
    (args.output / (prefix + "config.json")).write_text(json.dumps(dict(params=asdict(Params()),
        cases=cases, policies=POLICIES, python=platform.python_version(), numpy=np.__version__,
        sha256=hashlib.sha256(source).hexdigest()), indent=2), encoding="utf-8")
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, group in enumerate(executor.map(run_group, cases, chunksize=1), 1):
            rows.extend(group)
            if i % 12 == 0 or i == len(cases):
                write_csv(args.output / (prefix + "runs.csv"), rows)
                print(f"Completed {i}/{len(cases)} worlds ({len(rows)} runs); elapsed {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
