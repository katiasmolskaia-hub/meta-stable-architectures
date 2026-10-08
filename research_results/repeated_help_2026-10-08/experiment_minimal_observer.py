"""Two observable recovery signals with a bounded local resource intervention."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import time
import numpy as np
import experiment_local_support_signal as base

OUT = base.ROOT / 'outputs/minimal_observer_2026-10-08'
POLICIES = ('autonomous', 'level', 'persistent', 'two_signals', 'shuffled')
SCENARIOS = ('quiet', 'transient', 'mixed', 'recurrent', 'scarcity')


class Observer:
    def __init__(self, n, dt):
        self.dt = dt
        self.q = np.ones(n)
        self.below = np.zeros(n)
        self.window = round(4 / dt)
        self.buffer = np.ones((self.window + 1, n))
        self.count = 0
        self.trend = np.zeros(n)

    def update(self, service):
        self.q += (1 - np.exp(-self.dt / 2)) * (service - self.q)
        self.below = np.where(self.q < .95 - 1e-12, self.below + self.dt, 0.)
        self.count += 1
        previous = self.buffer[(self.count - self.window) % len(self.buffer)]
        self.trend = self.q - previous
        self.buffer[self.count % len(self.buffer)] = self.q

    def flags(self, policy):
        if policy == 'level':
            return self.q < .95 - 1e-12
        persistent = self.below >= 2 - 1e-12
        if policy == 'persistent':
            return persistent
        if policy == 'two_signals':
            return persistent & (self.trend <= .02 + 1e-12) & (self.count >= self.window)
        raise ValueError(policy)


def forcing(t, scenario, world, p, supply0, demand0):
    q = np.full(len(world['u']), p.good_quality)
    supply, demand = supply0.copy(), demand0.copy()
    if scenario == 'transient' and 40 <= t < 52:
        q = world['qualities'][0].copy()
    elif scenario == 'recurrent':
        if 30 <= t < 48:
            q = world['qualities'][0].copy()
        elif 70 <= t < 92:
            q = world['qualities'][1].copy()
    elif scenario == 'mixed':
        severity = min(np.clip((t - 30) / 25, 0, 1), np.clip((100 - t) / 15, 0, 1))
        edges = world['u'] < p.n // 3
        q[edges] += severity * (world['qualities'][0, edges] - p.good_quality)
        if 45 <= t < 65:
            demand[p.n // 3:2 * p.n // 3] *= 1.8
        if 55 <= t < 80:
            supply[2 * p.n // 3:] *= .45
    elif scenario == 'scarcity' and t >= 40:
        supply *= .55
    return q, supply, demand


def diagnostics(service, flags, p):
    rows = []
    # service[k] covers [k*dt,(k+1)*dt); flags[k] are formed at k*dt.
    width = round(8 / p.dt)
    for policy, history in flags.items():
        counts = np.zeros(4, dtype=int)
        for k in range(round(20 / p.dt), len(service) - width + 1, round(4 / p.dt)):
            truth = service[k:k + width].mean(axis=0) < .95 - 1e-12
            alarm = history[k]
            counts += [np.count_nonzero(alarm & truth), np.count_nonzero(alarm & ~truth),
                       np.count_nonzero(~alarm & truth), np.count_nonzero(~alarm & ~truth)]
        rows.append(dict(detector=policy, tp=int(counts[0]), fp=int(counts[1]), fn=int(counts[2]), tn=int(counts[3])))
    return rows


def simulate(policy, scenario, seed, p, varied=False, replay=None):
    world = base.make_world(seed, 'uniform', p)
    if varied:
        rng = np.random.default_rng(seed + 700000)
        world['supply'] *= rng.uniform(.8, 1.2, p.n)
        world['demand'] *= rng.uniform(.8, 1.2, p.n)
        p = replace(p, conductance=p.conductance * rng.uniform(.8, 1.2),
                    leakage=p.leakage * rng.uniform(.8, 1.2), per_node_budget=p.per_node_budget * rng.uniform(.8, 1.2))
    supply0, demand0 = world['supply'].copy(), world['demand'].copy()
    consumers = ~world['sources']
    ids = np.flatnonzero(consumers)
    stock = np.full(p.n, p.initial_stock)
    estimates = np.full(len(world['u']), p.prior)
    observer = Observer(p.n, p.dt)
    steps = round(p.horizon / p.dt)
    served = np.zeros((steps, len(ids)))
    required = np.zeros_like(served)
    service = np.zeros_like(served)
    flag_history = {name: np.zeros_like(served, dtype=bool) for name in POLICIES[1:4]}
    last = np.full(p.n, -np.inf)
    events, requests, messages, swaps = [], 0, 0, 0
    by_step = {}
    if policy == 'shuffled':
        if replay is None:
            raise ValueError('Shuffled control requires a fixed event schedule')
        mapping = dict(zip(ids.tolist(), np.random.default_rng(seed + 800000).permutation(ids).tolist()))
        for step, node in replay:
            by_step.setdefault(step, []).append(mapping[node])
    max_error, max_flow = 0., 0.
    spend_by_node = np.zeros(p.n, dtype=int)
    for step in range(steps):
        t = step * p.dt
        if policy == 'autonomous':
            actors = []
            for name in flag_history:
                flag_history[name][step] = observer.flags(name)[consumers]
        elif policy == 'shuffled':
            actors = by_step.get(step, [])
        else:
            actors = np.flatnonzero(consumers & observer.flags(policy) & (t - last >= 8 - 1e-9))
        actors = sorted(map(int, actors), key=lambda a: (a * 2654435761 + step * 2246822519 + seed) % 4294967291)
        for a in actors:
            assert t - last[a] >= 8 - 1e-9
            last[a] = t
            msg, swap, fee = base.request_help(a, stock, estimates, world, seed, step, p)
            requests += 1
            messages += msg
            swaps += swap
            spend_by_node[a] += msg
            events.append((step, a))
        quality, world['supply'], world['demand'] = forcing(t, scenario, world, p, supply0, demand0)
        stock, consumption, _, error, flow = base.flow_step(stock, world['active'], estimates, world, quality, p)
        max_error, max_flow = max(error, max_error), max(flow, max_flow)
        served[step] = consumption[consumers]
        required[step] = world['demand'][consumers] * p.dt
        service[step] = served[step] / required[step]
        local_service = np.ones(p.n)
        local_service[consumers] = service[step]
        observer.update(local_service)
    start, tail, final = round(20 / p.dt), round(100 / p.dt), round(110 / p.dt)
    individual = served[start:].sum(axis=0) / required[start:].sum(axis=0)
    final_individual = served[final:].sum(axis=0) / required[final:].sum(axis=0)
    assert messages == 24 * requests + 8 * swaps
    assert all(len(x) == 4 for x in world['adjacency'])
    assert spend_by_node.max() <= 32 * (1 + int((p.horizon - p.dt) // 8))
    row = dict(n=p.n, scenario=scenario, seed=seed, dt=p.dt, varied=int(varied), policy=policy,
        service=float(served[start:].sum() / required[start:].sum()), p10=float(np.quantile(individual, .1)),
        tail_deficit=float(1 - served[tail:].sum() / required[tail:].sum()),
        final_unserved_nodes=int((final_individual < .95 - 1e-12).sum()),
        full_final_recovery=int(np.all(service[final:] >= .95 - 1e-12)),
        messages=messages, requests=requests, swaps=swaps, resource_cost=swaps * p.swap_cost,
        max_mass_error=max_error, max_flow_ratio=max_flow)
    diag = diagnostics(service, flag_history, p) if policy == 'autonomous' else []
    trace = dict(service=service, requested=np.array(events, dtype=int).reshape(-1, 2))
    if policy == 'autonomous':
        trace.update({f'flag_{k}': v for k, v in flag_history.items()})
    return row, events, diag, trace


def run_group(case):
    n, scenario, seed, dt, varied = case
    p = base.Params(n=n, dt=dt, horizon=120., swap_cost=.12)
    rows, diagnostics_rows, traces = [], [], {}
    schedule = None
    for policy in POLICIES:
        row, events, diag, trace = simulate(policy, scenario, seed, p, varied, schedule)
        rows.append(row)
        if policy == 'two_signals':
            schedule = events
        for d in diag:
            diagnostics_rows.append(dict(n=n, scenario=scenario, seed=seed, dt=dt, varied=int(varied), **d))
        if seed in (21000, 21100):
            traces.update({policy + '_' + k: v for k, v in trace.items()})
    assert rows[-1]['requests'] == rows[-2]['requests']
    return case, rows, diagnostics_rows, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = list(product((96, 384), SCENARIOS, range(21000, 21020), (.2,), (False,)))
    cases += list(product((96,), ('mixed',), range(21000, 21020), (.2,), (True,)))
    cases += list(product((96,), ('mixed',), range(21100, 21106), (.2, .1), (False,)))
    hashes = {}
    for path in (Path(__file__), Path(base.__file__), Path(__file__).with_name('test_minimal_observer.py'),
                 base.ROOT / 'manuscripts/v6/minimal_observer_protocol_2026-10-08.md'):
        data = path.read_bytes()
        (out / path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    (out / 'config.json').write_text(json.dumps(dict(cases=cases, policies=POLICIES, hashes=hashes,
        params=asdict(base.Params(horizon=120., swap_cost=.12)), python=platform.python_version(),
        numpy=np.__version__, primary='n96 mixed dt0.2 varied0: two_signals minus persistent service'), indent=2) + '\n', encoding='utf-8')
    all_rows, all_diag, started = [], [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (case, rows, diag, traces) in enumerate(pool.map(run_group, cases), 1):
            all_rows.extend(rows)
            all_diag.extend(diag)
            if traces:
                n, s, seed, dt, varied = case
                np.savez_compressed(out / f'trace_{n}_{s}_{seed}_{dt}_{int(varied)}.npz', **traces)
            if i % 20 == 0 or i == len(cases):
                print(f'{i}/{len(cases)} worlds; {time.perf_counter()-started:.1f}s', flush=True)
    base.write_csv(out / 'runs.csv', all_rows)
    base.write_csv(out / 'diagnostics.csv', all_diag)
    print(f'Saved {len(all_rows)} runs')


if __name__ == '__main__':
    main()
