"""Counterfactual timing of one bounded help attempt on a passive trajectory."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import time
import numpy as np
import experiment_local_support_signal as base
import experiment_minimal_observer as prior

OUT = base.ROOT / 'outputs/help_timing_2026-10-08'
POLICIES = ('none', 'now', 'delay4', 'delay8', 'trend_rule')
TIMES = (40., 52., 64., 76., 88., 100.)


def snapshot(world, stock, estimates, observer):
    return dict(world=copy.deepcopy(world), stock=stock.copy(), estimates=estimates.copy(), observer=copy.deepcopy(observer))


def advance(state, t, scenario, p, supply0, demand0):
    w = state['world']
    quality, w['supply'], w['demand'] = prior.forcing(t, scenario, w, p, supply0, demand0)
    stock, used, _, error, ratio = base.flow_step(state['stock'], w['active'], state['estimates'], w, quality, p)
    state['stock'] = stock
    consumers = ~w['sources']
    demand = w['demand'] * p.dt
    service = np.ones(p.n)
    service[consumers] = used[consumers] / demand[consumers]
    state['observer'].update(service)
    return used, demand, service, error, ratio


def branch(state0, scenario, seed, t0, node, policy, p, supply0, demand0):
    if policy not in POLICIES:
        raise ValueError(policy)
    state = copy.deepcopy(state0)
    consumers = ~state['world']['sources']
    used_total = np.zeros(p.n)
    required_total = np.zeros(p.n)
    attempt, swap, messages, fee, when = 0, 0, 0, 0., -1.
    final_good = True
    max_error, max_flow = 0., 0.
    # Common random preference for all times; stock/estimates still evolve.
    selector_step = round(t0 / .2)
    for k in range(round(16 / p.dt)):
        elapsed = k * p.dt
        act = ((policy == 'now' and k == 0) or
               (policy == 'delay4' and k == round(4 / p.dt)) or
               (policy == 'delay8' and k == round(8 / p.dt)) or
               (policy == 'trend_rule' and elapsed <= 8 + 1e-9 and state['observer'].flags('two_signals')[node]))
        if act and not attempt:
            messages, swap, fee = base.request_help(node, state['stock'], state['estimates'], state['world'], seed, selector_step, p)
            attempt, when = 1, elapsed
        used, required, service, error, ratio = advance(state, t0 + elapsed, scenario, p, supply0, demand0)
        used_total += used
        required_total += required
        max_error, max_flow = max(max_error, error), max(max_flow, ratio)
        if elapsed >= 12 - 1e-9:
            final_good &= bool(np.all(service[consumers] >= .95 - 1e-12))
    return dict(policy=policy, served_units=float(used_total.sum()),
        service=float(used_total.sum() / required_total.sum()),
        requester_service=float(used_total[node] / required_total[node]),
        other_served_units=float(used_total.sum() - used_total[node]),
        requester_served_units=float(used_total[node]), full_final_recovery=int(final_good),
        attempts=attempt, swaps=swap, messages=messages, fee=fee, attempt_delay=when,
        max_mass_error=max_error, max_flow_ratio=max_flow)


def run_world(case):
    n, scenario, seed, dt = case
    p = base.Params(n=n, dt=dt, horizon=120., swap_cost=.12)
    world = base.make_world(seed, 'uniform', p)
    supply0, demand0 = world['supply'].copy(), world['demand'].copy()
    state = dict(world=world, stock=np.full(n, p.initial_stock), estimates=np.full(len(world['u']), p.prior),
                 observer=prior.Observer(n, dt))
    consumers = ~world['sources']
    rows, selections, saved = [], [], {}
    time_steps = {round(t / dt): t for t in TIMES}
    for step in range(round(max(TIMES) / dt) + 1):
        t = step * dt
        if step in time_steps:
            o = state['observer']
            persistent = o.flags('persistent') & consumers
            stalled = o.flags('two_signals') & persistent
            masks = {'improving': persistent & ~stalled, 'stalled': stalled}
            chooser = np.random.default_rng(seed + round(t) * 100000)
            chosen = []
            for label, mask in masks.items():
                candidates = np.flatnonzero(mask)
                node = int(chooser.choice(candidates)) if len(candidates) else -1
                selections.append(dict(n=n, scenario=scenario, seed=seed, dt=dt, t=t,
                    category=label, candidate_count=len(candidates), node=node))
                if node >= 0:
                    chosen.append((label, node))
            if chosen:
                prefix = f't{int(t)}_'
                saved.update({prefix + 'stock': state['stock'].copy(), prefix + 'estimates': state['estimates'].copy(),
                    prefix + 'q': o.q.copy(), prefix + 'below': o.below.copy(), prefix + 'trend': o.trend.copy(),
                    prefix + 'buffer': o.buffer.copy(), prefix + 'count': np.array(o.count)})
                initial = snapshot(world, state['stock'], state['estimates'], o)
                for category, node in chosen:
                    for policy in POLICIES:
                        result = branch(initial, scenario, seed, t, node, policy, p, supply0, demand0)
                        rows.append(dict(n=n, scenario=scenario, seed=seed, dt=dt, t=t, category=category, node=node, **result))
            # Counterfactuals must not mutate the autonomous timeline.
        if step < round(max(TIMES) / dt):
            advance(state, t, scenario, p, supply0, demand0)
    return case, rows, selections, saved


def restore_checkpoint(out, case, t):
    n, scenario, seed, dt = case
    p = base.Params(n=n, dt=dt, horizon=120., swap_cost=.12)
    w = base.make_world(seed, 'uniform', p)
    o = prior.Observer(n, dt)
    prefix = f't{int(t)}_'
    with np.load(out / f'states_{n}_{scenario}_{seed}_{dt}.npz') as f:
        for name in ('q', 'below', 'trend', 'buffer'):
            setattr(o, name, f[prefix + name].copy())
        o.count = int(f[prefix + 'count'])
        state = dict(world=w, stock=f[prefix + 'stock'].copy(), estimates=f[prefix + 'estimates'].copy(), observer=o)
    return state, p, w['supply'].copy(), w['demand'].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = list(product((96, 384), ('mixed', 'transient', 'scarcity'), range(23000, 23020), (.2,)))
    cases += list(product((96,), ('mixed',), range(23100, 23104), (.2, .1)))
    hashes = {}
    for path in (Path(__file__), Path(base.__file__), Path(prior.__file__),
                 Path(__file__).with_name('test_help_timing.py'),
                 base.ROOT / 'manuscripts/v6/help_timing_protocol_2026-10-08.md'):
        data = path.read_bytes()
        (args.out / path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    (args.out / 'config.json').write_text(json.dumps(dict(cases=cases, policies=POLICIES, times=TIMES,
        params=asdict(base.Params(horizon=120., swap_cost=.12)), hashes=hashes,
        python=platform.python_version(), numpy=np.__version__,
        primary='n96 mixed improving main: now minus trend_rule served_units, first averaged within seed'), indent=2)+'\n', encoding='utf-8')
    rows, selected, started = [], [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (case, result, selections, saved) in enumerate(pool.map(run_world, cases), 1):
            rows.extend(result)
            selected.extend(selections)
            n, scenario, seed, dt = case
            np.savez_compressed(args.out / f'states_{n}_{scenario}_{seed}_{dt}.npz', **saved)
            if i % 16 == 0 or i == len(cases):
                print(f'{i}/{len(cases)} worlds; {len(rows)} branches; {time.perf_counter()-started:.1f}s', flush=True)
    base.write_csv(args.out / 'runs.csv', rows)
    base.write_csv(args.out / 'selections.csv', selected)


if __name__ == '__main__':
    main()
