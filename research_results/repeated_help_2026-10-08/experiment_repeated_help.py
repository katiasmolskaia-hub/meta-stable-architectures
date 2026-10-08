"""Repeated local help following one mechanically successful swap."""
import argparse
import copy
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np
import experiment_local_support_signal as base
import experiment_minimal_observer as prior
import experiment_help_timing as timing

OUT = base.ROOT / 'outputs/repeated_help_2026-10-08'
POLICIES = ('pause', 'persistent', 'pulse8', 'pulse16')
TIMES = (40., 52., 64., 76., 88., 100.)
HORIZON = 48.


def recovery(service, dt):
    good = np.asarray(service) >= .95 - 1e-12
    bad = np.flatnonzero(~good)
    start = int(bad[-1] + 1) if len(bad) else 0
    recovered = len(good) - start >= round(4 / dt)
    return int(recovered), start * dt if recovered else len(good) * dt


def branch(initial, scenario, seed, t0, node, policy, p, supply, demand):
    if policy not in POLICIES:
        raise ValueError(policy)
    state = copy.deepcopy(initial)
    consumers = ~state['world']['sources']
    used_total, required_total = np.zeros(p.n), np.zeros(p.n)
    last = 0.
    events, trace = [], []
    fee, messages, swaps, max_error, max_flow = 0., 0, 0, 0., 0.
    full_final = True
    for k in range(round(HORIZON / p.dt)):
        elapsed = k * p.dt
        need = bool(state['observer'].flags('persistent')[node])
        pulse = (policy == 'pulse8' and k > 0 and k % round(8 / p.dt) == 0) or (
            policy == 'pulse16' and k > 0 and k % round(16 / p.dt) == 0)
        act = pulse or (policy == 'persistent' and need and elapsed - last >= 8 - 1e-9)
        if act:
            msg, swap, cost = base.request_help(node, state['stock'], state['estimates'], state['world'],
                                               seed, round((t0 + elapsed) / .2), p)
            events.append(dict(delay=elapsed, need=int(need), q=float(state['observer'].q[node]),
                               messages=msg, swap=swap, fee=cost))
            messages += msg
            swaps += swap
            fee += cost
            last = elapsed
        used, required, service, error, flow = timing.advance(state, t0 + elapsed, scenario, p, supply, demand)
        used_total += used
        required_total += required
        max_error, max_flow = max(max_error, error), max(max_flow, flow)
        trace.append((service[node], used.sum(), required.sum(), int(need)))
        if elapsed >= HORIZON - 4 - 1e-9:
            full_final &= bool(np.all(service[consumers] >= .95 - 1e-12))
    trace = np.asarray(trace)
    recovered, tail = recovery(trace[:, 0], p.dt)
    result = dict(policy=policy, served_units=float(used_total.sum()),
        service=float(used_total.sum() / required_total.sum()),
        requester_service=float(used_total[node] / required_total[node]),
        requester_deficit=float(required_total[node] - used_total[node]),
        other_served_units=float(used_total.sum() - used_total[node]),
        recovered=recovered, restricted_recovery_time=tail, full_final_recovery=int(full_final),
        attempts=len(events), swaps=swaps, messages=messages, fee=fee,
        max_mass_error=max_error, max_flow_ratio=max_flow)
    return result, events, trace


def initial_world(seed, p):
    w = base.make_world(seed, 'uniform', p)
    return dict(world=w, stock=np.full(p.n, p.initial_stock),
                estimates=np.full(len(w['u']), p.prior), observer=prior.Observer(p.n, p.dt))


def run_world(case):
    n, scenario, seed, dt = case
    p = base.Params(n=n, dt=dt, horizon=160., swap_cost=.12)
    state = initial_world(seed, p)
    supply, demand = state['world']['supply'].copy(), state['world']['demand'].copy()
    consumers = ~state['world']['sources']
    checkpoints = {round(t / dt): t for t in TIMES}
    selections, rows, events, traces = [], [], [], {}
    for step in range(round(max(TIMES) / dt) + 1):
        if step in checkpoints:
            t = checkpoints[step]
            candidates = np.flatnonzero(state['observer'].flags('persistent') & consumers)
            rng = np.random.default_rng(seed + round(t) * 100000)
            node = int(rng.choice(candidates)) if len(candidates) else -1
            fields = dict(n=n, scenario=scenario, seed=seed, dt=dt, t=t, node=node)
            initial = copy.deepcopy(state)
            msg, swap, cost = (base.request_help(node, initial['stock'], initial['estimates'], initial['world'],
                                seed, round(t / .2), p) if node >= 0 else (0, 0, 0.))
            selections.append(dict(**fields, candidates=len(candidates), first_messages=msg, first_swap=swap, first_fee=cost))
            if swap:
                for policy in POLICIES:
                    result, ev, tr = branch(initial, scenario, seed, t, node, policy, p, supply, demand)
                    rows.append(dict(**fields, **result))
                    events.extend(dict(**fields, policy=policy, **e) for e in ev)
                    traces[f't{int(t)}_{policy}'] = tr
        if step < round(max(TIMES) / dt):
            timing.advance(state, step * dt, scenario, p, supply, demand)
    return case, selections, rows, events, traces


def reconstruct(case, t, node):
    n, scenario, seed, dt = case
    p = base.Params(n=n, dt=dt, horizon=160., swap_cost=.12)
    state = initial_world(seed, p)
    supply, demand = state['world']['supply'].copy(), state['world']['demand'].copy()
    for k in range(round(t / dt)):
        timing.advance(state, k * dt, scenario, p, supply, demand)
    msg, swap, cost = base.request_help(node, state['stock'], state['estimates'], state['world'], seed, round(t / .2), p)
    assert swap == 1
    return state, p, supply, demand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = list(product((96, 384), ('mixed', 'transient'), range(24000, 24020), (.2,)))
    cases += list(product((96,), ('mixed',), range(24100, 24104), (.2, .1)))
    hashes = {}
    for path in (Path(__file__), Path(base.__file__), Path(prior.__file__), Path(timing.__file__),
                 Path(__file__).with_name('test_repeated_help.py'),
                 base.ROOT / 'manuscripts/v6/repeated_help_protocol_2026-10-08.md'):
        data = path.read_bytes()
        (args.out / path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    (args.out / 'config.json').write_text(json.dumps(dict(cases=cases, policies=POLICIES, times=TIMES,
        horizon=HORIZON, params=asdict(base.Params(horizon=160., swap_cost=.12)), hashes=hashes,
        trace_columns=['requester_service', 'network_used', 'network_required', 'need_before_step'],
        python=platform.python_version(), numpy=np.__version__,
        primary='n96 mixed main: pulse8 minus pause served_units, average within seed first'), indent=2)+'\n', encoding='utf-8')
    selections, rows, events, started = [], [], [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (case, sel, result, ev, trace) in enumerate(pool.map(run_world, cases), 1):
            selections.extend(sel)
            rows.extend(result)
            events.extend(ev)
            n, scenario, seed, dt = case
            np.savez_compressed(args.out / f'traces_{n}_{scenario}_{seed}_{dt}.npz', **trace)
            if i % 8 == 0 or i == len(cases):
                print(f'{i}/{len(cases)} worlds; {len(rows)} branches; {time.perf_counter()-started:.1f}s', flush=True)
    base.write_csv(args.out / 'selections.csv', selections)
    base.write_csv(args.out / 'runs.csv', rows)
    base.write_csv(args.out / 'events.csv', events)


if __name__ == '__main__':
    main()
