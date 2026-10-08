"""Within-state causal contrasts, clustered by seed; oracle is descriptive only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import experiment_help_timing as model
import experiment_local_support_signal as base
import experiment_minimal_observer as prior


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def state_key(r):
    return (int(r['n']), r['scenario'], int(r['seed']), float(r['dt']), float(r['t']), r['category'], int(r['node']))


def group_key(r):
    return (int(r['n']), r['scenario'], float(r['dt']), r['category'],
            'main' if int(r['seed']) < 23100 else 'step')


def report(out):
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    for name, h in config['hashes'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == h
    for module in (model, base, prior):
        p = Path(module.__file__)
        assert p.read_bytes().replace(b'\r\n', b'\n') == (out / p.name).read_bytes().replace(b'\r\n', b'\n')
    rows, selections = read(out / 'runs.csv'), read(out / 'selections.csv')
    assert len(selections) == len(config['cases']) * len(model.TIMES) * 2
    expected_selection_keys = {tuple(c) + (t, label) for c in config['cases'] for t in model.TIMES
                               for label in ('improving', 'stalled')}
    assert {(int(r['n']), r['scenario'], int(r['seed']), float(r['dt']), float(r['t']), r['category'])
            for r in selections} == expected_selection_keys
    chosen = {state_key(r) for r in selections if int(r['node']) >= 0}
    indexed = {(state_key(r), r['policy']): r for r in rows}
    assert len(indexed) == len(rows) == len(chosen) * len(model.POLICIES)
    assert set(indexed) == {(k, policy) for k in chosen for policy in model.POLICIES}
    for r in rows:
        assert int(r['attempts']) <= 1 and int(r['swaps']) <= int(r['attempts'])
        assert int(r['messages']) == 24 * int(r['attempts']) + 8 * int(r['swaps'])
        assert abs(float(r['fee']) - .12 * int(r['swaps'])) < 1e-12
        assert float(r['max_mass_error']) < 1e-8 and float(r['max_flow_ratio']) <= 1 + 1e-9
    # Exact replay from independent serialized states, all five policies.
    replay_count = 0
    replay_keys = []
    for group in sorted({group_key(r) for r in rows}):
        keys = sorted(k for k in chosen if (k[0], k[1], k[3], k[5], 'main' if k[2] < 23100 else 'step') == group)
        if keys:
            replay_keys.append(keys[0])
    for key in replay_keys:
        n, scenario, seed, dt, t, category, node = key
        state, p, supply, demand = model.restore_checkpoint(out, (n, scenario, seed, dt), t)
        for policy in model.POLICIES:
            replay = model.branch(state, scenario, seed, t, node, policy, p, supply, demand)
            saved = indexed[key, policy]
            assert all(saved[k] == str(v) for k, v in replay.items()), (key, policy)
            replay_count += 1
    opportunities = []
    metrics = ('served_units', 'service', 'requester_service', 'other_served_units', 'full_final_recovery')
    deltas = []
    for key in sorted(chosen):
        candidates = [indexed[key, p] for p in model.POLICIES]
        no = candidates[0]
        best = max(candidates, key=lambda r: float(r['served_units']))
        fields = {k: no[k] for k in ('n', 'scenario', 'seed', 'dt', 't', 'category', 'node')}
        gains = {r['policy']: float(r['served_units']) - float(no['served_units']) for r in candidates}
        requester_gains = {r['policy']: float(r['requester_service']) - float(no['requester_service']) for r in candidates}
        opportunity = dict(**fields, any_swap=int(any(int(r['swaps']) for r in candidates)),
            any_network_gain=int(max(gains.values()) > .01),
            any_requester_gain=int(max(requester_gains.values()) >= .05),
            any_both_gain=int(any(gains[p] > .01 and requester_gains[p] >= .05 for p in model.POLICIES)),
            oracle_policy=best['policy'], oracle_gain=gains[best['policy']],
            oracle_requester_gain=requester_gains[best['policy']],
            missed_by_trend=int(max(gains.values()) > .01 and gains['trend_rule'] <= .01))
        for policy in model.POLICIES[1:]:
            opportunity[policy + '_gain'] = gains[policy]
            opportunity[policy + '_helpful'] = int(gains[policy] > .01)
            opportunity[policy + '_harmful'] = int(gains[policy] < -.01)
            row = indexed[key, policy]
            deltas.append(dict(**fields, policy=policy,
                **{m + '_delta': float(row[m]) - float(no[m]) for m in metrics},
                attempts=int(row['attempts']), swaps=int(row['swaps']), messages=int(row['messages']),
                fee=float(row['fee']), attempt_delay=float(row['attempt_delay'])))
        if key[5] == 'stalled':
            a, b = indexed[key, 'now'], indexed[key, 'trend_rule']
            assert all(a[k] == b[k] for k in a if k != 'policy')
        opportunities.append(opportunity)
    base.write_csv(out / 'opportunities.csv', opportunities)
    base.write_csv(out / 'deltas.csv', deltas)
    summaries, contrasts = [], []
    groups = sorted({group_key(r) for r in rows})
    for gi, group in enumerate(groups):
        records = [r for r in opportunities if group_key(r) == group]
        seeds = sorted({int(r['seed']) for r in records})
        fields = dict(n=group[0], scenario=group[1], dt=group[2], category=group[3], cohort=group[4],
                      eligible_seeds=len(seeds), checkpoints=len(records))
        def seed_means(key):
            return np.array([np.mean([float(r[key]) for r in records if int(r['seed']) == s]) for s in seeds])
        names = ('any_swap', 'any_network_gain', 'any_requester_gain', 'any_both_gain', 'oracle_gain',
                 'oracle_requester_gain', 'missed_by_trend') + tuple(p + suffix for p in model.POLICIES[1:]
                                                                    for suffix in ('_gain', '_helpful', '_harmful'))
        summaries.append(dict(**fields, **{name: float(seed_means(name).mean()) for name in names}))
        indices = np.random.default_rng(24000 + gi).integers(0, len(seeds), (5000, len(seeds)))
        for policy, baseline in (('now', 'trend_rule'), ('delay4', 'now'), ('delay8', 'now'), ('now', 'none'),
                                 ('delay4', 'none'), ('delay8', 'none'), ('trend_rule', 'none')):
            a = seed_means(policy + '_gain')
            b = np.zeros(len(seeds)) if baseline == 'none' else seed_means(baseline + '_gain')
            diff = a - b
            lo, hi = np.quantile(diff[indices].mean(axis=1), [.025, .975])
            contrasts.append(dict(**fields, policy=policy, baseline=baseline, delta=float(diff.mean()),
                lower=float(lo), upper=float(hi),
                primary=int(group == (96, 'mixed', .2, 'improving', 'main') and policy == 'now' and baseline == 'trend_rule')))
    base.write_csv(out / 'summary.csv', summaries)
    base.write_csv(out / 'paired_comparisons.csv', contrasts)
    audit = dict(configurations=len(config['cases']), selection_slots=len(selections), selected_checkpoints=len(chosen),
        branches=len(rows), empty_slots=sum(int(r['node']) < 0 for r in selections), exact_replayed_branches=replay_count,
        complete_grid=True, hashes_verified=True, max_mass_error=max(float(r['max_mass_error']) for r in rows))
    (out / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    (out / 'report_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(audit=audit, primary=[r for r in contrasts if r['primary']],
        summaries=summaries,
        timing_contrasts=[r for r in contrasts if r['n']==96 and r['scenario']=='mixed' and r['category']=='improving']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=model.OUT)
    report(parser.parse_args().out)
