"""Paired analysis, grid audit and selected deterministic replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import experiment_minimal_observer as experiment
import experiment_local_support_signal as base


def report(out):
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    for name, expected in config['hashes'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    for module in (experiment, base):
        live = Path(module.__file__)
        assert live.read_bytes().replace(b'\r\n', b'\n') == (out / live.name).read_bytes().replace(b'\r\n', b'\n')
    with (out / 'runs.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    def key(r):
        return (int(r['n']), r['scenario'], int(r['seed']), float(r['dt']), bool(int(r['varied'])), r['policy'])
    indexed = {key(r): r for r in rows}
    expected = {tuple(c) + (p,) for c in config['cases'] for p in config['policies']}
    assert len(indexed) == len(rows) == len(expected) and set(indexed) == expected
    for r in rows:
        assert 0 <= float(r['service']) <= 1 + 1e-12
        assert float(r['max_mass_error']) < 1e-8
        assert float(r['max_flow_ratio']) <= 1 + 1e-9
        assert int(r['messages']) == 24 * int(r['requests']) + 8 * int(r['swaps'])
    for c in config['cases']:
        assert indexed[tuple(c) + ('two_signals',)]['requests'] == indexed[tuple(c) + ('shuffled',)]['requests']
    replay_count = 0
    for case in ((96, 'mixed', 21000, .2, False), (384, 'mixed', 21000, .2, False),
                 (96, 'mixed', 21100, .1, False)):
        _, replayed, _, _ = experiment.run_group(case)
        for r in replayed:
            assert indexed[key(r)] == {k: str(v) for k, v in r.items()}, key(r)
            replay_count += 1
    def group_key(r):
        return (int(r['n']), r['scenario'], float(r['dt']), int(r['varied']),
                'main' if int(r['seed']) < 21100 else 'step')
    groups = sorted({group_key(r) for r in rows})
    summaries, comparisons = [], []
    metrics = ('service', 'p10', 'tail_deficit', 'final_unserved_nodes', 'full_final_recovery',
               'requests', 'messages', 'swaps', 'resource_cost')
    for gidx, g in enumerate(groups):
        block = [r for r in rows if group_key(r) == g]
        by_policy = {p: sorted([r for r in block if r['policy'] == p], key=lambda r: int(r['seed']))
                     for p in experiment.POLICIES}
        values = {p: {m: np.array([float(r[m]) for r in records]) for m in metrics}
                  for p, records in by_policy.items()}
        fields = dict(n=g[0], scenario=g[1], dt=g[2], varied=g[3], cohort=g[4])
        count = len(by_policy['autonomous'])
        indices = np.random.default_rng(22000 + gidx).integers(0, count, (5000, count))
        for p in experiment.POLICIES:
            summaries.append(dict(**fields, policy=p, worlds=count,
                                  **{m: float(v.mean()) for m, v in values[p].items()}))
        for other in ('autonomous', 'level', 'persistent', 'shuffled'):
            for m in metrics:
                delta = values['two_signals'][m] - values[other][m]
                lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                comparisons.append(dict(**fields, baseline=other, metric=m, delta=float(delta.mean()),
                    lower=float(lo), upper=float(hi), primary=int(g == (96, 'mixed', .2, 0, 'main')
                                                               and other == 'persistent' and m == 'service')))
    with (out / 'diagnostics.csv').open(encoding='utf-8', newline='') as f:
        diagnostics = list(csv.DictReader(f))
    assert len(diagnostics) == 3 * len(config['cases'])
    diag_summary = []
    for g in groups:
        for detector in ('level', 'persistent', 'two_signals'):
            selected = [r for r in diagnostics if group_key(r) == g and r['detector'] == detector]
            counts = {m: sum(int(r[m]) for r in selected) for m in ('tp', 'fp', 'fn', 'tn')}
            diag_summary.append(dict(n=g[0], scenario=g[1], dt=g[2], varied=g[3], cohort=g[4],
                detector=detector, **counts, precision=counts['tp'] / max(1, counts['tp'] + counts['fp']),
                recall=counts['tp'] / max(1, counts['tp'] + counts['fn']),
                false_positive_rate=counts['fp'] / max(1, counts['fp'] + counts['tn'])))
    for name, records in (('summary.csv', summaries), ('paired_comparisons.csv', comparisons),
                           ('diagnostic_summary.csv', diag_summary)):
        base.write_csv(out / name, records)
    audit = dict(rows=len(rows), paired_worlds=len(config['cases']), complete_grid=True,
        deterministic_replayed_rows=replay_count, code_and_protocol_hashes_verified=True,
        max_mass_error=max(float(r['max_mass_error']) for r in rows),
        full_final_recovery_runs=sum(int(r['full_final_recovery']) for r in rows))
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    (out / 'report_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(audit=audit,
        primary=next(r for r in comparisons if r['primary']),
        mixed_summaries=[r for r in summaries if r['scenario'] == 'mixed'],
        main_service_comparisons=[r for r in comparisons if r['baseline'] == 'persistent' and r['metric'] == 'service'],
        mixed_diagnostics=[r for r in diag_summary if r['scenario'] == 'mixed' and r['n'] == 96 and r['cohort'] == 'main' and not r['varied']]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=experiment.OUT)
    report(parser.parse_args().out)
