"""Read-only archive audit for the local note; writes only evidence.json."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / 'research_results'
FILES = {}


def read_csv(folder, name):
    path = ARCHIVE / folder / name
    FILES[path.relative_to(ARCHIVE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def only(rows, **filters):
    selected = [r for r in rows if all(r[k] == str(v) for k, v in filters.items())]
    if len(selected) != 1:
        raise AssertionError((filters, len(selected)))
    return selected[0]


def clustered_mean(rows, metric):
    groups = defaultdict(list)
    for r in rows:
        groups[int(r['seed'])].append(float(r[metric]))
    return mean(mean(v) for v in groups.values())


def check_close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-11):
        raise AssertionError((a, b))


def state_key(row):
    return tuple(row[k] for k in ('seed', 't', 'node'))


def paired_mean(rows, a, b, metric):
    indexed = {(state_key(r), r['policy']): r for r in rows}
    diffs = []
    for r in rows:
        if r['policy'] == b:
            diff = float(indexed[state_key(r), a][metric]) - float(r[metric])
            diffs.append({'seed': r['seed'], 'value': diff})
    return clustered_mean(diffs, 'value')


def main():
    note = (HERE / 'note.ru.md').read_text(encoding='utf-8')
    checks = []

    def in_note(value, digits=2):
        formatted = f'{value:.{digits}f}'
        if formatted not in note:
            raise AssertionError(f'Expected rounded value absent from note: {formatted}')
        checks.append(formatted)

    observer = 'minimal_observer_2026-10-08'
    runs = read_csv(observer, 'runs.csv')
    selected = [r for r in runs if r['n'] == '96' and r['scenario'] == 'mixed'
                and r['dt'] == '0.2' and r['varied'] == '0' and int(r['seed']) < 21100]
    summary = read_csv(observer, 'summary.csv')
    pairs = read_csv(observer, 'paired_comparisons.csv')
    observer_means = {}
    for policy in ('autonomous', 'persistent', 'two_signals'):
        raw = [r for r in selected if r['policy'] == policy]
        assert len(raw) == 20 and len({r['seed'] for r in raw}) == 20
        saved = only(summary, n=96, scenario='mixed', dt=0.2, varied=0, cohort='main', policy=policy)
        values = {m: clustered_mean(raw, m) for m in ('service', 'messages', 'resource_cost', 'full_final_recovery')}
        for m, value in values.items():
            check_close(value, saved[m])
        in_note(values['service'] * 100)
        if policy != 'autonomous':
            in_note(values['resource_cost'], 3)
            assert f"{values['messages']:,.1f}".replace(',', ' ') in note
        observer_means[policy] = values
    primary_observer = only(pairs, n=96, scenario='mixed', dt=0.2, varied=0,
                            cohort='main', baseline='persistent', metric='service')
    diff = observer_means['two_signals']['service'] - observer_means['persistent']['service']
    check_close(diff, primary_observer['delta'])
    in_note(diff * 100, 3)
    in_note(float(primary_observer['lower']) * 100, 3)
    in_note(float(primary_observer['upper']) * 100, 4)
    message_reduction = 100 * (1 - observer_means['two_signals']['messages'] / observer_means['persistent']['messages'])
    in_note(message_reduction, 1)

    timing = 'help_timing_2026-10-08'
    timing_all = read_csv(timing, 'runs.csv')
    raw_timing = [r for r in timing_all if r['n'] == '96' and r['scenario'] == 'mixed'
                  and r['dt'] == '0.2' and r['category'] == 'improving' and int(r['seed']) < 23100]
    assert len(raw_timing) == 310
    assert len({state_key(r) for r in raw_timing}) == 62
    assert len({r['seed'] for r in raw_timing}) == 20
    timing_pairs = read_csv(timing, 'paired_comparisons.csv')
    primary_timing = only(timing_pairs, n=96, scenario='mixed', dt=0.2, category='improving',
                          cohort='main', policy='now', baseline='trend_rule')
    check_close(paired_mean(raw_timing, 'now', 'trend_rule', 'served_units'), primary_timing['delta'])
    timing_means = {}
    for policy in ('none', 'now', 'delay4', 'delay8', 'trend_rule'):
        raw = [r for r in raw_timing if r['policy'] == policy]
        values = {m: clustered_mean(raw, m) for m in ('service', 'requester_service', 'served_units')}
        values.update({m: sum(int(r[m]) for r in raw) for m in ('attempts', 'swaps')})
        in_note(values['service'] * 100)
        in_note(values['requester_service'] * 100)
        assert f"{values['attempts']} / {values['swaps']}" in note
        timing_means[policy] = values
    for k in ('delta', 'lower', 'upper'):
        in_note(float(primary_timing[k]), 3)
    for policy in ('delay4', 'delay8'):
        pair = only(timing_pairs, n=96, scenario='mixed', dt=0.2, category='improving',
                    cohort='main', policy=policy, baseline='now')
        check_close(paired_mean(raw_timing, policy, 'now', 'served_units'), pair['delta'])
        for k in ('delta', 'lower', 'upper'):
            in_note(float(pair[k]), 3)

    repeated = 'repeated_help_2026-10-08'
    repeated_all = read_csv(repeated, 'runs.csv')
    raw_repeated = [r for r in repeated_all if r['n'] == '96' and r['scenario'] == 'mixed'
                    and r['dt'] == '0.2' and int(r['seed']) < 24100]
    assert len(raw_repeated) == 192
    assert len({state_key(r) for r in raw_repeated}) == 48
    assert len({r['seed'] for r in raw_repeated}) == 19
    repeated_summary = read_csv(repeated, 'summary.csv')
    repeated_pairs = read_csv(repeated, 'paired_comparisons.csv')
    repeated_means = {}
    for policy in ('pause', 'persistent', 'pulse8', 'pulse16'):
        raw = [r for r in raw_repeated if r['policy'] == policy]
        saved = only(repeated_summary, n=96, scenario='mixed', dt=0.2, sample='main', subset='all', policy=policy)
        values = {m: clustered_mean(raw, m) for m in ('requester_service', 'restricted_recovery_time', 'messages')}
        for m, value in values.items():
            check_close(value, saved[m])
        in_note(values['requester_service'] * 100)
        in_note(values['restricted_recovery_time'])
        if policy != 'pause': in_note(values['messages'])
        repeated_means[policy] = values
    primary_repeated = only(repeated_pairs, n=96, scenario='mixed', dt=0.2, sample='main', subset='all',
                            contrast='pulse8-pause', metric='served_units')
    check_close(paired_mean(raw_repeated, 'pulse8', 'pause', 'served_units'), primary_repeated['mean'])
    for k in ('mean', 'lo', 'hi'):
        in_note(float(primary_repeated[k]), 3)
    for metric, digits in (('served_units', 3), ('restricted_recovery_time', 2)):
        pair = only(repeated_pairs, n=96, scenario='mixed', dt=0.2, sample='main', subset='all',
                    contrast='pulse8-persistent', metric=metric)
        check_close(paired_mean(raw_repeated, 'pulse8', 'persistent', metric), pair['mean'])
        for k in ('mean', 'lo', 'hi'):
            in_note(float(pair[k]), digits)
    in_note(repeated_means['pulse8']['messages'] / repeated_means['persistent']['messages'], 1)

    example = read_csv(repeated, 'example_step.csv')
    base = only(example, n=96, scenario='transient', seed=24000, dt=0.2, t=40.0, node=31, policy='pause')
    retry = only(example, n=96, scenario='transient', seed=24000, dt=0.2, t=40.0, node=31, policy='pulse8')
    in_note(float(base['requester_service']) * 100)
    in_note(float(retry['requester_service']) * 100)
    gain = float(retry['served_units']) - float(base['served_units'])
    in_note(gain, 3)
    for step in (0.2, 0.1, 0.05):
        a = only(example, n=96, scenario='transient', seed=24000, dt=step, t=40.0, node=31, policy='pause')
        b = only(example, n=96, scenario='transient', seed=24000, dt=step, t=40.0, node=31, policy='pulse8')
        assert a['recovered'] == '1' and b['recovered'] == '0'
        in_note(float(a['restricted_recovery_time']))

    parameter_specification = dict(n=96, dt=0.2, radius=6, storage=2.0, initial_stock=0.7,
        source_rate=0.55, demand_rate=0.15, conductance=0.55, per_node_budget=0.65,
        leakage=0.003, good_quality=0.98, bad_quality=0.20, bad_share=0.35, prior=0.75,
        swap_cost=0.12, helper_reserve=0.30, helper_margin=0.10, request_interval=8.0)
    for folder in (observer, timing, repeated):
        path = ARCHIVE / folder / 'config.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        assert config['python'] == '3.12.14' and config['numpy'] == '2.3.5'
        for key, value in parameter_specification.items():
            check_close(config['params'][key], value)
        FILES[path.relative_to(ARCHIVE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    figure = ARCHIVE / repeated / 'local_counterexample.png'
    FILES[figure.relative_to(ARCHIVE).as_posix()] = hashlib.sha256(figure.read_bytes()).hexdigest()
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text(encoding='utf-8'))
    expected = {r['path']: r['sha256'] for r in manifest['files']}
    for name, digest in FILES.items():
        assert expected[name] == digest, name
    output = dict(review_date='2026-11-17', base_revision='a4c8880e469445cb203ee157d52990b2e9b6d5bc',
                  note_sha256=hashlib.sha256((HERE / 'note.ru.md').read_bytes()).hexdigest(),
                  validation='Means and paired contrasts recomputed from archived runs; intervals read from archived tables; no full simulation or bootstrap rerun.',
                  rounded_values_checked=len(checks), input_sha256=FILES,
                  model_parameters_checked_against_three_configs=parameter_specification,
                  observer=dict(policy_means=observer_means, primary=primary_observer),
                  timing=dict(policy_means=timing_means, primary=primary_timing),
                  repeated=dict(policy_means=repeated_means, primary=primary_repeated),
                  counterexample=dict(pause=base, pulse8=retry, network_gain=gain))
    (HERE / 'evidence.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rounded_values_checked': len(checks), 'archive_files_verified': len(FILES),
                      'primary_means_recomputed': 3, 'result': 'pass'}))


if __name__ == '__main__':
    main()
