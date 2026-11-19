"""One fixed observer example; no archive writes and no population inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'simulations'))
import experiment_local_support_signal as base
import experiment_minimal_observer as model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=21000)
    args = parser.parse_args()
    archive = ROOT / 'research_results/minimal_observer_2026-10-08'
    config = json.loads((archive / 'config.json').read_text(encoding='utf-8'))
    for name in ('experiment_local_support_signal.py', 'experiment_minimal_observer.py'):
        live = (ROOT / 'simulations' / name).read_bytes().replace(b'\r\n', b'\n')
        saved = (archive / name).read_bytes().replace(b'\r\n', b'\n')
        if live != saved:
            raise RuntimeError(f'Working module differs from archived code: {name}')
        if hashlib.sha256((archive / name).read_bytes()).hexdigest() != config['hashes'][name]:
            raise RuntimeError(f'Archive snapshot hash mismatch: {name}')
    p = base.Params(**config['params'])
    with (archive / 'runs.csv').open(encoding='utf-8', newline='') as stream:
        saved_rows = list(csv.DictReader(stream))
    print('One fixed world: n=96, mixed, dt=0.2. This is not a statistical comparison.')
    print('policy            service_pct   messages   swaps   fee   archive_match')
    for policy in ('autonomous', 'persistent', 'two_signals'):
        row, _, _, _ = model.simulate(policy, 'mixed', args.seed, p)
        candidates = [r for r in saved_rows if r['n'] == '96' and r['scenario'] == 'mixed'
                      and r['dt'] == '0.2' and r['varied'] == '0'
                      and int(r['seed']) == args.seed and r['policy'] == policy]
        match = 'not_in_archive'
        if candidates:
            fields = ('service', 'messages', 'requests', 'swaps', 'resource_cost',
                      'full_final_recovery', 'max_mass_error', 'max_flow_ratio')
            errors = {key: abs(float(row[key]) - float(candidates[0][key])) for key in fields}
            if max(errors.values()) > 1e-10:
                raise AssertionError(f'Archived row not reproduced: {policy}: {errors}')
            match = 'yes'
        print(f'{policy:17s} {row["service"]*100:10.4f} {row["messages"]:10d}'
              f' {row["swaps"]:7d} {row["resource_cost"]:6.3f} {match}')
    print('No files were written. Use all archived seeds for study-level conclusions.')


if __name__ == '__main__':
    main()
