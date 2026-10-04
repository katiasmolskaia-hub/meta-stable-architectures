"""Continue or pause rewiring after recovery, then expose both copies to a shock."""
import argparse
import csv
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from bond_recovery_model import run_episode, THRESHOLD, STABLE_WINDOW, QUIET_DURATION, SECOND_DURATION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bond_recovery_stop_2026-10-04"
SIZES = (24, 96)
COSTS = (1, 20)
SCENARIOS = ("persistent_damage", "temporary_damage")
POLICIES = ("recent_contact", "experience_swap")


def execute(case):
    n, cost, scenario, policy, seed, dt, noise, trace = case
    return run_episode(n, cost, scenario, policy, seed, dt, noise, trace)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--step-check", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "step_" if args.step_check else ""
    if (args.output / (prefix + "episodes.csv")).exists():
        parser.error("Refusing to overwrite episodes; choose a new --output")
    if args.step_check:
        cases = [(96, 20, "persistent_damage", p, seed, dt, 0., False)
                 for p, seed, dt in product(POLICIES, range(8900, 8910), (.2, .1))]
    else:
        cases = [(n, c, s, p, seed, .2, .03, seed == 8000)
                 for n, c, s, p, seed in product(SIZES, COSTS, SCENARIOS, POLICIES, range(8000, 8030))]
    hashes = {}
    for name in ("bond_recovery_model.py", "experiment_bond_recovery_stop.py", "bond_rewiring_scaled_model.py"):
        source = (Path(__file__).parent / name).read_bytes()
        (args.output / (prefix + name)).write_bytes(source)
        hashes[name] = hashlib.sha256(source).hexdigest()
    (args.output / (prefix + "config.json")).write_text(json.dumps(dict(
        cases=len(cases), threshold=THRESHOLD, stable_window=STABLE_WINDOW,
        quiet_duration=QUIET_DURATION, second_duration=SECOND_DURATION,
        cases_spec=cases, sha256=hashes, python=platform.python_version(), numpy=np.__version__
    ), indent=2), encoding="utf-8")
    episodes, branches, traces = [], [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, (episode, paired, trace) in enumerate(executor.map(execute, cases, chunksize=2), 1):
            episodes.append(episode)
            branches.extend(paired)
            traces.extend(trace)
            if i % 40 == 0 or i == len(cases):
                print(f"Completed {i}/{len(cases)} initial networks; recovered {len(branches)//2}; elapsed {time.perf_counter()-start:.1f}s", flush=True)
                write_csv(args.output / (prefix + "episodes.csv"), episodes)
                write_csv(args.output / (prefix + "branches.csv"), branches)
    write_csv(args.output / (prefix + "traces.csv"), traces)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
