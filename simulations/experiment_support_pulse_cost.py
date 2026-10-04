"""Close the pulse-cost hypothesis without retuning the two schedules."""
from __future__ import annotations

import argparse
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
import experiment_support_pulse as pulse

OUT = base.ROOT / "outputs" / "support_pulse_cost_2026-10-04"
COSTS = (1, 5, 20)
RATES = {"fixed": 4., "pulse_only": 2.3, "hybrid_50": 4.}


def run_group(case):
    n, scenario, seed, dt = case
    rows = []
    # One unchanged-network control is reused across prices, not counted as
    # independent evidence three times.
    settings = [("fixed", 0)] + [(p, c) for c in COSTS for p in ("pulse_only", "hybrid_50")]
    reference = base.make_world(seed, scenario, base.Params(n=n, dt=dt))
    production = float(reference["supply"].sum()*160.)
    for policy, factor in settings:
        p = base.Params(n=n, dt=dt, swap_cost=base.Params().swap_cost*factor)
        row = pulse.simulate(policy, scenario, seed, p, RATES[policy])
        row.update(cost_factor=factor, swap_cost=p.swap_cost,
                   gross_production=production, cost_fraction=row["resource_cost"]/production)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--step-check", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "step_" if args.step_check else ""
    if (args.output/(prefix+"runs.csv")).exists():
        parser.error("Refusing to overwrite existing results")
    cases = list(product((96, 960), ("repeated_damage",), range(13100, 13106), (.2, .1))) if args.step_check else list(
        product((96, 960), base.SCENARIOS, range(13000, 13030), (.2,)))
    hashes = {}
    for path in (Path(__file__), Path(base.__file__), Path(pulse.__file__)):
        data = path.read_bytes()
        (args.output/(prefix+path.name)).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    config = dict(params=asdict(base.Params()), cases=cases, cost_factors=COSTS, rates=RATES,
                  primary=dict(n=960, scenario="repeated_damage", cost_factor=20, metric="service"),
                  hashes=hashes, python=platform.python_version(), numpy=np.__version__)
    (args.output/(prefix+"config.json")).write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, group in enumerate(pool.map(run_group, cases, chunksize=1), 1):
            rows.extend(group)
            if i % 12 == 0 or i == len(cases):
                base.write_csv(args.output/(prefix+"runs.csv"), rows)
                print(f"Completed {i}/{len(cases)} worlds; {len(rows)} runs; {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
