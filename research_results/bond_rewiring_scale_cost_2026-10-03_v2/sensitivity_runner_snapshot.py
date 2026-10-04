"""Preregistered scale/cost sweep; original experiment remains unchanged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np

from bond_rewiring_scaled_model import Params, POLICIES, simulate
from experiment_bond_rewiring import write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bond_rewiring_scale_cost_2026-10-03_v2"
SIZES = (24, 48, 96)
COSTS = (1, 5, 20)
SCENARIOS = ("uniform", "persistent_damage", "rapid_change", "separated_resources")
METRICS = ("service_fraction", "consumer_p10", "underserved_share", "source_reach_mean",
           "rewiring_cost", "cost_supply_fraction", "swaps", "max_mass_error", "max_budget_ratio")


def parameters(n, cost, dt=.2, noise=.03):
    return replace(Params(), n=n, sources=n // 4, swaps_per_event=n // 24,
                   swap_cost=.024 * cost, dt=dt, observation_noise=noise)


def run_case(case):
    n, cost, scenario, seed, policy, dt, noise = case
    r, _, _, _ = simulate(policy, scenario, seed, parameters(n, cost, dt, noise))
    return dict(n=n, cost_multiplier=cost, dt=dt, noise=noise,
                cost_supply_fraction=r["rewiring_cost"] / r["gross_supply"], **r)


def summarize(rows):
    summary, comparisons = [], []
    rng = np.random.default_rng(7803)
    for n, cost, scenario in product(SIZES, COSTS, SCENARIOS):
        groups = {}
        for policy in POLICIES:
            selected = [r for r in rows if r["n"] == n and r["scenario"] == scenario
                        and r["policy"] == policy and r["cost_multiplier"] == (0 if policy == "fixed" else cost)]
            groups[policy] = {r["seed"]: r for r in selected}
            r = dict(n=n, cost_multiplier=cost, scenario=scenario, policy=policy, seeds=len(selected))
            for key in METRICS:
                vals = np.array([x[key] for x in selected])
                r[key + "_mean"] = float(vals.mean())
                r[key + "_sd"] = float(vals.std(ddof=1))
            summary.append(r)
        for a, b in (("recent_contact", "random_swap"), ("experience_swap", "random_swap"),
                     ("recent_contact", "fixed"), ("experience_swap", "fixed"),
                     ("experience_swap", "recent_contact")):
            seeds = sorted(groups[a])
            assert seeds == sorted(groups[b])
            for metric in ("service_fraction", "consumer_p10"):
                delta = np.array([groups[a][s][metric] - groups[b][s][metric] for s in seeds])
                boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                comparisons.append(dict(n=n, cost_multiplier=cost, scenario=scenario,
                                        treatment=a, baseline=b, metric=metric,
                                        mean=float(delta.mean()), ci_low=float(lo), ci_high=float(hi)))
    return summary, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=6100)
    parser.add_argument("--step-check", action="store_true")
    parser.add_argument("--sensitivity-check", action="store_true")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("At least two seeds required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.sensitivity_check:
        corners = ((24, 20, "persistent_damage"), (24, 1, "separated_resources"),
                   (96, 20, "persistent_damage"))
        cases = [(n, c, s, seed, p, dt, 0.) for n, c, s in corners
                 for seed, p, dt in product(range(7000, 7020), POLICIES, (.2, .1, .05))]
        target = args.output / "step_sensitivity.csv"
    elif args.step_check:
        cases = [(n, c, s, 6900, p, dt, 0.) for n, c, s, p, dt in
                 product((24, 96), (1, 20), ("persistent_damage", "separated_resources"), POLICIES, (.2, .1))]
        target = args.output / "step_refinement.csv"
    else:
        cases = []
        for n, scenario, seed in product(SIZES, SCENARIOS, range(args.seed_start, args.seed_start + args.seeds)):
            cases.append((n, 0, scenario, seed, "fixed", .2, .03))
            cases.extend((n, c, scenario, seed, p, .2, .03) for c, p in product(COSTS, POLICIES[1:]))
        target = args.output / "runs.csv"
    if target.exists():
        parser.error(f"Refusing to overwrite {target}; choose another --output")
    if not (args.step_check or args.sensitivity_check):
        hashes = {}
        for name in ("experiment_bond_rewiring_scale_cost.py", "bond_rewiring_scaled_model.py", "experiment_bond_rewiring.py"):
            data = (Path(__file__).parent / name).read_bytes()
            (args.output / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        (args.output / "config.json").write_text(json.dumps(dict(
            base_params=asdict(Params()), sizes=SIZES, costs=COSTS, scenarios=SCENARIOS, policies=POLICIES,
            seeds=args.seeds, seed_start=args.seed_start, cases=len(cases), sha256=hashes,
            python=platform.python_version(), numpy=np.__version__,
            normalization="n/24 sequential swaps per two time units; fixed control reused across costs"
        ), indent=2), encoding="utf-8")
    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor, target.open("w", encoding="utf-8", newline="") as f:
        writer = None
        for i, row in enumerate(executor.map(run_case, cases, chunksize=5), 1):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            rows.append(row)
            if i % 100 == 0 or i == len(cases):
                f.flush()
                print(f"Completed {i}/{len(cases)}; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    if not (args.step_check or args.sensitivity_check):
        summary, comparisons = summarize(rows)
        write_csv(args.output / "summary.csv", summary)
        write_csv(args.output / "paired_comparisons.csv", comparisons)
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
