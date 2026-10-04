"""Exploratory holdout: can a slower simple pulse match the hybrid tradeoff?"""
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment_support_pulse as model
import experiment_local_support_signal as base

# Rates chosen from discovery-set traffic only: approximately 244k and 332k.
# No holdout outcome is used to tune them. Actual traffic need not be identical.
CASES = (("hybrid_50", 4.), ("pulse_only", 2.3),
         ("hybrid_75", 4.), ("pulse_only", 3.15))


def run(seed):
    return [model.simulate(policy, "repeated_damage", seed, base.Params(n=960), rate)
            for policy, rate in CASES]


def main():
    out = model.OUT
    if (out/"traffic_runs.csv").exists():
        raise FileExistsError("Refusing to overwrite holdout")
    hashes = {}
    for path in (Path(__file__), Path(model.__file__), Path(base.__file__)):
        data = path.read_bytes()
        (out/("traffic_"+path.name)).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    config = dict(seeds=list(range(12250, 12280)), cases=CASES, scenario="repeated_damage", n=960, dt=.2,
                  common_upper_rate=4., hashes=hashes,
                  note="Exploratory held-out traffic-matching check. Slower pulse has stricter internal cap; actual messages are not forced equal.")
    (out/"traffic_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, group in enumerate(pool.map(run, config["seeds"]), 1):
            rows.extend(group)
            if i % 10 == 0:
                print(f"Holdout: {i}/30 worlds", flush=True)
    base.write_csv(out/"traffic_runs.csv", rows)
    assert len(rows) == 120
    assert all(r["max_message_budget_excess"] == 0 and r["max_mass_error"] < 1e-8 for r in rows)
    summary = []
    for policy, rate in CASES:
        select = [r for r in rows if r["policy"] == policy and r["rate"] == rate]
        summary.append(dict(policy=policy, rate=rate, seeds=len(select), **{
            metric: float(np.mean([r[metric] for r in select])) for metric in ("service", "p10", "messages", "swaps")}))
    base.write_csv(out/"traffic_summary.csv", summary)
    by = {(r["seed"], r["policy"], r["rate"]): r for r in rows}
    rng = np.random.default_rng(12300)
    effects = []
    for hybrid, rate in (("hybrid_50", 2.3), ("hybrid_75", 3.15)):
        for metric in ("service", "p10", "messages"):
            vals = np.array([by[s, hybrid, 4.][metric]-by[s, "pulse_only", rate][metric] for s in config["seeds"]])
            lo, hi = np.quantile(rng.choice(vals, (4000, len(vals))).mean(axis=1), [.025, .975])
            effects.append(dict(treatment=hybrid, pulse_rate=rate, metric=metric, delta=float(vals.mean()), ci_low=float(lo), ci_high=float(hi)))
    base.write_csv(out/"traffic_comparisons.csv", effects)
    print(json.dumps(dict(summary=summary, effects=effects), indent=2))


if __name__ == "__main__":
    main()
