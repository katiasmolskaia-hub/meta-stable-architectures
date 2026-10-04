"""Audit all cost-study cases and estimate paired effects without selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import experiment_local_support_signal as base
from experiment_support_pulse_cost import OUT, COSTS, RATES

METRICS = ("service", "p10", "messages", "requests", "swaps", "resource_cost", "cost_fraction",
           "success_fraction", "active_need_end", "source_reach_end", "pulse_messages", "signal_messages")
EFFECT_METRICS = ("service", "p10", "messages", "swaps", "resource_cost", "cost_fraction", "success_fraction")


def audit(out, prefix):
    config = json.loads((out/(prefix+"config.json")).read_text(encoding="utf-8"))
    with (out/(prefix+"runs.csv")).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    key = lambda r: (int(r["n"]), r["scenario"], int(r["seed"]), float(r["dt"]), r["policy"], int(r["cost_factor"]))
    settings = [("fixed", 0)] + [(policy, cost) for cost in config["cost_factors"] for policy in ("pulse_only", "hybrid_50")]
    expected = {(n, scenario, seed, dt, policy, cost) for n, scenario, seed, dt in config["cases"] for policy, cost in settings}
    assert len(rows) == len(expected) and {key(r) for r in rows} == expected
    for name, sha in config["hashes"].items():
        assert hashlib.sha256((out/(prefix+name)).read_bytes()).hexdigest() == sha
    for r in rows:
        assert int(r["messages"]) == 24*int(r["requests"])+8*int(r["swaps"])
        assert int(r["messages"]) == int(r["pulse_messages"])+int(r["signal_messages"])
        assert int(r["requests"]) == int(r["pulse_requests"])+int(r["signal_requests"])
        assert int(r["swaps"]) <= int(r["requests"])
        assert float(r["messages"]) <= float(r["allowed_message_envelope"])
        assert float(r["max_message_budget_excess"]) == 0.
        assert float(r["max_mass_error"]) < 1e-8 and float(r["max_flow_ratio"]) <= 1.+1e-10
        assert float(r["minimum_request_gap"]) == -1. or float(r["minimum_request_gap"]) >= 8.-1e-9
        assert int(r["active_edges"]) == 2*int(r["n"])
        assert float(r["rate"]) == config["rates"][r["policy"]]
        assert abs(float(r["swap_cost"])-int(r["cost_factor"])*config["params"]["swap_cost"]) < 1e-12
        assert abs(float(r["resource_cost"])-int(r["swaps"])*float(r["swap_cost"])) < 1e-8
        assert abs(float(r["cost_fraction"])-float(r["resource_cost"])/float(r["gross_production"])) < 1e-12
        assert 0 <= float(r["service"]) <= 1.+1e-10 and 0 <= float(r["p10"]) <= 1.+1e-10
        if r["policy"] == "fixed":
            assert int(r["cost_factor"]) == 0 and int(r["messages"]) == 0
        r["success_fraction"] = int(r["swaps"])/int(r["requests"]) if int(r["requests"]) else 0.
    return rows, dict(runs=len(rows), complete_grid=True, hashes_verified=True,
                     max_mass_error=max(float(r["max_mass_error"]) for r in rows),
                     max_message_budget_excess=0.,
                     minimum_final_source_reach=min(float(r["source_reach_end"]) for r in rows))


def interval(values, rng):
    values = np.asarray(values)
    lo, hi = np.quantile(rng.choice(values, (5000, len(values))).mean(axis=1), [.025, .975])
    return dict(delta=float(values.mean()), ci_low=float(lo), ci_high=float(hi))


def summarize(out, prefix=""):
    rows, checks = audit(out, prefix)
    rng = np.random.default_rng(13200)
    summary, effects, interactions = [], [], []
    groups = sorted({(int(r["n"]), r["scenario"], float(r["dt"])) for r in rows})
    for n, scenario, dt in groups:
        group = [r for r in rows if (int(r["n"]), r["scenario"], float(r["dt"])) == (n, scenario, dt)]
        by = {(r["policy"], int(r["cost_factor"]), int(r["seed"])): r for r in group}
        seeds = sorted({int(r["seed"]) for r in group})
        context = dict(n=n, scenario=scenario, dt=dt, seeds=len(seeds))
        for policy, cost in [("fixed", 0)] + [(p, c) for c in COSTS for p in ("pulse_only", "hybrid_50")]:
            select = [by[policy, cost, s] for s in seeds]
            summary.append(dict(**context, policy=policy, cost_factor=cost,
                                **{m: float(np.mean([float(r[m]) for r in select])) for m in METRICS}))
        differences = {}
        for cost in COSTS:
            for metric in EFFECT_METRICS:
                vals = np.array([float(by["hybrid_50", cost, s][metric])-float(by["pulse_only", cost, s][metric]) for s in seeds])
                differences[cost, metric] = vals
                effects.append(dict(**context, cost_factor=cost, metric=metric, **interval(vals, rng)))
        for cost in (5, 20):
            for metric in ("service", "p10", "messages", "resource_cost"):
                vals = differences[cost, metric]-differences[1, metric]
                interactions.append(dict(**context, cost_factor=cost, metric=metric, **interval(vals, rng)))
    for name, data in (("summary.csv", summary), ("paired_comparisons.csv", effects), ("cost_interactions.csv", interactions)):
        base.write_csv(out/(prefix+name), data)
    (out/(prefix+"audit.json")).write_text(json.dumps(checks, indent=2), encoding="utf-8")
    lines = ["# Цена перестройки: полная таблица", "", "Снабжение и P10 за t=40–160. Сообщения и затраты за t=0–160. Доля затрат — от всего произведённого ресурса, не от доступного после потерь.", "",
             "| N | Сценарий | dt | Цена | Вариант | Спрос, % | P10, % | Сообщений | Обменов | Затраты / производство, % |",
             "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['n']} | {r['scenario']} | {r['dt']} | {r['cost_factor']} | {r['policy']} | {100*r['service']:.2f} | {100*r['p10']:.2f} | {r['messages']:.0f} | {r['swaps']:.1f} | {100*r['cost_fraction']:.2f} |")
    (out/(prefix+"tables.md")).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(prefix=prefix, **checks)))
    for r in summary:
        if r["n"] == 960 and r["scenario"] == "repeated_damage":
            print(json.dumps({k:r[k] for k in ("dt", "cost_factor", "policy", "service", "p10", "messages", "swaps", "cost_fraction")}))
    for r in effects:
        if r["n"] == 960 and r["scenario"] == "repeated_damage" and r["metric"] in ("service", "p10"):
            print(json.dumps(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    summarize(args.output)
    summarize(args.output, "step_")
