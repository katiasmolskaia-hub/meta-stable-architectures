"""Audit complete paired cases and summarize the pulse-budget experiment."""
import csv
import hashlib
import json

import numpy as np

from experiment_support_pulse import OUT, SHARES
from experiment_local_support_signal import write_csv

METRICS = ("service", "p10", "messages", "requests", "swaps", "resource_cost",
           "pulse_messages", "signal_messages", "pulse_requests", "signal_requests",
           "active_need_end", "source_reach_end") + tuple(f"phase{i}_service" for i in range(4))


def report(prefix=""):
    config = json.loads((OUT/(prefix+"config.json")).read_text(encoding="utf-8"))
    with (OUT/(prefix+"runs.csv")).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {(n, scenario, seed, dt, rate, policy) for n, scenario, seed, dt, rate in config["cases"] for policy in SHARES}
    key = lambda r: (int(r["n"]), r["scenario"], int(r["seed"]), float(r["dt"]), float(r["rate"]), r["policy"])
    assert len(rows) == len(expected) and {key(r) for r in rows} == expected
    for filename, sha in config["hashes"].items():
        assert hashlib.sha256((OUT/(prefix+filename)).read_bytes()).hexdigest() == sha
    for r in rows:
        assert int(r["messages"]) == 24*int(r["requests"])+8*int(r["swaps"])
        assert int(r["messages"]) == int(r["pulse_messages"])+int(r["signal_messages"])
        assert float(r["messages"]) <= float(r["allowed_message_envelope"])
        assert float(r["max_mass_error"]) < 1e-8
        assert float(r["max_flow_ratio"]) <= 1.+1e-10
        assert float(r["max_message_budget_excess"]) == 0.
        assert int(r["active_edges"]) == 2*int(r["n"])
        assert int(r["swaps"]) <= int(r["requests"])
        assert abs(float(r["resource_cost"])-.024*int(r["swaps"])) < 1e-8
        assert float(r["minimum_request_gap"]) == -1. or float(r["minimum_request_gap"]) >= 8.-1e-9
    rng = np.random.default_rng(12200)
    summary, effects = [], []
    for n, scenario, dt, rate in sorted({(int(r["n"]), r["scenario"], float(r["dt"]), float(r["rate"])) for r in rows}):
        group = [r for r in rows if (int(r["n"]), r["scenario"], float(r["dt"]), float(r["rate"])) == (n, scenario, dt, rate)]
        by = {p: {int(r["seed"]): r for r in group if r["policy"] == p} for p in SHARES}
        seeds = sorted(by["fixed"])
        context = dict(n=n, scenario=scenario, dt=dt, rate=rate, seeds=len(seeds))
        for policy in SHARES:
            assert sorted(by[policy]) == seeds
            summary.append(dict(**context, policy=policy, **{m: float(np.mean([float(by[policy][s][m]) for s in seeds])) for m in METRICS}))
        for treatment in ("hybrid_25", "hybrid_50", "hybrid_75", "signal_only"):
            for baseline in ("pulse_only", "signal_only"):
                if treatment == baseline:
                    continue
                for metric in ("service", "p10", "messages"):
                    values = np.array([float(by[treatment][s][metric])-float(by[baseline][s][metric]) for s in seeds])
                    lo, hi = np.quantile(rng.choice(values, (4000, len(seeds)), replace=True).mean(axis=1), [.025, .975])
                    effects.append(dict(**context, treatment=treatment, baseline=baseline, metric=metric,
                                        delta=float(values.mean()), ci_low=float(lo), ci_high=float(hi)))
    write_csv(OUT/(prefix+"summary.csv"), summary)
    write_csv(OUT/(prefix+"paired_comparisons.csv"), effects)
    audit = dict(runs=len(rows), max_mass_error=max(float(r["max_mass_error"]) for r in rows),
                 max_message_budget_excess=0., minimum_final_source_reach=min(float(r["source_reach_end"]) for r in rows))
    (OUT/(prefix+"audit.json")).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Пульс и запрос: полная таблица", "", "Доли спроса указаны в процентах; сообщения — за весь опыт. Парные интервалы доступны в paired_comparisons.csv.", "",
             "| N | Сценарий | dt | Бюджет/время | Вариант | Спрос, % | P10, % | Сообщения | Из них пульс |",
             "|---:|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['n']} | {r['scenario']} | {r['dt']} | {r['rate']} | {r['policy']} | {100*r['service']:.2f} | {100*r['p10']:.2f} | {r['messages']:.0f} | {r['pulse_messages']:.0f} |")
    (OUT/(prefix+"tables.md")).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(prefix=prefix, **audit)))
    for r in summary:
        if r["scenario"] == "repeated_damage" and (prefix or r["n"] == 960):
            print(json.dumps({k:r[k] for k in ("n", "dt", "rate", "policy", "service", "p10", "messages")}))


if __name__ == "__main__":
    report()
    report("step_")
