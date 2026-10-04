"""Validate the sparse experiment and report paired effects and control traffic."""
import csv
import hashlib
import json
from itertools import product

import numpy as np

from experiment_local_support_signal import OUT, POLICIES, write_csv

METRICS = ("service", "p10", "messages", "requests", "swaps", "resource_cost", "active_need_end",
           "request_rate_90_100", "request_rate_110_140", "source_reach_end") + tuple(
    f"phase{i}_{metric}" for i in range(4) for metric in ("service", "requests", "need_onsets", "need_offsets"))


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit(prefix):
    rows = read(prefix + "runs.csv")
    config = json.loads((OUT / (prefix + "config.json")).read_text(encoding="utf-8"))
    expected = {(str(n), s, str(seed), str(dt), p) for n, s, seed, dt in config["cases"] for p in POLICIES}
    key = lambda r: (r["n"], r["scenario"], r["seed"], r["dt"], r["policy"])
    assert len(rows) == len(expected) and {key(r) for r in rows} == expected
    assert hashlib.sha256((OUT / (prefix + "experiment_snapshot.py")).read_bytes()).hexdigest() == config["sha256"]
    lookup = {key(r): r for r in rows}
    for r in rows:
        n = int(r["n"])
        assert int(r["allowed_edges"]) == 6*n and int(r["active_edges"]) == 2*n
        assert int(r["messages"]) == 24*int(r["requests"]) + 8*int(r["swaps"])
        assert int(r["requests"]) <= 20 * (n * 3 // 4)
        assert abs(float(r["resource_cost"]) - int(r["swaps"])*.024) < 1e-8
        assert float(r["max_mass_error"]) < 1e-8 and float(r["max_budget_ratio"]) <= 1 + 1e-10
        if r["policy"] == "local_signal":
            other = lookup[(r["n"], r["scenario"], r["seed"], r["dt"], "shuffled_signal")]
            assert r["requests"] == other["requests"]
            for phase in range(4):
                assert r[f"phase{phase}_requests"] == other[f"phase{phase}_requests"]
    return rows, dict(runs=len(rows), max_mass_error=max(float(r["max_mass_error"]) for r in rows),
                      max_budget_ratio=max(float(r["max_budget_ratio"]) for r in rows),
                      minimum_final_source_reach=min(float(r["source_reach_end"]) for r in rows))


def summarize(rows, prefix):
    rng = np.random.default_rng(11004)
    summaries, comparisons = [], []
    groups = sorted({(int(r["n"]), r["scenario"], float(r["dt"])) for r in rows})
    for n, scenario, dt in groups:
        selected = [r for r in rows if int(r["n"]) == n and r["scenario"] == scenario and float(r["dt"]) == dt]
        by = {p: {int(r["seed"]): r for r in selected if r["policy"] == p} for p in POLICIES}
        ids = sorted(by["local_signal"])
        for p in POLICIES:
            assert sorted(by[p]) == ids
            r = dict(n=n, scenario=scenario, dt=dt, policy=p, seeds=len(ids))
            for metric in METRICS:
                r[metric + "_mean"] = float(np.mean([float(by[p][seed][metric]) for seed in ids]))
            summaries.append(r)
        for baseline in ("fixed", "periodic", "shuffled_signal"):
            for metric in ("service", "p10", "messages", "phase3_service"):
                delta = np.array([float(by["local_signal"][s][metric]) - float(by[baseline][s][metric]) for s in ids])
                lo, hi = np.quantile(rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1), [.025, .975])
                comparisons.append(dict(n=n, scenario=scenario, dt=dt, treatment="local_signal", baseline=baseline,
                                        metric=metric, seeds=len(ids), mean=float(delta.mean()), ci_low=float(lo), ci_high=float(hi)))
    write_csv(OUT / (prefix + "summary.csv"), summaries)
    write_csv(OUT / (prefix + "paired_comparisons.csv"), comparisons)
    return summaries, comparisons


def main():
    rows, checks = audit("")
    summary, comparisons = summarize(rows, "")
    step, step_checks = audit("step_")
    ss, sc = summarize(step, "step_")
    checks["step_check"] = step_checks
    checks["step_means"] = [{k: r[k] for k in ("policy", "dt", "service_mean", "messages_mean", "requests_mean")} for r in ss]
    (OUT / "audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    labels = dict(uniform="Хорошие связи", persistent_damage="Длительное повреждение",
                  repeated_damage="Два повреждения", scarcity="Нехватка производства")
    lines = ["# Локальный сигнал: полные результаты", "",
             "По 12 парных seed. Снабжение — доля спроса за t=40–160; сообщения — логические управляющие сообщения за весь опыт, без энергетической цены и обычного трафика передачи ресурса.", "",
             "| Участников | Сценарий | Без изменений | По расписанию | Местный сигнал | Перемешанный сигнал |",
             "|---:|---|---:|---:|---:|---:|"]
    index = {(r["n"], r["scenario"], r["policy"]):r for r in summary}
    for n, scenario in sorted({(r["n"], r["scenario"]) for r in summary}):
        vals = [100*index[n, scenario, p]["service_mean"] for p in POLICIES]
        lines.append(f"| {n} | {labels[scenario]} | " + " | ".join(f"{v:.2f}%" for v in vals) + " |")
    lines += ["", "## Сообщения", "", "| Участников | Сценарий | По расписанию | Местный сигнал | Перемешанный сигнал | Сокращение относительно расписания |", "|---:|---|---:|---:|---:|---:|"]
    for n, scenario in sorted({(r["n"], r["scenario"]) for r in summary}):
        vals = [index[n, scenario, p]["messages_mean"] for p in POLICIES[1:]]
        lines.append(f"| {n} | {labels[scenario]} | " + " | ".join(f"{v:.0f}" for v in vals) + f" | {100*(1-vals[1]/vals[0]):.1f}% |")
    lines += ["", "## Парные эффекты местного сигнала", "",
              "95% bootstrap-интервалы без поправки на множество сравнений; 12 seed недостаточно для универсального вывода.", "",
              "| Узлов | Сценарий | Базовый вариант | Снабжение: разность, п.п. [95% интервал] |", "|---:|---|---|---:|"]
    for r in comparisons:
        if r["metric"] == "service":
            lines.append(f"| {r['n']} | {labels[r['scenario']]} | {r['baseline']} | {100*r['mean']:+.2f} [{100*r['ci_low']:+.2f}; {100*r['ci_high']:+.2f}] |")
    (OUT / "tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    for n, scenario in sorted({(r["n"], r["scenario"]) for r in summary}):
        print(json.dumps(dict(n=n, scenario=scenario,
            service={p:100*index[n, scenario,p]["service_mean"] for p in POLICIES},
            messages={p:index[n,scenario,p]["messages_mean"] for p in POLICIES})))


if __name__ == "__main__":
    main()
