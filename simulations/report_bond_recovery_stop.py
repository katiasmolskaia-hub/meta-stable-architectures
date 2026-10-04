"""Audit paired post-recovery continuations, retaining non-recovered networks."""
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from experiment_bond_recovery_stop import OUT, SIZES, COSTS, SCENARIOS, POLICIES, write_csv

METRICS = ("quiet_service", "quiet_p10", "quiet_low_time_fraction", "quiet_relapse", "quiet_cost",
           "quiet_stock_end", "quiet_fork_edge_share", "second_service", "second_p10",
           "second_low_time_fraction", "second_unrecovered", "second_cost")


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def key(r):
    return tuple(r[k] for k in ("n", "cost_multiplier", "scenario", "policy", "seed", "dt", "noise"))


def audit(prefix=""):
    episodes, branches = read(prefix + "episodes.csv"), read(prefix + "branches.csv")
    config = json.loads((OUT / (prefix + "config.json")).read_text(encoding="utf-8"))
    expected = {tuple(map(str, c[:2])) + (c[2], c[3], str(c[4]), str(c[5]), str(c[6])) for c in config["cases_spec"]}
    assert len(episodes) == len(expected) == config["cases"]
    assert {key(r) for r in episodes} == expected
    recovered = {key(r): r for r in episodes if r["status"] == "recovered"}
    assert len(branches) == 2 * len(recovered)
    assert {(key(r), r["branch"]) for r in branches} == {(k, b) for k in recovered for b in ("continue", "pause")}
    for r in episodes + branches:
        assert float(r["max_mass_error"]) < 1e-8
        assert float(r["max_budget_ratio"]) <= 1 + 1e-10
    for r in branches:
        batch, cost = int(r["n"]) // 24, int(r["cost_multiplier"]) * .024
        quiet_swaps = 15 * batch if r["branch"] == "continue" else 0
        assert int(r["quiet_swaps"]) == quiet_swaps
        assert int(r["second_swaps"]) == 30 * batch
        assert abs(float(r["quiet_cost"]) - quiet_swaps * cost) < 1e-9
        assert abs(float(r["second_cost"]) - 30 * batch * cost) < 1e-9
        assert abs(float(r["second_shock_time"]) - float(r["recovery_time"]) - 30) < 1e-9
        if r["branch"] == "pause":
            assert float(r["quiet_fork_edge_share"]) == 1
    for name, h in config["sha256"].items():
        assert hashlib.sha256((OUT / (prefix + name)).read_bytes()).hexdigest() == h
        if name != "experiment_bond_recovery_stop.py":
            assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == h
    result = dict(initial_networks=len(episodes), recovered=len(recovered), paired_branches=len(branches),
                  not_recovered=sum(r["status"] == "not_recovered" for r in episodes),
                  unaffected=sum(r["status"] == "unaffected" for r in episodes),
                  max_mass_error=max(float(r["max_mass_error"]) for r in episodes + branches),
                  max_budget_ratio=max(float(r["max_budget_ratio"]) for r in episodes + branches))
    return episodes, branches, result


def summarize(episodes, branches, prefix=""):
    groups = sorted({tuple(r[k] for k in ("n", "cost_multiplier", "scenario", "policy", "dt")) for r in episodes})
    rng = np.random.default_rng(10043)
    summaries, comparisons = [], []
    for group in groups:
        matches = lambda r: tuple(r[k] for k in ("n", "cost_multiplier", "scenario", "policy", "dt")) == group
        ee = [r for r in episodes if matches(r)]
        by_branch = {b: {int(r["seed"]): r for r in branches if matches(r) and r["branch"] == b}
                     for b in ("continue", "pause")}
        ids = sorted(by_branch["continue"])
        assert ids == sorted(by_branch["pause"])
        meta = dict(zip(("n", "cost_multiplier", "scenario", "policy", "dt"), group))
        totals = dict(total=len(ee), recovered=len(ids),
                      not_recovered=sum(r["status"] == "not_recovered" for r in ee),
                      unaffected=sum(r["status"] == "unaffected" for r in ee))
        for b in by_branch:
            r = dict(**meta, **totals, branch=b)
            for metric in METRICS:
                r[metric + "_mean"] = float(np.mean([float(by_branch[b][i][metric]) for i in ids])) if ids else None
            summaries.append(r)
        for metric in METRICS:
            delta = np.array([float(by_branch["continue"][i][metric]) - float(by_branch["pause"][i][metric]) for i in ids])
            mean = float(delta.mean()) if len(delta) else None
            lo = hi = None
            if len(delta) >= 2:
                boot = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
                lo, hi = map(float, np.quantile(boot, [.025, .975]))
            comparisons.append(dict(**meta, **totals, metric=metric, mean=mean, ci_low=lo, ci_high=hi))
    write_csv(OUT / (prefix + "summary.csv"), summaries)
    write_csv(OUT / (prefix + "paired_comparisons.csv"), comparisons)
    return summaries, comparisons


def tables(summaries, comparisons):
    labels = dict(persistent_damage="Длительное", temporary_damage="Временное",
                  recent_contact="Последний контакт", experience_swap="Накопленный опыт")
    lines = ["# Перестройка после восстановления: полные таблицы", "",
        "Эффекты относятся только к сетям, которые сначала потеряли снабжение и затем достигли критерия 95% для каждого потребителя на протяжении 10 единиц. Остальные исходы перечислены отдельно.", "",
        "| Узлов | Цена | Первое повреждение | Выбор соседей | Восстановились / всего | Не восстановились | Без падения |",
        "|---:|---:|---|---|---:|---:|---:|"]
    for r in summaries:
        if r["branch"] == "continue":
            lines.append(f"| {r['n']} | {r['cost_multiplier']}× | {labels[r['scenario']]} | {labels[r['policy']]} | {r['recovered']}/{r['total']} | {r['not_recovered']} | {r['unaffected']} |")
    for phase, title in (("quiet", "Период между повреждениями"), ("second", "После второго повреждения")):
        lines += ["", "## " + title, "",
                  "| Узлов | Цена | Первое повреждение | Выбор соседей | Пар | Продолжать: спрос | Пауза: спрос | Продолжать − пауза, п.п. [95% интервал] |",
                  "|---:|---:|---|---|---:|---:|---:|---:|"]
        for r in summaries:
            if r["branch"] != "continue":
                continue
            same = lambda x: all(x[k] == r[k] for k in ("n", "cost_multiplier", "scenario", "policy", "dt"))
            b = next(x for x in summaries if same(x) and x["branch"] == "pause")
            c = next(x for x in comparisons if same(x) and x["metric"] == phase + "_service")
            def fmt(v):
                return "—" if v is None else f"{100*v:.2f}"
            lines.append(f"| {r['n']} | {r['cost_multiplier']}× | {labels[r['scenario']]} | {labels[r['policy']]} | {r['recovered']} | {fmt(r[phase+'_service_mean'])}% | {fmt(b[phase+'_service_mean'])}% | {fmt(c['mean'])} [{fmt(c['ci_low'])}; {fmt(c['ci_high'])}] |")
    lines += ["", "Интервалы исследовательские, без поправки на множество сравнений. Малые группы не позволяют делать уверенные выводы. Эффекты разных правил выбора условны на разные подмножества восстановившихся сетей и не являются прямым сравнением правил.", ""]
    (OUT / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def step_agreement(episodes, branches):
    # Compare only the same seeds eligible at BOTH resolutions, and report
    # selection disagreements rather than silently dropping them.
    results = []
    for policy in POLICIES:
        dts = sorted({float(r["dt"]) for r in episodes})
        selected = {dt: {int(r["seed"]) for r in episodes if r["policy"] == policy and float(r["dt"]) == dt and r["status"] == "recovered"} for dt in dts}
        common = sorted(set.intersection(*selected.values()))
        index = {(int(r["seed"]), float(r["dt"]), r["branch"]): r for r in branches if r["policy"] == policy}
        record = dict(policy=policy, recovered_by_dt={str(dt): len(s) for dt, s in selected.items()},
                      shared=len(common), differing_selection=sorted(set.union(*selected.values()) - set(common)))
        for phase in ("quiet", "second"):
            for dt in dts:
                effects = [float(index[s, dt, "continue"][phase+"_service"]) - float(index[s, dt, "pause"][phase+"_service"]) for s in common]
                record[phase + "_effect_pp_dt_" + str(dt)] = 100 * float(np.mean(effects)) if effects else None
        results.append(record)
    return results


def main():
    episodes, branches, checks = audit()
    summary, comparisons = summarize(episodes, branches)
    tables(summary, comparisons)
    se, sb, sc = audit("step_")
    summarize(se, sb, "step_")
    checks["step_check"] = sc
    checks["step_agreement"] = step_agreement(se, sb)
    if (OUT / "recovery_step_episodes.csv").exists():
        re, rb, rc = audit("recovery_step_")
        summarize(re, rb, "recovery_step_")
        checks["recovery_step_check"] = rc
        checks["recovery_step_agreement"] = step_agreement(re, rb)
    (OUT / "audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    for r in comparisons:
        if r["metric"] in ("quiet_service", "second_service"):
            print(json.dumps(r))


if __name__ == "__main__":
    main()
