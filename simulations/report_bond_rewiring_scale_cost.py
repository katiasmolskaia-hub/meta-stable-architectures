"""Validate and summarize the scale/cost sweep without changing its data."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_bond_rewiring_scale_cost import OUT, SIZES, COSTS, SCENARIOS, POLICIES


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sensitivity_report():
    rows = read("step_sensitivity.csv")
    assert len(rows) == 720
    keys = {(r["n"], r["cost_multiplier"], r["scenario"], r["seed"], r["policy"], r["dt"]) for r in rows}
    corners = ((24, 20, "persistent_damage"), (24, 1, "separated_resources"), (96, 20, "persistent_damage"))
    expected = {(str(n), str(cost), scenario, str(seed), policy, str(dt))
                for n, cost, scenario in corners for seed in range(7000, 7020)
                for policy in POLICIES for dt in (.2, .1, .05)}
    assert keys == expected
    for r in rows:
        swaps = 0 if r["policy"] == "fixed" else 59 * (int(r["n"]) // 24)
        assert int(r["swaps"]) == swaps
        assert abs(float(r["rewiring_cost"]) - swaps * .024 * int(r["cost_multiplier"])) < 1e-9
        assert float(r["max_mass_error"]) < 1e-8 and float(r["max_budget_ratio"]) <= 1 + 1e-10
    source = (Path(__file__).parent / "experiment_bond_rewiring_scale_cost.py").read_bytes()
    (OUT / "sensitivity_runner_snapshot.py").write_bytes(source)
    (OUT / "sensitivity_config.json").write_text(json.dumps(dict(corners=corners,
        seed_start=7000, seeds=20, noise=0, steps=[.2, .1, .05], runs=720,
        sha256=hashlib.sha256(source).hexdigest(),
        max_mass_error=max(float(r["max_mass_error"]) for r in rows),
        max_budget_ratio=max(float(r["max_budget_ratio"]) for r in rows)
    ), indent=2), encoding="utf-8")
    rng = np.random.default_rng(7921)
    results = []
    lines = ["# Чувствительность к шагу: дополнительная серия без шума", "",
             "Три случая выбраны после расхождений в одиночной проверке; по 20 новых seed 7000–7019. Это диагностика, не повтор всех условий.", "",
             "| Узлов | Цена | Сценарий | Стратегия | dt=0.2 | dt=0.1 | dt=0.05 | Изменение 0.05−0.2, п.п. [95% интервал] |",
             "|---:|---:|---|---|---:|---:|---:|---:|"]
    for n, cost, scenario in ((24, 20, "persistent_damage"), (24, 1, "separated_resources"), (96, 20, "persistent_damage")):
        for policy in POLICIES:
            subset = [r for r in rows if int(r["n"]) == n and int(r["cost_multiplier"]) == cost and r["scenario"] == scenario and r["policy"] == policy]
            arrays = {dt: np.array([float(r["service_fraction"]) for r in sorted(subset, key=lambda r: int(r["seed"])) if float(r["dt"]) == dt]) for dt in (.2, .1, .05)}
            assert all(len(a) == 20 for a in arrays.values())
            delta = 100 * (arrays[.05] - arrays[.2])
            boot = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            result = dict(n=n, cost=cost, scenario=scenario, policy=policy,
                          service_dt_02=float(arrays[.2].mean()), service_dt_01=float(arrays[.1].mean()),
                          service_dt_005=float(arrays[.05].mean()), mean_delta_pp=float(delta.mean()),
                          ci_low_pp=float(lo), ci_high_pp=float(hi), max_abs_individual_delta_pp=float(abs(delta).max()))
            results.append(result)
            values = " | ".join(f"{100*arrays[dt].mean():.2f}%" for dt in (.2, .1, .05))
            lines.append(f"| {n} | {cost} | {scenario} | {policy} | {values} | {delta.mean():+.2f} [{lo:+.2f}; {hi:+.2f}] |")
    (OUT / "sensitivity_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


def main():
    rows, summary, comparisons, refinement = [read(n) for n in
        ("runs.csv", "summary.csv", "paired_comparisons.csv", "step_refinement.csv")]
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    key = lambda r: (int(r["n"]), int(r["cost_multiplier"]), r["scenario"], int(r["seed"]), r["policy"])
    actual = {key(r) for r in rows}
    expected = set()
    for n in SIZES:
        for scenario in SCENARIOS:
            for seed in range(config["seed_start"], config["seed_start"] + config["seeds"]):
                expected.add((n, 0, scenario, seed, "fixed"))
                expected.update((n, c, scenario, seed, p) for c in COSTS for p in POLICIES[1:])
    assert actual == expected and len(rows) == len(expected) == config["cases"]
    assert len(summary) == 144 and len(comparisons) == 360 and len(refinement) == 64
    for name, digest in config["sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
        if name != "experiment_bond_rewiring_scale_cost.py":
            assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest
    for r in rows + refinement:
        swaps = 0 if r["policy"] == "fixed" else 59 * (int(r["n"]) // 24)
        assert int(r["swaps"]) == swaps
        assert abs(float(r["rewiring_cost"]) - swaps * .024 * int(r["cost_multiplier"])) < 1e-9
        assert float(r["max_mass_error"]) < 1e-8 and float(r["max_budget_ratio"]) <= 1 + 1e-10
    fine = {(*key(r), float(r["dt"])): r for r in refinement}
    deviations = []
    for r in refinement:
        if float(r["dt"]) == .2:
            s = fine[(*key(r), .1)]
            deviations.append(dict(n=int(r["n"]), cost=int(r["cost_multiplier"]),
                                   scenario=r["scenario"], policy=r["policy"],
                                   delta_service_pp=100 * (float(s["service_fraction"]) - float(r["service_fraction"])),
                                   delta_p10_pp=100 * (float(s["consumer_p10"]) - float(r["consumer_p10"]))))
    previous = OUT.parent / "bond_rewiring_scale_cost_2026-10-03" / "runs.csv"
    with previous.open(encoding="utf-8", newline="") as f:
        original_rows = list(csv.DictReader(f))
    assert len(original_rows) == len(rows)
    original = {key(r): r for r in original_rows}
    assert all(original[key(r)] == r for r in rows), "Payment correction changed main output"
    audit = dict(unique_runs=len(rows), summary_rows=len(summary), comparison_rows=len(comparisons),
                 refinement_runs=len(refinement), original_main_output_identical=True,
                 max_mass_error=max(float(r["max_mass_error"]) for r in rows + refinement),
                 max_budget_ratio=max(float(r["max_budget_ratio"]) for r in rows + refinement),
                 min_source_reach=min(float(r["source_reach_mean"]) for r in rows),
                 max_step_service_pp=max(abs(r["delta_service_pp"]) for r in deviations),
                 max_step_p10_pp=max(abs(r["delta_p10_pp"]) for r in deviations),
                 step_deviations=deviations)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    indexed = {(int(r["n"]), int(r["cost_multiplier"]), r["scenario"], r["policy"]): r for r in summary}
    labels = dict(uniform="Все связи хорошие", persistent_damage="Длительное повреждение",
                  rapid_change="Быстрая смена условий", separated_resources="Источники в одной половине")
    lines = ["# Смена соседей: полные таблицы размера и цены", "",
             "Проценты обслуженного спроса после t=40; по 20 seed на сравнение. Неизменная сеть повторена в таблице для удобства, но вычислена один раз на размер/сценарий/seed.", ""]
    for scenario in SCENARIOS:
        lines += ["## " + labels[scenario], "", "| Узлов | Цена | Без перестройки | Случайно | Последний контакт | Накопленный опыт |", "|---:|---:|---:|---:|---:|---:|"]
        for n in SIZES:
            for cost in COSTS:
                values = [100 * float(indexed[n, cost, scenario, p]["service_fraction_mean"]) for p in POLICIES]
                lines.append(f"| {n} | {cost}× | " + " | ".join(f"{v:.2f}%" for v in values) + " |")
        lines.append("")
    lines += ["## Десятый процентиль обслуживания потребителей", "",
              "Это нижняя часть распределения, не среднее худших 10% узлов.", "",
              "| Сценарий | Узлов | Цена | Без перестройки | Случайно | Последний контакт | Опыт |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in SCENARIOS:
        for n in SIZES:
            for cost in COSTS:
                vals = [100 * float(indexed[n, cost, scenario, p]["consumer_p10_mean"]) for p in POLICIES]
                lines.append(f"| {labels[scenario]} | {n} | {cost}× | " + " | ".join(f"{v:.2f}%" for v in vals) + " |")
    lines += ["", "## Парные разности обслуживания", "",
              "95% bootstrap-интервалы по seed, без поправки на множество сравнений. Единица — процентный пункт.", "",
              "| Сценарий | Узлов | Цена | Сравнение | Разность [95% интервал] |", "|---|---:|---:|---|---:|"]
    names = dict(fixed="без перестройки", random_swap="случайно", recent_contact="последний контакт", experience_swap="опыт")
    for r in comparisons:
        if r["metric"] == "service_fraction":
            m, lo, hi = (100 * float(r[k]) for k in ("mean", "ci_low", "ci_high"))
            lines.append(f"| {labels[r['scenario']]} | {r['n']} | {r['cost_multiplier']}× | {names[r['treatment']]} − {names[r['baseline']]} | {m:+.2f} [{lo:+.2f}; {hi:+.2f}] |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "step_deviations"}, indent=2))
    print("Saved tables.md and audit.json")
    if (OUT / "step_sensitivity.csv").exists():
        sensitivity_report()


if __name__ == "__main__":
    main()
