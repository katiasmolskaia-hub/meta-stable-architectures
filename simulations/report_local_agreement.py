"""Audit the isolated agreement bench; no swarm-level inference."""
import csv
import hashlib
import json
import math
from statistics import mean

from experiment_local_agreement import OUT, POLICIES, PROFILES, write_csv


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return max(0., center-half), min(1., center+half)


def main():
    config = json.loads((OUT/"config.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((OUT/"experiment_snapshot.py").read_bytes()).hexdigest() == config["sha256"]
    with (OUT/"runs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {(p, policy, seed) for p in config["profiles"] for policy in config["policies"] for seed in config["seeds"]}
    assert len(rows) == len(expected) and {(r["profile"], r["policy"], int(r["seed"])) for r in rows} == expected
    for r in rows:
        assert int(r["max_node_messages"]) <= config["per_node_messages"]
        assert int(r["messages"]) <= 4*config["per_node_messages"]
        assert int(r["max_apply"]) <= 1 and int(r["fee_charges"]) <= 1
        assert int(r["conflict_ticks"]) == 0
        assert int(r["complete"])+int(r["partial_end"])+int(r["unchanged_end"]) == 1
    summary = []
    for profile in PROFILES:
        for policy in POLICIES:
            group = [r for r in rows if r["profile"] == profile and r["policy"] == policy]
            count = len(group)
            complete = sum(int(r["complete"]) for r in group)
            lo, hi = wilson(complete, count)
            times = [int(r["completion_time"]) for r in group if int(r["complete"])]
            summary.append(dict(profile=profile, policy=policy, runs=count,
                complete_fraction=complete/count, complete_ci_low=lo, complete_ci_high=hi,
                partial_fraction=mean(int(r["partial_end"]) for r in group),
                prepared_fraction=mean(int(r["prepared_end"]) > 0 for r in group),
                clean_abort_fraction=mean(r["decision"] == "ABORT" and int(r["prepared_end"]) == 0 and int(r["unchanged_end"]) for r in group),
                fully_acknowledged_fraction=mean(int(r["fully_acknowledged"]) for r in group),
                messages=mean(int(r["messages"]) for r in group),
                mixed_view_ticks=mean(int(r["mixed_view_ticks"]) for r in group),
                unavailable_changed_edge_node_ticks=mean(int(r["unavailable_changed_edge_node_ticks"]) for r in group),
                completion_time_conditional=mean(times) if times else "",
                budget_exhausted_fraction=mean(int(r["budget_suppressed"]) > 0 for r in group)))
    write_csv(OUT/"summary.csv", summary)
    audit = dict(runs=len(rows), unique_grid_verified=True, snapshot_hash_verified=True,
                 max_node_messages=max(int(r["max_node_messages"]) for r in rows),
                 max_fee_charges=max(int(r["fee_charges"]) for r in rows),
                 conflicting_final_decision_runs=sum(int(r["conflict_ticks"]) > 0 for r in rows))
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Изолированное согласование: полные результаты", "", "По 200 seed на строку. Отказ от обмена не считается успешным созданием новых связей. Незавершённость измеряется на такте 80, не на бесконечном горизонте.", "",
             "| Канал | Вариант | Обмен завершён, % | Частичная смена к концу, % | Есть зависшие обещания, % | Чистый отказ от обмена, % | Сообщений, среднее |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['profile']} | {r['policy']} | {100*r['complete_fraction']:.1f} | {100*r['partial_fraction']:.1f} | {100*r['prepared_fraction']:.1f} | {100*r['clean_abort_fraction']:.1f} | {r['messages']:.2f} |")
    (OUT/"tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(audit))
    for r in summary:
        if r["policy"] not in ("fixed", "ideal_atomic"):
            print(json.dumps(r))


if __name__ == "__main__":
    main()
