"""Audit saved trajectories and report paired, fixed-calibration uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_recovery_history import DEFAULT_OUT, LAWS, POLICIES, SCENARIOS, evaluate


def report(out):
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    worlds = config["worlds"]
    for filename, key in (("experiment_snapshot.py", "code_sha256"), ("protocol_snapshot.md", "protocol_sha256")):
        assert hashlib.sha256((out / filename).read_bytes()).hexdigest() == config[key]
    # Git may convert working-source line endings on Windows; archive bytes stay exact.
    live = Path(__file__).with_name("experiment_recovery_history.py").read_bytes()
    snapshot = (out / "experiment_snapshot.py").read_bytes()
    assert live.replace(b"\r\n", b"\n") == snapshot.replace(b"\r\n", b"\n")
    calibration, histories, futures = {}, {}, {}
    for law in LAWS:
        with np.load(out / f"calibration_{law}.npz") as data:
            calibration[law] = data["history"], data["future"]
    for scenario in SCENARIOS:
        with np.load(out / f"checkpoints_{scenario}.npz") as data:
            histories[scenario], futures[scenario] = data["history"], data["future"]
    with (out / "runs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    replayed = evaluate(histories, futures, calibration, worlds)
    assert len(rows) == len(replayed) == len(SCENARIOS) * worlds * len(POLICIES)
    for saved, replay in zip(rows, replayed):
        assert saved == {k: str(v) for k, v in replay.items()}, (saved, replay)
    grouped = {(s, p): sorted([r for r in rows if r["scenario"] == s and r["policy"] == p],
                             key=lambda r: int(r["world"])) for s in SCENARIOS for p in POLICIES}
    for group in grouped.values():
        assert [int(r["world"]) for r in group] == list(range(worlds))
    def values(s, p, metric="quality"):
        return np.array([float(r[metric]) for r in grouped[s, p]])
    means = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            means.append(dict(scenario=scenario, policy=policy, worlds=worlds,
                **{m: float(values(scenario, policy, m).mean()) for m in
                   ("quality", "early_quality", "regret", "prediction_mse", "selected_failure_age")}))
    comparisons = []
    for i, scenario in enumerate(SCENARIOS):
        indices = np.random.default_rng(20000 + i).integers(0, worlds, size=(5000, worlds))
        for baseline in ("last_contact", "ema", "failure_age", "shuffled_history"):
            for metric in ("quality", "early_quality"):
                delta = values(scenario, "route_history", metric) - values(scenario, baseline, metric)
                lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                comparisons.append(dict(scenario=scenario, baseline=baseline, metric=metric,
                    delta_pp=float(delta.mean() * 100), lower_pp=float(lo * 100), upper_pp=float(hi * 100),
                    primary=int(scenario == "finite_duration" and baseline == "failure_age" and metric == "quality")))
    for name, records in (("summary.csv", means), ("paired_comparisons.csv", comparisons)):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    audit = dict(exact_replay_rows=len(rows), worlds_per_scenario=worlds,
        unique_scenario_worlds=worlds * len(SCENARIOS), bootstrap_replicates=5000,
        intervals="paired by world; conditional on one fixed calibration per law",
        code_and_protocol_hashes_verified=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (out / "report_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(audit=audit, means=means,
        comparisons=[r for r in comparisons if r["metric"] == "quality"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    report(parser.parse_args().out)
