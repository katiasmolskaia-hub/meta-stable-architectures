"""Finite-storage buffer bridge vs rate control and an ordinary queue.

Run from repository root:
    python simulations/experiment_bridge_buffer.py
Protocol: manuscripts/v6/bridge_buffer_protocol_2026-10-03.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bridge_buffer_2026-10-03"
POLICIES = ("unregulated", "rate_limiter", "ordinary_buffer", "adaptive_bridge")
SCENARIOS = ("steady", "bursty_supply", "upstream_windows", "global_outage", "delayed_sensor", "insufficient_capacity")


@dataclass(frozen=True)
class Params:
    dt: float = 0.1
    horizon: float = 90.0
    crisis_start: float = 20.0
    crisis_end: float = 30.0
    supply: float = 0.24
    demand: float = 0.18
    receiver_capacity: float = 0.32
    budget: float = 0.85
    total_storage: float = 1.2
    bridge_storage: float = 0.8
    receiver_storage: float = 0.25
    receiver_target: float = 0.20
    fill_gain: float = 1.5
    buffer_fill_gain: float = 2.0
    safety_margin: float = 0.90
    leak_rate: float = 0.003
    stress_decay: float = 0.25
    overload_stress: float = 2.0
    deficit_stress: float = 1.1
    stress_capacity_loss: float = 0.6
    permission_gain: float = 0.70
    transport_cost: float = 0.03
    holding_cost: float = 0.002
    sensor_noise: float = 0.015
    window_period: float = 4.0


def conditions(t: float, scenario: str, phase: float, p: Params):
    """Exogenous conditions; only current measured values are exposed to policies."""
    crisis = float(p.crisis_start <= t < p.crisis_end)
    if scenario == "delayed_sensor":
        crisis = max(crisis, float(48.0 <= t < 56.0))
    factor = 1.0 - 0.60 * crisis
    if scenario == "insufficient_capacity" and t >= p.crisis_start:
        factor = 0.32
    wave = ((t + phase) % p.window_period) < (p.window_period / 2)
    supply_factor = (2.0 if wave else 0.0) if scenario == "bursty_supply" else 1.0
    upstream = wave if scenario in ("upstream_windows", "global_outage") else True
    downstream = upstream if scenario == "global_outage" else True
    return factor, 0.035 * crisis, supply_factor, bool(upstream), bool(downstream)


def simulate(policy: str, scenario: str, seed: int, p: Params = Params(), keep_trace: bool = False):
    if policy not in POLICIES or scenario not in SCENARIOS:
        raise ValueError((policy, scenario))
    rng = np.random.default_rng(seed)
    demand = p.demand * rng.uniform(0.92, 1.08)
    supply = p.supply * rng.uniform(0.94, 1.06)
    capacity = p.receiver_capacity * rng.uniform(0.9, 1.1)
    shock_scale = rng.uniform(0.85, 1.15)
    phase = rng.uniform(0, p.window_period)
    steps = round(p.horizon / p.dt)
    noises = rng.normal(0, p.sensor_noise, steps)
    buffered = policy in ("ordinary_buffer", "adaptive_bridge")
    qmax = p.bridge_storage if buffered else 0.0
    amax = p.total_storage - qmax
    a, q, b, stress = 0.4, 0.0, 0.2, 0.0
    history = []
    trace = []
    totals = dict(demand=0.0, served=0.0, stress_area=0.0, rejected=0.0, leaked=0.0,
                  spilled=0.0, moved=0.0, cost=0.0, starvation_time=0.0,
                  upstream_closed_delivery=0.0, downstream_closed_delivery=0.0)
    max_mass_error = max_budget_ratio = 0.0
    full_cost = full_moved = 0.0
    for step in range(steps):
        t = step * p.dt
        before = a + b + q
        factor, shock, supply_factor, up, down = conditions(t, scenario, phase, p)
        injection = supply * supply_factor * p.dt
        a += injection
        spilled = max(0.0, a - amax)
        a = min(a, amax)
        decay = math.exp(-p.leak_rate * p.dt)
        leaked = (a + q + b) * (1 - decay)
        a, q, b = a * decay, q * decay, b * decay
        true_capacity = capacity * factor * (1 - p.stress_capacity_loss * stress)
        history.append((true_capacity, stress))
        delay_steps = round(1.5 / p.dt) if scenario == "delayed_sensor" else 0
        observed_capacity, observed_stress = history[max(0, step - delay_steps)]
        observed_capacity = max(0.0, observed_capacity * (1 + noises[step]))
        request = max(0.0, demand + p.fill_gain * (p.receiver_target - b))
        allowance = p.safety_margin * observed_capacity
        permission = 1.0
        if policy == "adaptive_bridge":
            permission = max(0.15, 1.0 - p.permission_gain * observed_stress)
        desired_release = request if policy == "unregulated" else min(request, allowance * permission)
        release = min(desired_release * p.dt, p.budget * p.dt)
        release = min(release, q if buffered else a)
        if not down or (not buffered and not up):
            release = 0.0
        # Reserve the remaining bus capacity for filling the queue. No resource
        # received this step can be released in the same step.
        fill = 0.0
        if buffered and up:
            target_q = qmax * permission
            refill_rate = max(0.0, desired_release + p.buffer_fill_gain * (target_q - q))
            fill = min(refill_rate * p.dt, a, max(0.0, qmax - q), p.budget * p.dt - release)
        if buffered:
            q += fill - release
            a -= fill
        else:
            a -= release
        accepted = min(release, true_capacity * p.dt, max(0.0, p.receiver_storage - b))
        rejected = release - accepted
        b += accepted
        wanted = demand * p.dt
        consumed = min(b, wanted)
        b -= consumed
        deficit = wanted - consumed
        stress = min(1.0, max(0.0, stress + shock * shock_scale * p.dt
                              + p.overload_stress * rejected + p.deficit_stress * deficit
                              - p.stress_decay * stress * p.dt))
        moved = fill + release
        cost = p.transport_cost * moved + p.holding_cost * (a + q) * p.dt
        full_cost += cost
        full_moved += moved
        mass_error = abs((a + b + q) - (before + injection - spilled - leaked - rejected - consumed))
        max_mass_error = max(max_mass_error, mass_error)
        max_budget_ratio = max(max_budget_ratio, moved / (p.budget * p.dt) if p.budget else 0.0)
        assert mass_error < 1e-9
        assert min(a, b, q) >= -1e-10
        assert a <= amax + 1e-9 and q <= qmax + 1e-9 and b <= p.receiver_storage + 1e-9
        assert moved <= p.budget * p.dt + 1e-10
        assert down or release == 0
        if t >= p.crisis_start:
            totals["demand"] += wanted
            totals["served"] += consumed
            totals["stress_area"] += stress * p.dt
            totals["rejected"] += rejected
            totals["leaked"] += leaked
            totals["spilled"] += spilled
            totals["moved"] += moved
            totals["cost"] += cost
            totals["starvation_time"] += float(consumed < 0.5 * wanted) * p.dt
            totals["upstream_closed_delivery"] += accepted if not up else 0
            totals["downstream_closed_delivery"] += accepted if not down else 0
        if keep_trace:
            trace.append(dict(time=t, policy=policy, scenario=scenario, seed=seed,
                              service=consumed / wanted, stress=stress, donor=a, buffer=q, receiver=b,
                              upstream_open=int(up), downstream_open=int(down), permission=permission,
                              fill_rate=fill / p.dt, release_rate=release / p.dt,
                              accepting_capacity=true_capacity, rejected_rate=rejected / p.dt))
    return {"policy": policy, "scenario": scenario, "seed": seed,
            "service_fraction": totals["served"] / totals["demand"], **totals,
            "full_cost": full_cost, "full_moved": full_moved,
            "max_mass_error": max_mass_error, "max_budget_ratio": max_budget_ratio}, trace


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    result = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            sub = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            r = dict(scenario=scenario, policy=policy, n=len(sub))
            for key in ("service_fraction", "stress_area", "rejected", "leaked", "spilled", "moved", "cost", "full_cost", "full_moved", "starvation_time"):
                x = np.array([r[key] for r in sub])
                r[key + "_mean"] = float(x.mean())
                r[key + "_sd"] = float(x.std(ddof=1))
            result.append(r)
    return result


def comparisons(rows):
    rng = np.random.default_rng(20261003)
    result = []
    for scenario in SCENARIOS:
        for treatment, baseline in (("adaptive_bridge", "ordinary_buffer"), ("ordinary_buffer", "rate_limiter"), ("adaptive_bridge", "rate_limiter")):
            a = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == treatment}
            b = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == baseline}
            for key in ("service_fraction", "stress_area", "cost", "full_cost"):
                delta = np.array([a[s][key] - b[s][key] for s in sorted(a)])
                boot = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                result.append(dict(scenario=scenario, treatment=treatment, baseline=baseline, metric=key,
                                   mean=float(delta.mean()), ci_low=float(lo), ci_high=float(hi)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--step-check", action="store_true")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("At least two seeds required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.step_check:
        rows = []
        for scenario in SCENARIOS:
            for policy in POLICIES:
                for dt in (.1, .05, .025):
                    r, _ = simulate(policy, scenario, 4900, replace(Params(), dt=dt, sensor_noise=0))
                    rows.append({"dt": dt, **r})
        write_csv(args.output / "step_refinement.csv", rows)
        print(f"Step refinement: {len(rows)} runs")
        return
    source = Path(__file__).read_bytes()
    (args.output / "experiment_snapshot.py").write_bytes(source)
    (args.output / "config.json").write_text(json.dumps(dict(params=asdict(Params()), seeds=args.seeds,
        seed_start=args.seed_start, policies=POLICIES, scenarios=SCENARIOS, python=platform.python_version(),
        numpy=np.__version__, sha256=hashlib.sha256(source).hexdigest()), indent=2), encoding="utf-8")
    rows, traces = [], []
    for scenario in SCENARIOS:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for policy in POLICIES:
                r, trace = simulate(policy, scenario, seed, keep_trace=seed == args.seed_start)
                rows.append(r)
                traces.extend(trace)
        print(f"Completed {scenario}", flush=True)
    summary = aggregate(rows)
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "paired_comparisons.csv", comparisons(rows))
    write_csv(args.output / "first_seed_traces.csv", traces)
    for r in summary:
        print(f"{r['scenario']:22} {r['policy']:17} service={r['service_fraction_mean']:.4f} "
              f"stress={r['stress_area_mean']:.3f} cost={r['cost_mean']:.3f}")


if __name__ == "__main__":
    main()
