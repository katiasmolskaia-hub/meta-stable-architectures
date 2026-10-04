"""Independent bridge experiment: resource transport, endogenous stress, local feedback.

Run: python simulations/experiment_bridge_feedback.py --seeds 40
Only NumPy and (for plots) Matplotlib are required. No Volume VI code is modified.
See manuscripts/v6/bridge_feedback_protocol.md for hypotheses and limitations.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ((0, 1), (0, 2), (1, 2))
POLICIES = ("old_repair", "threshold", "feedback_gate", "lifecycle", "no_memory")
SCENARIOS = ("unchanged", "persistent_damage", "temporary_damage", "no_safe_detour", "changed_again")


@dataclass(frozen=True)
class Params:
    dt: float = 0.1
    horizon: float = 100.0
    crisis_start: float = 20.0
    crisis_end: float = 30.0
    change_again: float = 65.0
    budget: float = 0.65
    supply: float = 0.30
    demand_b: float = 0.17
    demand_c: float = 0.10
    transport_gain: float = 1.4
    observation_sigma: float = 0.08
    learning_rate: float = 3.0
    probe_weight: float = 0.035
    gate_threshold: float = 0.65
    cooldown: float = 5.0
    gate_sample_volume: float = 0.025
    damage_quality: float = 0.12
    good_quality: float = 0.96
    contact_cost: float = 0.015
    reserve_cost: float = 0.18
    stress_decay: float = 0.22
    failure_stress: float = 1.3
    recovery_hold: float = 5.0


def environment(t: float, scenario: str, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Hidden link quality and external shock. Never passed to a policy."""
    quality = np.full(3, p.good_quality)
    shock = np.zeros(3)
    if p.crisis_start <= t < p.crisis_end:
        shock[1] = 0.065
    if scenario != "unchanged" and t >= p.crisis_start:
        quality[0] = p.damage_quality
    if scenario == "temporary_damage" and t >= p.crisis_end:
        quality[0] = p.good_quality
    if scenario == "no_safe_detour" and t >= p.crisis_start:
        quality[2] = p.damage_quality
    if scenario == "changed_again" and t >= p.change_again:
        quality[0] = p.good_quality
        quality[2] = p.damage_quality
    return quality, shock


def recovery_delay(times: np.ndarray, service: np.ndarray, stress: np.ndarray, p: Params) -> float:
    """First sustained healthy service window; censored runs remain NaN."""
    hold = max(1, round(p.recovery_hold / p.dt))
    healthy = (service >= 0.95) & (stress < 0.20)
    first = int(np.searchsorted(times, p.crisis_end))
    for i in range(first, len(times) - hold + 1):
        if healthy[i:i + hold].all():
            return float(times[i] - p.crisis_end)
    return float("nan")


def simulate(policy: str, scenario: str, seed: int, p: Params = Params(), keep_trace: bool = False):
    if policy not in POLICIES or scenario not in SCENARIOS:
        raise ValueError((policy, scenario))
    rng = np.random.default_rng(seed)
    # Common random numbers: every policy sees the same exogenous environment.
    demand_scale = rng.uniform(0.92, 1.08, 2)
    supply_scale = rng.uniform(0.96, 1.04)
    crisis_scale = rng.uniform(0.8, 1.2)
    steps = round(p.horizon / p.dt)
    noise = rng.normal(0, p.observation_sigma, (steps, 3))
    stock = np.array([1.40, 0.75, 0.75])
    stress = np.zeros(3)
    reserve = np.full(3, 0.8)
    estimate = np.array([0.9, 0.9, 0.5])
    reopen_at = np.zeros(3)
    gate_volume = np.zeros(3)
    gate_outcome = np.zeros(3)
    service_history, stress_history, times, trace = [], [], [], []
    totals = dict(demand=0.0, served=0.0, sent=0.0, lost=0.0, resource_cost=0.0,
                  stress_area=0.0, donor_stress_area=0.0, isolation_time=0.0,
                  overload_time=0.0, detour_volume=0.0, damaged_edge_volume=0.0)
    max_budget_ratio = 0.0
    max_mass_error = 0.0
    for step in range(steps):
        t = step * p.dt
        quality, shock = environment(t, scenario, p)
        shock *= crisis_scale
        # External resource balance precedes transport; transfers are simultaneous.
        before = stock.sum()
        injected = p.supply * supply_scale * p.dt
        stock[0] += injected
        spill = max(0.0, stock[0] - 2.0)
        stock[0] = min(2.0, stock[0])
        desired = np.zeros(3)
        senders, receivers = [], []
        for edge, (a, b) in enumerate(PAIRS):
            sender, receiver = (a, b) if stock[a] >= stock[b] else (b, a)
            senders.append(sender)
            receivers.append(receiver)
            gradient = abs(stock[a] - stock[b])
            readiness = min(reserve[a], reserve[b]) * max(0.0, 1.0 - max(stress[a], stress[b]))
            if policy == "old_repair":
                weight = (1.0, 0.4, 0.0)[edge]
            elif policy == "threshold":
                weight = float(max(stress[a], stress[b]) < p.gate_threshold)
            elif policy == "feedback_gate":
                weight = float(t >= reopen_at[edge] and max(stress[a], stress[b]) < p.gate_threshold)
            else:
                # No-memory receives identical readiness and probing; only the
                # retained assessment of contact quality is removed.
                belief = estimate[edge]
                weight = p.probe_weight + (1.0 - p.probe_weight) * belief ** 3
            desired[edge] = p.transport_gain * gradient * readiness * weight
        if desired.sum() > p.budget:
            desired *= p.budget / desired.sum()
        amount = desired * p.dt
        # A node cannot spend stock received during this same step.
        for node in range(3):
            outgoing = [e for e in range(3) if senders[e] == node]
            request = sum(amount[e] for e in outgoing)
            if request > stock[node]:
                amount[outgoing] *= stock[node] / request
        if p.budget > 0:
            max_budget_ratio = max(max_budget_ratio, float(amount.sum() / (p.budget * p.dt)))
        loss = amount * (1.0 - quality)
        incoming_harm = np.zeros(3)
        draw = np.zeros(3)
        changes = np.zeros(3)
        for edge, (a, b) in enumerate(PAIRS):
            changes[senders[edge]] -= amount[edge]
            changes[receivers[edge]] += amount[edge] - loss[edge]
            incoming_harm[a] += loss[edge]
            incoming_harm[b] += loss[edge]
            fee = p.contact_cost * amount[edge] + p.reserve_cost * amount[edge]
            draw[a] += fee / 2
            draw[b] += fee / 2
            if amount[edge] > 1e-10:
                observation = float(np.clip(quality[edge] + noise[step, edge], 0.0, 1.0))
                if policy == "lifecycle":
                    alpha = 1.0 - np.exp(-p.learning_rate * amount[edge])
                    estimate[edge] += alpha * (observation - estimate[edge])
                elif policy == "feedback_gate":
                    # A decision requires a finite amount of attempted transport,
                    # not one integration tick (which becomes free as dt -> 0).
                    gate_volume[edge] += amount[edge]
                    gate_outcome[edge] += amount[edge] * observation
                    if gate_volume[edge] >= p.gate_sample_volume:
                        if gate_outcome[edge] / gate_volume[edge] < 0.5:
                            reopen_at[edge] = t + p.cooldown
                        gate_volume[edge] = 0.0
                        gate_outcome[edge] = 0.0
        stock += changes
        demand = np.array([0.0, p.demand_b * demand_scale[0], p.demand_c * demand_scale[1]]) * p.dt
        consumed = np.minimum(stock, demand)
        stock -= consumed
        unmet = demand - consumed
        stress = np.clip(stress + shock * p.dt + 1.8 * unmet
                         + p.failure_stress * incoming_harm - p.stress_decay * stress * p.dt, 0, 1)
        reserve = np.clip(reserve + 0.10 * (1.0 - stress) * p.dt - draw, 0, 1)
        mass_error = abs(stock.sum() - (before + injected - spill - loss.sum() - consumed.sum()))
        max_mass_error = max(max_mass_error, mass_error)
        assert mass_error < 1e-9, "Resource is not conserved"
        assert np.all(stock >= -1e-10) and np.isfinite(stock).all()
        assert amount.sum() <= p.budget * p.dt + 1e-10
        service_b = float(consumed[1] / demand[1])
        service_history.append(service_b)
        stress_history.append(float(stress[1]))
        times.append(t)
        if t >= p.crisis_start:
            totals["demand"] += float(demand[1:].sum())
            totals["served"] += float(consumed[1:].sum())
            totals["sent"] += float(amount.sum())
            totals["lost"] += float(loss.sum())
            totals["resource_cost"] += float(draw.sum())
            totals["stress_area"] += float(stress.sum()) * p.dt
            totals["donor_stress_area"] += float(stress[0]) * p.dt
            totals["isolation_time"] += float(service_b < 0.5) * p.dt
            totals["overload_time"] += float(np.any(stress > 0.5)) * p.dt
            totals["detour_volume"] += float(amount[2])
            totals["damaged_edge_volume"] += float(amount[quality < 0.5].sum())
        if keep_trace:
            row = {"time": t, "policy": policy, "scenario": scenario, "seed": seed,
                   "service_b": service_b, "transport_rate": float(amount.sum() / p.dt)}
            for i, name in enumerate("ABC"):
                row[f"stock_{name}"] = float(stock[i])
                row[f"stress_{name}"] = float(stress[i])
            for i, name in enumerate(("AB", "AC", "BC")):
                row[f"flow_{name}"] = float(amount[i] / p.dt)
                row[f"estimate_{name}"] = float(estimate[i])
            trace.append(row)
    recovery = recovery_delay(np.array(times), np.array(service_history), np.array(stress_history), p)
    metrics = {"policy": policy, "scenario": scenario, "seed": seed,
               "service_fraction": totals["served"] / totals["demand"],
               "recovered": int(np.isfinite(recovery)), "recovery_delay": recovery,
               # Capped waiting time includes failures, unlike recovered-only means.
               "capped_recovery_delay": recovery if np.isfinite(recovery) else p.horizon - p.crisis_end,
               **totals, "max_budget_ratio": max_budget_ratio, "max_mass_error": max_mass_error}
    return metrics, trace


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    keys = ("service_fraction", "recovered", "capped_recovery_delay", "lost", "sent",
            "resource_cost", "stress_area", "donor_stress_area", "isolation_time", "detour_volume")
    results = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            selected = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            if not selected:
                continue
            row = {"scenario": scenario, "policy": policy, "n": len(selected)}
            for key in keys:
                vals = np.array([r[key] for r in selected])
                row[key + "_mean"] = float(vals.mean())
                row[key + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            results.append(row)
    return results


def paired_comparisons(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(1701)
    comparisons = []
    for scenario in SCENARIOS:
        full = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == "lifecycle"}
        for baseline in ("old_repair", "threshold", "feedback_gate", "no_memory"):
            base = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["policy"] == baseline}
            for metric in ("service_fraction", "stress_area", "resource_cost", "capped_recovery_delay"):
                delta = np.array([full[s][metric] - base[s][metric] for s in sorted(full)])
                boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                comparisons.append(dict(scenario=scenario, baseline=baseline, metric=metric,
                                        delta_mean=float(delta.mean()), ci_low=float(lo), ci_high=float(hi)))
    return comparisons


def make_plot(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    colors = ("#8c8c8c", "#e69f00", "#009e73", "#0072b2", "#cc79a7")
    x = np.arange(len(SCENARIOS))
    labels = ("Unchanged", "Persistent\ndamage", "Temporary\ndamage", "No safe\ndetour", "Changed\nagain")
    for j, policy in enumerate(POLICIES):
        sub = [next(r for r in summary if r["scenario"] == s and r["policy"] == policy) for s in SCENARIOS]
        for ax, key in zip(axes, ("service_fraction", "stress_area")):
            ax.bar(x + (j - 2) * 0.15, [r[key + "_mean"] for r in sub], width=0.15,
                   color=colors[j], label=policy,
                   yerr=[r[key + "_sd"] for r in sub], capsize=2)
            ax.set_xticks(x, labels)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Demand served (higher is better)")
    axes[1].set_ylabel("Integrated group stress (lower is better)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Bridge feedback experiment | means and seed SD | synthetic resource network")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--no-plot", action="store_true", help="Compute results without Matplotlib")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "bridge_feedback_v2")
    args = parser.parse_args()
    if args.seeds < 2 or args.dt <= 0:
        parser.error("At least two seeds and a positive dt are required")
    p = replace(Params(), dt=args.dt)
    args.output.mkdir(parents=True, exist_ok=True)
    import hashlib
    import platform
    source = Path(__file__).read_bytes()
    (args.output / "experiment_snapshot.py").write_bytes(source)
    (args.output / "runtime.json").write_text(json.dumps({"python": platform.python_version(),
        "numpy": np.__version__, "source_sha256": hashlib.sha256(source).hexdigest()}, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps({"params": asdict(p), "seed_start": args.seed_start,
        "seeds": args.seeds, "policies": POLICIES, "scenarios": SCENARIOS}, indent=2), encoding="utf-8")
    rows, trace = [], []
    for scenario in SCENARIOS:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for policy in POLICIES:
                metrics, series = simulate(policy, scenario, seed, p, keep_trace=seed == args.seed_start)
                rows.append(metrics)
                trace.extend(series)
        print(f"Completed {scenario}", flush=True)
    summary = aggregate(rows)
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "paired_comparisons.csv", paired_comparisons(rows))
    write_csv(args.output / "first_seed_traces.csv", trace)
    if not args.no_plot:
        make_plot(summary, args.output / "comparison.png")
    print(f"Saved {len(rows)} runs to {args.output}")
    for r in summary:
        print(f"{r['scenario']:19} {r['policy']:14} service={r['service_fraction_mean']:.3f} "
              f"stress={r['stress_area_mean']:.2f} recovery={r['recovered_mean']:.2f}")


if __name__ == "__main__":
    main()
