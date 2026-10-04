"""A staggered background pulse and autonomous requests share a local message cap."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

import experiment_local_support_signal as base

OUT = base.ROOT / "outputs" / "support_pulse_2026-10-04"
SHARES = {"fixed": None, "pulse_only": 1., "hybrid_75": .75,
          "hybrid_50": .5, "hybrid_25": .25, "signal_only": 0.}
RATES = (1., 4.)


class MessageBudget:
    """Charge every logical request/reply/handshake to its initiator.

    This is NOT a bound on each radio's received/forwarded traffic.
    A common bucket permits an initial burst of 32 messages, then rate*t.
    The signal sub-bucket reserves room in the long-run budget for pulses.
    Pulses have no separate spending allowance and use the common bucket.
    """
    def __init__(self, n, rate, pulse_share):
        self.rate = rate
        self.share = pulse_share
        self.common = np.full(n, 32.)
        self.signal = np.full(n, 32.*(1.-pulse_share))
        self.spent = np.zeros(n, dtype=int)
        self.signal_spent = np.zeros(n, dtype=int)

    def refill(self, dt):
        self.common = np.minimum(32., self.common + self.rate*dt)
        self.signal = np.minimum(32., self.signal + self.rate*(1.-self.share)*dt)

    def ready(self, channel):
        result = self.common >= 32.-1e-9
        if channel == "signal":
            result &= self.signal >= 32.-1e-9
        return result

    def charge(self, node, channel, messages):
        assert messages in (24, 32) and self.ready(channel)[node]
        self.common[node] -= messages
        self.spent[node] += messages
        if channel == "signal":
            self.signal[node] -= messages
            self.signal_spent[node] += messages


def simulate(policy, scenario, seed, p=base.Params(), rate=1.):
    share = SHARES[policy]
    w = base.make_world(seed, scenario, p)
    consumers = ~w["sources"]
    stock = np.full(p.n, p.initial_stock)
    estimates = np.full(len(w["u"]), p.prior)
    need = np.zeros(p.n, bool)
    below, above = np.zeros(p.n, int), np.zeros(p.n, int)
    last = np.full(p.n, -1e9)
    budget = MessageBudget(p.n, rate, share or 0.)
    interval = 32./(rate*share) if share else float("inf")
    # The same physical phase fractions at both integration steps.
    fractions = np.random.default_rng(seed+200000).integers(1, 41, p.n)/40.
    next_pulse = fractions*interval
    served = np.zeros((4, p.n))
    messages = np.zeros((4, 2), int)
    requests = np.zeros((4, 2), int)
    swaps = np.zeros((4, 2), int)
    max_mass = max_flow = max_excess = 0.
    minimum_gap = float("inf")
    for step in range(round(p.horizon/p.dt)):
        t = step*p.dt
        if step:
            budget.refill(p.dt)
        phase = 0 if t < 40 else 1 if t < 70 else 2 if t < 100 else 3
        available = consumers & ((t-last) >= p.request_interval-1e-9)
        pulse = available & (t >= next_pulse-1e-9) & budget.ready("pulse") if share else np.zeros(p.n, bool)
        signal = (available & need & ~pulse & budget.ready("signal")) if share is not None and share < 1. else np.zeros(p.n, bool)
        actors = sorted(map(int, np.flatnonzero(pulse | signal)),
                        key=lambda a: (a*2654435761+step*2246822519+seed) % 4294967291)
        for a in actors:
            channel = "pulse" if pulse[a] else "signal"
            j = 0 if pulse[a] else 1
            minimum_gap = min(minimum_gap, t-last[a])
            last[a] = t
            msg, swap, _ = base.request_help(a, stock, estimates, w, seed, step, p)
            budget.charge(a, channel, msg)
            messages[phase, j] += msg
            requests[phase, j] += 1
            swaps[phase, j] += swap
            if pulse[a]:
                # A delayed pulse is coalesced, never replayed as a burst.
                next_pulse[a] += (np.floor((t-next_pulse[a]+1e-9)/interval)+1)*interval
        assert np.all(budget.spent <= 32.+rate*t+1e-8)
        max_excess = max(max_excess, float(np.max(budget.spent-(32.+rate*t))))
        stock, consumption, _, error, flow = base.flow_step(
            stock, w["active"], estimates, w, base.quality_at(t, scenario, w, p), p)
        max_mass, max_flow = max(max_mass, error), max(max_flow, flow)
        service = np.ones(p.n)
        service[consumers] = consumption[consumers]/(w["demand"][consumers]*p.dt)
        base.update_need(need, below, above, service, consumers, p)
        served[phase] += consumption
    individual = served[1:, consumers].sum(axis=0)/(120.*w["demand"][consumers])
    assert p.horizon == 160.
    assert all(len(a) == 4 for a in w["adjacency"])
    assert messages.sum() == budget.spent.sum()
    assert messages.sum() == 24*requests.sum()+8*swaps.sum()
    assert minimum_gap >= p.request_interval-1e-9
    result = dict(n=p.n, scenario=scenario, seed=seed, dt=p.dt, rate=rate, policy=policy,
        service=float(served[1:].sum()/(120.*w["demand"].sum())),
        p10=float(np.quantile(individual, .1)), messages=int(messages.sum()),
        requests=int(requests.sum()), swaps=int(swaps.sum()), resource_cost=float(swaps.sum()*p.swap_cost),
        pulse_messages=int(messages[:, 0].sum()), signal_messages=int(messages[:, 1].sum()),
        pulse_requests=int(requests[:, 0].sum()), signal_requests=int(requests[:, 1].sum()),
        active_need_end=int(need.sum()), source_reach_end=base.source_reach(w),
        max_mass_error=max_mass, max_flow_ratio=max_flow, max_message_budget_excess=max_excess,
        minimum_request_gap=minimum_gap if np.isfinite(minimum_gap) else -1.,
        allowed_message_envelope=float(consumers.sum()*(32.+rate*(p.horizon-p.dt))),
        active_edges=int(w["active"].sum()))
    for i, duration in enumerate((40., 30., 30., 60.)):
        result[f"phase{i}_service"] = float(served[i].sum()/(duration*w["demand"].sum()))
        result[f"phase{i}_messages"] = int(messages[i].sum())
    return result


def run_group(case):
    n, scenario, seed, dt, rate = case
    return [simulate(policy, scenario, seed, base.Params(n=n, dt=dt), rate) for policy in SHARES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--step-check", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "step_" if args.step_check else ""
    if (args.output/(prefix+"runs.csv")).exists():
        parser.error("Refusing to overwrite results")
    cases = list(product((96,), ("repeated_damage",), range(12100, 12108), (.2, .1), RATES)) if args.step_check else list(
        product((96, 960), base.SCENARIOS, range(12000, 12020), (.2,), RATES))
    hashes = {}
    for path in (Path(__file__), Path(base.__file__)):
        data = path.read_bytes()
        (args.output/(prefix+path.name)).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    config = dict(params=asdict(base.Params()), shares=SHARES, cases=cases, hashes=hashes,
                  python=platform.python_version(), numpy=np.__version__)
    (args.output/(prefix+"config.json")).write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, group in enumerate(pool.map(run_group, cases, chunksize=1), 1):
            rows.extend(group)
            if i % 20 == 0 or i == len(cases):
                base.write_csv(args.output/(prefix+"runs.csv"), rows)
                print(f"Completed {i}/{len(cases)} worlds; {len(rows)} runs; {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
