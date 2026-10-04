"""Four-party link-swap agreement bench, NOT a full swarm/resource simulation."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "local_agreement_2026-10-04"
POLICIES = ("fixed", "ideal_atomic", "command_once", "confirmed_once", "confirmed_retry")
PROFILES = {
    "reliable_delay": dict(loss=0., max_delay=4, duplicate=0.),
    "loss_10": dict(loss=.1, max_delay=4, duplicate=0.),
    "loss_30": dict(loss=.3, max_delay=4, duplicate=0.),
    "loss_duplicates": dict(loss=.1, max_delay=4, duplicate=.3),
    "temporary_partition": dict(loss=0., max_delay=4, duplicate=0., outage_until=24),
    "final_never_arrives": dict(loss=0., max_delay=4, duplicate=0., lose_final=True),
}


class Bench:
    """One transaction, four live nodes, an initiator with a persistent decision.

    Each node changes just its own neighbor record. Reciprocal edge availability
    is measured separately from agreement on the transaction's final decision.
    Old partners: 0-1, 2-3; new partners: 0-2, 1-3. No shared atomic graph update.
    """
    def __init__(self, policy, profile, seed, limit=24, horizon=80):
        self.policy, self.profile, self.seed = policy, profile, seed
        self.limit, self.horizon = limit, horizon
        self.state = ["old"]*4
        self.applied = [0]*4
        self.sent = [0]*4
        self.queue = []
        self.serial = 0
        self.t = 0
        self.votes, self.acks = {0}, {0}
        self.decision = None
        self.last_round = -1
        self.dropped = self.suppressed = self.duplicates = self.delivered = 0
        self.mixed_ticks = self.unavailable_node_ticks = self.conflict_ticks = 0
        self.completion_time = self.agreement_time = -1
        self.max_prepared = 0

    def send(self, src, dst, kind):
        if self.sent[src] >= self.limit:
            self.suppressed += 1
            return
        self.sent[src] += 1
        p = self.profile
        # Exogenous directed-link condition at each integer time, shared across
        # policies. Simultaneous messages on the same directed link share fate.
        rng = random.Random(self.seed*1000003+self.t*10007+src*101+dst*13)
        lost = rng.random() < p["loss"]
        lost |= 3 in (src, dst) and self.t < p.get("outage_until", 0)
        lost |= bool(p.get("lose_final")) and dst == 3 and kind in ("COMMIT", "ABORT", "SWITCH")
        delay = rng.randint(1, p["max_delay"])
        duplicate = rng.random() < p["duplicate"]
        if lost:
            self.dropped += 1
            return
        self.enqueue(self.t+delay, src, dst, kind)
        if duplicate:
            self.duplicates += 1
            self.enqueue(self.t+delay+rng.randint(1, 8), src, dst, kind)

    def enqueue(self, when, src, dst, kind):
        self.serial += 1
        heapq.heappush(self.queue, (when, self.serial, src, dst, kind))

    def apply(self, node):
        if self.state[node] != "new":
            assert self.state[node] != "aborted"
            self.state[node] = "new"
            self.applied[node] += 1

    def decide(self, decision):
        assert self.decision is None
        self.decision = decision
        if decision == "COMMIT":
            assert self.votes == {0, 1, 2, 3}
            self.apply(0)
        else:
            self.state[0] = "aborted"
        self.acks = {0}
        self.broadcast()

    def broadcast(self):
        self.last_round = self.t
        waiting = self.acks if self.decision else self.votes
        for node in (1, 2, 3):
            if node not in waiting:
                self.send(0, node, self.decision or "PREPARE")

    def receive(self, src, dst, kind):
        self.delivered += 1
        if kind == "SWITCH":
            self.apply(dst)
        elif kind == "PREPARE":
            if self.state[dst] == "old":
                self.state[dst] = "prepared"
            if self.state[dst] == "prepared":
                self.send(dst, 0, "YES")
            else:
                self.send(dst, 0, "ACK_"+("COMMIT" if self.state[dst] == "new" else "ABORT"))
        elif kind == "YES":
            if self.decision is None:
                self.votes.add(src)
                if len(self.votes) == 4:
                    self.decide("COMMIT")
        elif kind in ("COMMIT", "ABORT"):
            if kind == "COMMIT":
                assert self.state[dst] in ("prepared", "new")
                self.apply(dst)
            else:
                assert self.state[dst] != "new"
                self.state[dst] = "aborted"
            self.send(dst, 0, "ACK_"+kind)
        elif kind.startswith("ACK_") and kind[4:] == self.decision:
            self.acks.add(src)
        else:
            assert kind.startswith("ACK_")

    def measure(self):
        new = [s == "new" for s in self.state]
        mixed = any(new) and not all(new)
        self.mixed_ticks += mixed
        self.conflict_ticks += any(new) and "aborted" in self.state
        old_partner, new_partner = (1, 0, 3, 2), (2, 3, 0, 1)
        for node in range(4):
            partner = new_partner[node] if new[node] else old_partner[node]
            other = new_partner[partner] if new[partner] else old_partner[partner]
            self.unavailable_node_ticks += other != node
        self.max_prepared = max(self.max_prepared, self.state.count("prepared"))
        if all(new) and self.completion_time < 0:
            self.completion_time = self.t
        if len(self.acks) == 4 and self.decision and self.agreement_time < 0:
            self.agreement_time = self.t

    def run(self):
        if self.policy == "ideal_atomic":
            for node in range(4):
                self.apply(node)
        elif self.policy == "command_once":
            self.apply(0)
            for node in (1, 2, 3):
                self.send(0, node, "SWITCH")
        elif self.policy in ("confirmed_once", "confirmed_retry"):
            self.state[0] = "prepared"
            self.broadcast()
        else:
            assert self.policy == "fixed"
        for t in range(self.horizon):
            self.t = t
            while self.queue and self.queue[0][0] <= t:
                _, _, src, dst, kind = heapq.heappop(self.queue)
                self.receive(src, dst, kind)
            if self.policy in ("confirmed_once", "confirmed_retry"):
                # Only the undecided initiator may abort at its deadline.
                # A participant must not roll back a prepared transaction by
                # its own timeout: a COMMIT may already exist elsewhere.
                if self.decision is None and t >= 16:
                    self.decide("ABORT")
                if self.policy == "confirmed_retry" and t % 4 == 0 and self.last_round != t and len(self.acks) < 4:
                    self.broadcast()
            self.measure()
        assert max(self.applied) <= 1 and max(self.sent) <= self.limit
        count_new = self.state.count("new")
        return dict(policy=self.policy, seed=self.seed, complete=int(count_new == 4),
            partial_end=int(0 < count_new < 4), unchanged_end=int(count_new == 0),
            prepared_end=self.state.count("prepared"), fully_acknowledged=int(self.agreement_time >= 0),
            decision=self.decision or "none", messages=sum(self.sent), lost=self.dropped,
            duplicates=self.duplicates, delivered=self.delivered, budget_suppressed=self.suppressed,
            max_node_messages=max(self.sent), mixed_view_ticks=self.mixed_ticks,
            unavailable_changed_edge_node_ticks=self.unavailable_node_ticks,
            conflict_ticks=self.conflict_ticks, completion_time=self.completion_time,
            agreement_time=self.agreement_time, max_prepared=self.max_prepared,
            fee_charges=self.applied[2], max_apply=max(self.applied))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"runs.csv").exists():
        raise FileExistsError("Refusing to overwrite results")
    source = Path(__file__).read_bytes()
    (OUT/"experiment_snapshot.py").write_bytes(source)
    config = dict(policies=POLICIES, profiles=PROFILES, seeds=list(range(14000, 14200)),
                  horizon=80, per_node_messages=24, abort_deadline=16, retry_interval=4,
                  sha256=hashlib.sha256(source).hexdigest(),
                  scope="Single isolated transaction. Four live nodes. No resource dynamics, competing transactions, or coordinator failure.")
    (OUT/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    for profile, params in PROFILES.items():
        for seed in config["seeds"]:
            for policy in POLICIES:
                rows.append(dict(profile=profile, **Bench(policy, params, seed).run()))
        print(f"Completed {profile}; {len(rows)} runs", flush=True)
    write_csv(OUT/"runs.csv", rows)


if __name__ == "__main__":
    main()
