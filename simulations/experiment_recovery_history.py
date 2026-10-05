"""History information gate: channel selection, not a full recovery network."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/recovery_history_2026-10-05"
PAST, FUTURE, BURN, CANDIDATES = 64, 16, 128, 8
LAWS = ("memoryless", "finite_duration", "stable_types")
SCENARIOS = LAWS + ("regime_reset",)
POLICIES = ("last_contact", "ema", "failure_age", "route_history", "shuffled_history", "oracle")


def generate(count, law, seed):
    """Draw independent channels conditional ONLY on current observed failure."""
    if law not in LAWS:
        raise ValueError(law)
    rng = np.random.default_rng(seed)
    histories, futures = [], []
    remaining = count
    while remaining:
        batch = max(32, remaining * 3)
        state = rng.random(batch) < .5
        slow = rng.random(batch) < .5
        low = np.where(slow, 16, 4) if law == "stable_types" else np.full(batch, 4)
        high = np.where(slow, 28, 8) if law == "stable_types" else np.full(batch, 20)
        duration = rng.integers(low, high + 1)
        trajectory = np.empty((batch, PAST + FUTURE), dtype=bool)
        for t in range(BURN + PAST + FUTURE):
            if law == "memoryless":
                state ^= rng.random(batch) < .1
            else:
                switch = duration == 0
                state[switch] ^= True
                duration[switch] = rng.integers(low[switch], high[switch] + 1)
                duration -= 1
            if t >= BURN:
                trajectory[:, t - BURN] = state
        accepted = trajectory[~trajectory[:, PAST - 1]][:remaining]
        histories.append(accepted[:, :PAST])
        futures.append(accepted[:, PAST:])
        remaining -= len(accepted)
    return np.concatenate(histories), np.concatenate(futures)


def reset_future(count, seed):
    rng = np.random.default_rng(seed)
    state = np.zeros(count, dtype=bool)
    future = np.empty((count, FUTURE), dtype=bool)
    for t in range(FUTURE):
        state ^= rng.random(count) < .1
        future[:, t] = state
    return future


def ages(history):
    if history.ndim != 2 or history.shape[1] != PAST or history[:, -1].any():
        raise ValueError("Expected 64 observations ending in failure")
    reverse = history[:, ::-1]
    return np.where(reverse.any(axis=1), np.argmax(reverse, axis=1), PAST)


def features(history):
    age = ages(history)
    previous_good = np.zeros(len(history), dtype=int)
    for i, row in enumerate(history):
        j = PAST - int(age[i]) - 1
        while j >= 0 and row[j]:
            previous_good[i] += 1
            j -= 1
    flips = np.count_nonzero(history[:, -32:-1] != history[:, -31:], axis=1)
    weights = .8 ** np.arange(PAST - 1, -1, -1)
    ema = history @ weights / weights.sum()
    age_key = np.minimum(age, 32)
    return age_key, np.minimum((ema * 10).astype(int), 9), list(zip(
        age_key.tolist(), np.minimum(previous_good // 4, 7).tolist(),
        np.minimum(flips // 2, 7).tolist()))


def shuffle_history(history, seed):
    """Destroy older order; preserve failure age, its boundary, and state counts."""
    rng = np.random.default_rng(seed)
    shuffled = history.copy()
    for i, age in enumerate(ages(history)):
        stop = PAST - int(age) - 1
        if stop > 0:
            shuffled[i, :stop] = rng.permutation(shuffled[i, :stop])
    return shuffled


def table(keys, outcomes):
    result = {}
    for key, y in zip(keys, outcomes):
        total, count = result.get(key, (0., 0))
        result[key] = (total + float(y), count + 1)
    return result


def smooth(stats, prior, strength):
    total, count = stats
    return (total + strength * prior) / (count + strength)


def fit(history, future):
    age, ema, route = features(history)
    y = future.mean(axis=1)
    return dict(mean=float(y.mean()), age=table(age, y), ema=table(ema, y), route=table(route, y))


def predict(model, history, policy):
    """No future or hidden type is available to this interface."""
    age, ema, route = features(history)
    base = model["mean"]
    age_score = np.array([smooth(model["age"].get(k, (0., 0)), base, 16) for k in age])
    if policy == "last_contact":
        return np.full(len(history), base)
    if policy == "ema":
        return np.array([smooth(model["ema"].get(k, (0., 0)), base, 16) for k in ema])
    if policy == "failure_age":
        return age_score
    if policy in ("route_history", "shuffled_history"):
        return np.array([smooth(model["route"].get(k, (0., 0)), a, 32)
                         for k, a in zip(route, age_score)])
    raise ValueError(policy)


def choose(scores, order):
    return int(order[np.argmax(scores[order])])


def evaluate(histories, futures, calibration, worlds):
    models = {law: fit(*calibration[law]) for law in LAWS}
    shuffled_models = {law: fit(shuffle_history(calibration[law][0], 15100 + i), calibration[law][1])
                       for i, law in enumerate(LAWS)}
    rows = []
    for i, scenario in enumerate(SCENARIOS):
        law = "stable_types" if scenario == "regime_reset" else scenario
        history, future = histories[scenario], futures[scenario]
        shuffled = shuffle_history(history, 18000 + i)
        scores = {p: predict(shuffled_models[law] if p == "shuffled_history" else models[law],
                             shuffled if p == "shuffled_history" else history, p)
                  for p in POLICIES if p != "oracle"}
        scores["oracle"] = future.mean(axis=1)
        quality = .2 + .78 * future.mean(axis=1)
        early = .2 + .78 * future[:, :4].mean(axis=1)
        age = ages(history)
        for world in range(worlds):
            sl = slice(world * CANDIDATES, (world + 1) * CANDIDATES)
            order = np.random.default_rng(19000 + i * 1000 + world).permutation(CANDIDATES)
            for policy in POLICIES:
                score = scores[policy][sl]
                selected = choose(score, order)
                rows.append(dict(scenario=scenario, world=world, policy=policy, selected=selected,
                    quality=float(quality[sl][selected]), early_quality=float(early[sl][selected]),
                    regret=float(quality[sl].max() - quality[sl][selected]),
                    prediction_mse=float(np.mean((.2 + .78 * score - quality[sl]) ** 2)),
                    selected_failure_age=int(age[sl][selected]), common_probe_samples=PAST * CANDIDATES))
    return rows


def run(out, worlds=400, training=30000):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    calibration, histories, futures = {}, {}, {}
    for i, law in enumerate(LAWS):
        h, f = generate(training, law, 15000 + i)
        calibration[law] = h, f
        np.savez_compressed(out / f"calibration_{law}.npz", history=h, future=f)
    for i, scenario in enumerate(SCENARIOS):
        law = "stable_types" if scenario == "regime_reset" else scenario
        h, f = generate(worlds * CANDIDATES, law, 16000 + i)
        if scenario == "regime_reset":
            f = reset_future(len(h), 17003)
        histories[scenario], futures[scenario] = h, f
        np.savez_compressed(out / f"checkpoints_{scenario}.npz", history=h, future=f)
    rows = evaluate(histories, futures, calibration, worlds)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source = Path(__file__).read_bytes()
    (out / "experiment_snapshot.py").write_bytes(source)
    protocol = ROOT / "manuscripts/v6/recovery_history_protocol_2026-10-05.md"
    (out / "protocol_snapshot.md").write_bytes(protocol.read_bytes())
    config = dict(date="2026-10-05", worlds=worlds, training_per_law=training,
        past=PAST, future=FUTURE, burn=BURN, candidates=CANDIDATES,
        policies=POLICIES, scenarios=SCENARIOS, python=platform.python_version(), numpy=np.__version__,
        primary="finite_duration: route_history minus failure_age; mean selected quality over 16 steps",
        limitation="Shared diagnostic histories of all candidates; no diagnostic, switching, or network costs",
        code_sha256=hashlib.sha256(source).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(rows=len(rows), out=str(out))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.out)
