"""Step refinement under noiseless observations; no parameter tuning."""
from dataclasses import replace
from pathlib import Path

from experiment_bridge_feedback import Params, simulate, write_csv, SCENARIOS


def main():
    out = Path(__file__).resolve().parents[1] / "outputs" / "bridge_feedback_v2"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for policy in ("threshold", "feedback_gate", "lifecycle", "no_memory"):
            for dt in (0.1, 0.05, 0.025):
                p = replace(Params(), dt=dt, observation_sigma=0)
                m, _ = simulate(policy, scenario, 3000, p)
                rows.append({"dt": dt, **m})
        print(f"Step refinement: {scenario}", flush=True)
    write_csv(out / "step_refinement.csv", rows)
    for scenario in SCENARIOS:
        for policy in ("threshold", "feedback_gate", "lifecycle", "no_memory"):
            sub = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            service = [r["service_fraction"] for r in sub]
            stress = [r["stress_area"] for r in sub]
            print(f"{scenario:19} {policy:14} service={service} stress={stress}")


if __name__ == "__main__":
    main()
