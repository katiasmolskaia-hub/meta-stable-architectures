"""
Parameter sweep for networked Meta-Stable Architectures (no metronome).

Outputs:
- CSV with recovery_time, peak isolation, peak mean H, final mean S

Usage:
    python simulations/sweep_network_v1.py
"""

from __future__ import annotations

import csv
from dataclasses import asdict

import numpy as np

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulations.simulation_network_v1 import NetworkParams, MasterParams, simulate_network


def _frange(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def main() -> None:
    # Coarse sweep defaults (fits within typical timeouts).
    # Full grid can be enabled by setting COARSE = False.
    COARSE = True

    if COARSE:
        stress_amps = _frange(0.5, 3.0, 0.5)
        stress_fracs = _frange(0.1, 0.6, 0.25)
        couplings = _frange(0.05, 0.3, 0.1)
        net_base = NetworkParams(
            n_agents=24,
            t_end=60.0,
            dt=0.03,
            ring_k=2,
            stress_duration=2.0,
            stress_y_amp=1.0,
        )
    else:
        # User-specified full ranges
        stress_amps = _frange(0.5, 3.0, 0.2)
        stress_fracs = _frange(0.1, 0.6, 0.1)
        couplings = _frange(0.05, 0.3, 0.05)
        net_base = NetworkParams(
            n_agents=24,
            t_end=80.0,
            dt=0.02,
            ring_k=2,
            stress_duration=2.0,
            stress_y_amp=1.0,
        )

    out_path = "E:\\MyProject\\meta-stable-architectures\\outputs\\network_sweep.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stress_amp",
                "stress_frac",
                "coupling",
                "recovery_time",
                "peak_mean_S",
                "peak_fraction_isolated",
                "peak_mean_H",
                "final_mean_S",
            ]
        )

        total = len(stress_amps) * len(stress_fracs) * len(couplings)
        idx = 0
        for amp in stress_amps:
            for frac in stress_fracs:
                for coup in couplings:
                    idx += 1
                    net = NetworkParams(**asdict(net_base))
                    net.stress_amp = amp
                    net.stress_frac = frac
                    net.coupling = coup

                    out = simulate_network(net=net)
                    writer.writerow(
                        [
                            amp,
                            frac,
                            coup,
                            out["recovery_time"],
                            float(np.max(out["mean_s"])),
                            float(np.max(out["fraction_isolated"])),
                            float(np.max(out["mean_h"])),
                            float(out["mean_s"][-1]),
                        ]
                    )
                    if idx % 10 == 0:
                        print(f"Progress: {idx}/{total}")

    print(f"Sweep completed. Wrote {out_path}")


if __name__ == "__main__":
    main()
