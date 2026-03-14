# Thermal Noise Experiments

This folder contains experiments that add Langevin noise directly to the core variable a(t).

## Files

- simulation_noise.py: Langevin variant of the deterministic core (noise on a).
- experiment_phase_map.py: parameter scan over sigma_a and eta with saved heatmaps.
- experiment_hysteresis.py: sigma schedule (high -> low) to test hysteresis and recovery time.

## Run

```bash
python experiments/thermal_noise/simulation_noise.py
```

```bash
python experiments/thermal_noise/experiment_phase_map.py
```

```bash
python experiments/thermal_noise/experiment_hysteresis.py
```

Outputs are written to outputs/experiments/YYYYMMDD_HHMMSS/ and include a RUN_NOTES.txt file
with the exact grid and settings used.

## Experiment log

| Run ID | Purpose | sigma_a grid | eta grid |
| --- | --- | --- | --- |
| 20260311_103118 | Focused grid around baseline/stress | [0.0, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30] | [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] |
| 20260311_110551 | Extended noise range | [0.0, 0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50, 0.60] | [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] |
| 20260311_113713_noiseGrid_refine | Dense scan near transition | [0.45, 0.50, 0.55, 0.60, 0.65] | [0.70, 0.75, 0.80, 0.85, 0.90, 0.95] |
| 20260311_124621_hysteresis_sigma_0.6_to_0.2 | Hysteresis: sigma 0.6 -> 0.2 | schedule | [0.70, 0.75, 0.80] |
| 20260311_162640_noiseGrid_boundary | Boundary scan | [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66] | [0.75, 0.80, 0.85, 0.90, 0.95] |

Add new runs here as they are generated.
