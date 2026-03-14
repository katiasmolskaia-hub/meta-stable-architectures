# Simulations

Numerical experiments related to the Meta-Stable Architectures trilogy.

## New v2 assets

- `simulation_v2.py` - unified deterministic + stochastic simulation aligned with v2 manuscripts.
- `SIMULATION_V2_EXPLAIN_RU.md` - plain Russian explanation of variables, logic, and how to read outputs.
- `experiment_compare.py` - multi-scenario experiment (`baseline`, `stress_1`, `stress_2`) with saved CSV, PNG, and metrics table.

## Research framing (RU)

??? ? ??

## Quick start

```bash
python simulations/simulation_v2.py
```

## Scenario comparison with saved outputs

```bash
python simulations/experiment_compare.py
```

The script creates a run folder:

`outputs/simulations/YYYYMMDD_HHMMSS/`

and also refreshes a stable snapshot:

`outputs/simulations/latest/`

Saved files include deterministic/stochastic CSV per scenario, `comparison_metrics.csv`, `comparison_plot.png`, and `*_params_used.txt`.

## Thermal noise experiments

See `experiments/thermal_noise/` for Langevin noise on a(t) and phase map scans.
