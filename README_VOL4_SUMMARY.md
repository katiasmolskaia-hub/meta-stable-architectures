# Meta-Stable Architectures (Vol IV) — Quick Summary

**What this is:**
A soft coordination layer (QRC) for networked, metastable systems. It preserves coherence and enables recovery from absorbing isolation without hard control.

**Key idea:**
Instead of forcing agents, QRC **aligns phase information** and uses **coherence-driven relaxation** to let the system exit isolation naturally.

## What we tested
- **Baseline vs QRC:** baseline gets stuck; QRC recovers.
- **Stress duration:** recovery time grows with duration, but remains finite.
- **Topology:** ring / ER / small-world / scale-free; small-world is slightly better.
- **Noise:** higher noise increases dispersion, recovery remains finite.
- **Heterogeneity:** per-agent noise/thresholds/alpha/gamma; QRC still recovers.
- **Experience:** repeated stress episodes recover faster over time.

## Key results (short)
- QRC removes NaN recovery time at critical stress fractions (0.22–0.26).
- Recovery remains finite even at stress_frac = 1.0 and stress_duration up to 20.
- Heterogeneity increases tail delays but does not break recovery.

## How to run
```powershell
python simulations/simulation_network_v1.py
```

## Key figures
- `manuscripts/v3/figures/network_qrc_vs_baseline_plot.png`
- `manuscripts/v3/figures/qrc_stress_duration_plot.png`
- `manuscripts/v3/figures/qrc_noise_sweep_plot.png`
- `manuscripts/v3/figures/qrc_topology_compare_n100.png`

## Volume IV draft
`manuscripts/v3/volume_IV_v1.tex`

---
If you're new: start with the figures above and the summary table in the volume.
