# Long Variable-Noise Pilot (2026-03-14)

**Setup**
- Long trajectory with variable stress windows and variable noise
- Models: baseline vs metaslayer_v1

**Summary (avg over full run)**
| model | success | failure | avg_time | avg_E | avg_E_eff |
|---|---:|---:|---:|---:|---:|
| baseline | 0.528 | 0.472 | 1.283 | 1.336 | 1.336 |
| metaslayer_v1 | 1.000 | 0.000 | 1.198 | 2.125 | 0.810 |

**Key takeaways**
- Metaslayer achieves perfect success on this long variable-noise run.
- Raw energy cost is higher, but effective energy **drops**, showing learning-driven efficiency.
- Average time improves vs baseline.
