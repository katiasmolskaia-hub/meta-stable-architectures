# Pilot Run Report (stress test)

Run date: 2026-03-13

Scenario:
- Stress test with increasing noise scale.
- sigma levels: 0.6, 0.7, 0.8 (noise_scale = sigma / 0.6).

Results:
- sigma=0.60: baseline 0.675/0.325/1.164; metaslayer_simple 0.892/0.108/1.173; metaslayer_real_I 1.000/0.000/1.193; metaslayer_soft 1.000/0.000/1.100
- sigma=0.70: baseline 0.650/0.350/1.180; metaslayer_simple 0.925/0.075/1.187; metaslayer_real_I 1.000/0.000/1.195; metaslayer_soft 1.000/0.000/1.100
- sigma=0.80: baseline 0.592/0.408/1.195; metaslayer_simple 0.925/0.075/1.195; metaslayer_real_I 1.000/0.000/1.195; metaslayer_soft 1.000/0.000/1.100

Notes:
- Baseline degrades as noise increases.
- Both metaslayer_real_I and metaslayer_soft remain stable across this range.
- Consider multi-seed stress test if publishing this as a primary result.
