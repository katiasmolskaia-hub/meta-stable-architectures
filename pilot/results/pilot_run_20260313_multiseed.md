# Pilot Run Report (multi-seed)

Run date: 2026-03-13

Scenario:
- 10 seeds, 120 tasks each
- Compare baseline vs metaslayer_v0 vs metaslayer_v1 (real I) vs metaslayer_soft

Summary (mean ? std):
- baseline: success=0.6825?0.0299, failure=0.3175?0.0299, avg_time=1.1572?0.0070
- metaslayer_v0: success=0.9050?0.0085, failure=0.0950?0.0085, avg_time=1.1641?0.0123
- metaslayer_v1: success=0.9983?0.0033, failure=0.0017?0.0033, avg_time=1.1915?0.0043
- metaslayer_soft: success=0.9892?0.0084, failure=0.0108?0.0084, avg_time=1.0992?0.0009

Notes:
- metaslayer_soft trades a small failure rate for the best speed.
- metaslayer_v1 minimizes failures with a modest time cost.
