import unittest
from dataclasses import replace

from experiment_bridge_buffer import Params, POLICIES, simulate


class BufferTests(unittest.TestCase):
    def test_balance_budget_storage(self):
        for policy in POLICIES:
            m, _ = simulate(policy, "bursty_supply", 27)
            self.assertLess(m["max_mass_error"], 1e-9)
            self.assertLessEqual(m["max_budget_ratio"], 1 + 1e-9)
            self.assertTrue(0 <= m["service_fraction"] <= 1 + 1e-9)

    def test_no_transport_same_service_and_stress(self):
        rows = [simulate(policy, "steady", 27, replace(Params(), budget=0))[0] for policy in POLICIES]
        for key in ("served", "stress_area", "rejected"):
            self.assertEqual(len({r[key] for r in rows}), 1)

    def test_ablation_exactly_matches_ordinary_buffer(self):
        p = replace(Params(), permission_gain=0)
        for scenario in ("steady", "upstream_windows", "delayed_sensor"):
            a, _ = simulate("adaptive_bridge", scenario, 27, p)
            b, _ = simulate("ordinary_buffer", scenario, 27, p)
            for key in ("served", "stress_area", "rejected", "moved", "cost"):
                self.assertEqual(a[key], b[key])

    def test_closed_downstream_never_delivers(self):
        for policy in POLICIES:
            m, trace = simulate(policy, "global_outage", 27, keep_trace=True)
            self.assertEqual(m["downstream_closed_delivery"], 0)
            self.assertTrue(all(r["release_rate"] == 0 for r in trace if not r["downstream_open"]))

    def test_resource_cannot_cross_empty_buffer_in_same_step(self):
        _, trace = simulate("ordinary_buffer", "steady", 27, keep_trace=True)
        self.assertEqual(trace[0]["release_rate"], 0)
        self.assertGreater(trace[0]["fill_rate"], 0)

    def test_reproducible(self):
        a, at = simulate("adaptive_bridge", "delayed_sensor", 27, keep_trace=True)
        b, bt = simulate("adaptive_bridge", "delayed_sensor", 27, keep_trace=True)
        self.assertEqual(a, b)
        self.assertEqual(at, bt)

    def test_transport_changes_receiver_state(self):
        a, _ = simulate("rate_limiter", "steady", 27)
        b, _ = simulate("rate_limiter", "steady", 27, replace(Params(), budget=0))
        self.assertGreater(a["served"], b["served"])
        self.assertLess(a["stress_area"], b["stress_area"])


if __name__ == "__main__":
    unittest.main()
