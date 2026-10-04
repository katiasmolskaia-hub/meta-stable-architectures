"""Physical/experimental invariants, not assertions that a policy must win."""
import unittest
from dataclasses import replace

import numpy as np

from experiment_bridge_feedback import Params, POLICIES, simulate, recovery_delay


class BridgeFeedbackTests(unittest.TestCase):
    def test_resource_and_budget_invariants(self):
        for policy in POLICIES:
            m, _ = simulate(policy, "persistent_damage", 7)
            self.assertLess(m["max_mass_error"], 1e-9)
            self.assertLessEqual(m["max_budget_ratio"], 1 + 1e-10)
            self.assertGreaterEqual(m["service_fraction"], 0)
            self.assertLessEqual(m["service_fraction"], 1 + 1e-10)

    def test_no_transport_removes_policy_effect(self):
        results = [simulate(policy, "persistent_damage", 7, replace(Params(), budget=0))[0]
                   for policy in POLICIES]
        for key in ("served", "lost", "stress_area", "resource_cost", "isolation_time"):
            self.assertEqual(len({row[key] for row in results}), 1)

    def test_reproducible(self):
        a, at = simulate("lifecycle", "temporary_damage", 9, keep_trace=True)
        b, bt = simulate("lifecycle", "temporary_damage", 9, keep_trace=True)
        self.assertEqual(at, bt)
        self.assertEqual(a["served"], b["served"])

    def test_memory_ablation_starts_with_identical_actions(self):
        _, a = simulate("lifecycle", "persistent_damage", 7, keep_trace=True)
        _, b = simulate("no_memory", "persistent_damage", 7, keep_trace=True)
        for key in ("flow_AB", "flow_AC", "flow_BC", "stress_A", "stress_B", "stress_C"):
            self.assertEqual(a[0][key], b[0][key])

    def test_bridges_change_group_dynamics(self):
        _, a = simulate("old_repair", "persistent_damage", 7, keep_trace=True)
        _, b = simulate("lifecycle", "persistent_damage", 7, keep_trace=True)
        self.assertGreater(max(abs(x["stress_B"] - y["stress_B"]) for x, y in zip(a, b)), 0.001)

    def test_recovery_requires_full_window_and_keeps_failures(self):
        p = replace(Params(), horizon=12, crisis_end=2, dt=1, recovery_hold=3)
        times = np.arange(12)
        service = np.zeros(12)
        service[3:5] = 1
        self.assertTrue(np.isnan(recovery_delay(times, service, np.zeros(12), p)))
        service[7:10] = 1
        self.assertEqual(recovery_delay(times, service, np.zeros(12), p), 5)


if __name__ == "__main__":
    unittest.main()
