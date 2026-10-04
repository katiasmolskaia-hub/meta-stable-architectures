import unittest
from dataclasses import replace

import numpy as np

from experiment_bond_rewiring import simulate as original_simulate
from experiment_bond_rewiring_scale_cost import parameters, SIZES, COSTS, POLICIES
from bond_rewiring_scaled_model import simulate


class ScaleCostTests(unittest.TestCase):
    def test_high_fee_can_use_source_stock_at_smaller_step(self):
        for n in (24, 96):
            r = simulate("experience_swap", "persistent_damage", 6032,
                         parameters(n, 20, dt=.1))[0]
            self.assertEqual(r["swaps"], 59 * (n // 24))
            self.assertAlmostEqual(r["rewiring_cost"], 59 * (n // 24) * .48)
            self.assertLess(r["max_mass_error"], 1e-8)

    def test_reference_equivalence(self):
        for policy in POLICIES:
            old = original_simulate(policy, "persistent_damage", 6031, trace_enabled=True)
            new = simulate(policy, "persistent_damage", 6031, parameters(24, 1), trace_enabled=True)
            for key, value in old[0].items():
                self.assertAlmostEqual(value, new[0][key], places=11) if isinstance(value, float) else self.assertEqual(value, new[0][key])
            self.assertEqual(old[1:3], new[1:3])
            np.testing.assert_array_equal(old[3], new[3])

    def test_equal_per_node_events_and_cost_high_price_balance(self):
        for n in SIZES:
            for cost in COSTS:
                p = replace(parameters(n, cost), horizon=8, crisis_start=0)
                for policy in POLICIES:
                    r = simulate(policy, "persistent_damage", 6032, p)[0]
                    expected = 0 if policy == "fixed" else 3 * (n // 24)
                    self.assertEqual(r["swaps"], expected)
                    self.assertAlmostEqual(r["rewiring_cost"], expected * .024 * cost)
                    self.assertLess(r["max_mass_error"], 1e-8)
                    self.assertLessEqual(r["max_budget_ratio"], 1 + 1e-10)
                    self.assertTrue(0 <= r["service_fraction"] <= 1 + 1e-10)
                    if policy != "fixed":
                        self.assertAlmostEqual(r["swaps"] / n, 3 / 24)


if __name__ == "__main__":
    unittest.main()
