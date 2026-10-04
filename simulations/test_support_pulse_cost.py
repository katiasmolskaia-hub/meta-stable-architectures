import unittest
from dataclasses import replace

import numpy as np

import experiment_local_support_signal as base
import experiment_support_pulse as pulse
from experiment_support_pulse_cost import run_group


class PulseCostTests(unittest.TestCase):
    def test_unaffordable_swap_does_not_spend_or_change_graph(self):
        p = base.Params(n=24, swap_cost=.48)
        w = base.make_world(12901, "uniform", p)
        stock = np.full(p.n, .7)
        stock[0] = 0.
        estimates = np.full(len(w["u"]), p.prior)
        initial_stock, initial_edges = stock.copy(), w["active"].copy()
        msg, swaps, fee = base.request_help(0, stock, estimates, w, 12901, 1, p)
        self.assertEqual((msg, swaps, fee), (24, 0, 0.))
        np.testing.assert_array_equal(stock, initial_stock)
        np.testing.assert_array_equal(w["active"], initial_edges)
        # The same configuration can afford a cheap swap; refusal above is
        # actually due to the fee, not an impossible local geometry.
        msg, swaps, fee = base.request_help(0, stock, estimates, w, 12901, 1, replace(p, swap_cost=.024))
        self.assertEqual((msg, swaps, fee), (32, 1, .024))

    def test_full_fee_is_paid_once_and_reserve_preserved(self):
        for factor in (1, 5, 20):
            p = base.Params(n=96, swap_cost=.024*factor)
            w = base.make_world(12902, "repeated_damage", p)
            stock = np.random.default_rng(12903).uniform(.3, 2., p.n)
            estimates = np.full(len(w["u"]), p.prior)
            total = 0
            for step in range(120):
                before = stock.copy()
                msg, swaps, fee = base.request_help(step % p.n, stock, estimates, w, 12902, step, p)
                self.assertAlmostEqual(before.sum()-stock.sum(), fee, places=10)
                changed = np.flatnonzero(np.abs(before-stock) > 1e-12)
                self.assertEqual(len(changed), swaps)
                if swaps:
                    self.assertAlmostEqual(fee, p.swap_cost)
                    self.assertGreaterEqual(stock[changed[0]], p.helper_reserve-1e-12)
                total += swaps
            self.assertGreater(total, 0)

    def test_price_zero_control_and_saved_cost_accounting(self):
        rows = run_group((96, "repeated_damage", 12904, .2))
        self.assertEqual(len(rows), 7)
        old = pulse.simulate("fixed", "repeated_damage", 12904, base.Params(n=96), 4.)
        self.assertEqual(rows[0]["service"], old["service"])
        for row in rows:
            self.assertAlmostEqual(row["resource_cost"], row["swaps"]*row["swap_cost"])
            self.assertAlmostEqual(row["cost_fraction"], row["resource_cost"]/row["gross_production"])
            self.assertEqual(row["messages"], 24*row["requests"]+8*row["swaps"])
            self.assertEqual(row["max_message_budget_excess"], 0.)
            self.assertLess(row["max_mass_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
