import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from bond_recovery_model import (
    RecoveryDetector, advance, fingerprint, initialize, parameters, quality_at, run_episode, run_phase,
)
from bond_rewiring_scaled_model import simulate as reference


class RecoveryTests(unittest.TestCase):
    def test_recovery_requires_observed_failure_and_full_window(self):
        d = RecoveryDetector(.2)
        for _ in range(60):
            self.assertFalse(d.observe(1.))
        self.assertFalse(d.observe(.94))
        for _ in range(49):
            self.assertFalse(d.observe(.95))
        self.assertTrue(d.observe(.95))
        self.assertFalse(d.observe(.94))
        self.assertEqual(d.stable_steps, 0)
        for _ in range(60):
            self.assertFalse(d.observe(1., eligible=False))

    def test_refactor_reproduces_old_model(self):
        for n, cost in ((24, 1), (96, 20)):
            for policy in ("recent_contact", "experience_swap"):
                p = parameters(n, cost)
                old, _, _, old_estimates = reference(policy, "persistent_damage", 9011, p)
                state, world = initialize(9011, "persistent_damage", p)
                served = loss = 0.
                for step in range(round(p.horizon / p.dt)):
                    r = advance(state, world, policy, p, step,
                                quality_at(step * p.dt, "persistent_damage", world.qualities, p))
                    if step * p.dt >= p.crisis_start:
                        served += r["consumption"].sum()
                        loss += r["loss"]
                self.assertAlmostEqual(served / (world.demand.sum() * 80), old["service_fraction"], places=12)
                self.assertAlmostEqual(loss, old["lost"], places=12)
                self.assertEqual(state.swaps, old["swaps"])
                self.assertEqual(state.cost, old["rewiring_cost"])
                np.testing.assert_array_equal(state.estimates, old_estimates)

    def test_pause_keeps_graph_and_copy_independent(self):
        p = parameters(24, 20)
        state, world = initialize(9012, "persistent_damage", p)
        frozen = fingerprint(state)
        branch = copy.deepcopy(state)
        fork_adj = state.adj.copy()
        r, _ = run_phase(branch, world, "recent_contact", "persistent_damage", p, 0, 30.,
                         False, False, fork_adj, "pause", False)
        self.assertEqual(r["swaps"], 0)
        self.assertEqual(r["cost"], 0)
        np.testing.assert_array_equal(branch.adj, fork_adj)
        self.assertEqual(fingerprint(state), frozen)
        self.assertLess(branch.max_mass_error, 1e-8)

    def test_paired_fork_costs_and_post_shock_opportunities(self):
        # A deliberately severe temporary failure guarantees an actual crisis;
        # ordinary random damage may never break service and must not count.
        p = replace(parameters(24, 1), bad_share=1.)
        with patch("bond_recovery_model.parameters", return_value=p):
            episode, rows, _ = run_episode(24, 1, "temporary_damage", "experience_swap", 9013)
        self.assertEqual(episode["status"], "recovered")
        self.assertGreaterEqual(episode["recovery_time"], 80.)
        self.assertEqual(len(rows), 2)
        a, b = rows
        self.assertEqual(a["quiet_swaps"], 15)
        self.assertEqual(b["quiet_swaps"], 0)
        self.assertEqual(a["second_swaps"], b["second_swaps"])
        self.assertEqual(a["second_cost"], b["second_cost"])
        self.assertAlmostEqual(a["second_shock_time"], episode["recovery_time"] + 30)
        self.assertEqual(b["quiet_fork_edge_share"], 1.)
        for r in rows:
            self.assertLess(r["max_mass_error"], 1e-8)
            self.assertLessEqual(r["max_budget_ratio"], 1 + 1e-10)

    def test_unaffected_is_not_called_recovered(self):
        episode, rows, _ = run_episode(24, 1, "temporary_damage", "experience_swap", 9013)
        self.assertEqual(episode["status"], "unaffected")
        self.assertIsNone(episode["recovery_time"])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
