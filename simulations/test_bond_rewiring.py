import unittest
from dataclasses import replace

import numpy as np

from experiment_bond_rewiring import Params, POLICIES, simulate, initial_graph, assert_graph, candidate_swaps, apply_swap, graph_metrics


class RewiringTests(unittest.TestCase):
    def test_many_swaps_preserve_degree_and_edge_count(self):
        rng = np.random.default_rng(7)
        adj = initial_graph(24, rng.permutation(24))
        for i in range(200):
            proposal = candidate_swaps(adj, i % 24, rng, 12)[0]
            apply_swap(adj, proposal)
            assert_graph(adj)
            self.assertEqual(int(adj.sum()), 96)

    def test_balance_and_budget(self):
        for policy in POLICIES:
            r, _, _, _ = simulate(policy, "persistent_damage", 19)
            self.assertLess(r["max_mass_error"], 1e-8)
            self.assertLessEqual(r["max_budget_ratio"], 1 + 1e-10)
            self.assertTrue(0 <= r["service_fraction"] <= 1 + 1e-10)

    def test_equal_rewiring_cost_and_count(self):
        rows = [simulate(p, "noisy_feedback", 19)[0] for p in POLICIES[1:]]
        self.assertEqual({r["swaps"] for r in rows}, {59})
        self.assertEqual(len({r["rewiring_cost"] for r in rows}), 1)

    def test_no_contact_no_learning(self):
        p = replace(Params(), total_budget_per_node=0)
        rows = []
        for policy in POLICIES:
            r, _, _, estimates = simulate(policy, "persistent_damage", 19, p)
            np.testing.assert_array_equal(estimates, np.full((p.n, p.n), p.prior))
            rows.append(r)
        self.assertEqual(len({r["service_fraction"] for r in rows}), 1)

    def test_fixed_graph_and_repeatability(self):
        r, _, _, _ = simulate("fixed", "persistent_damage", 19)
        self.assertEqual(r["final_old_edge_share"], 1)
        self.assertEqual(r["swaps"], 0)
        a = simulate("experience_swap", "persistent_damage", 19, trace_enabled=True)
        b = simulate("experience_swap", "persistent_damage", 19, trace_enabled=True)
        self.assertEqual(a[:3], b[:3])
        np.testing.assert_array_equal(a[3], b[3])

    def test_components_detect_isolated_consumers(self):
        adj = np.zeros((10, 10), dtype=bool)
        for start in (0, 5):
            adj[start:start+5, start:start+5] = True
        np.fill_diagonal(adj, False)
        sources = np.zeros(10, dtype=bool)
        sources[0] = True
        largest, reach = graph_metrics(adj, sources)
        self.assertEqual(largest, .5)
        self.assertAlmostEqual(reach, 4/9)


if __name__ == "__main__":
    unittest.main()
