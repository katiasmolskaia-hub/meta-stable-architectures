import unittest
from dataclasses import replace

import numpy as np

from experiment_local_support_signal import Params, make_world, flow_step, request_help, update_need, run_group


class LocalSignalTests(unittest.TestCase):
    def test_sparse_flow_matches_independent_dense_equations(self):
        p = Params(n=24)
        w = make_world(991, "persistent_damage", p)
        initial = np.random.default_rng(992).uniform(0, 2, p.n)
        est = np.full(6*p.n, p.prior)
        quality = w["qualities"][0]
        sparse, served, loss, error, ratio = flow_step(initial.copy(), w["active"], est, w, quality, p)
        adj = np.zeros((p.n, p.n))
        q = np.zeros((p.n, p.n))
        for e, (a, b) in enumerate(zip(w["u"], w["v"])):
            adj[a, b] = adj[b, a] = w["active"][e]
            q[a, b] = q[b, a] = quality[e]
        stock = np.minimum(initial + w["supply"]*p.dt, p.storage)
        stock *= np.exp(-p.leakage*p.dt)
        flow = p.conductance*np.maximum(stock[:, None]-stock[None, :], 0)*adj*p.dt
        flow *= np.minimum(1., np.minimum(stock, p.per_node_budget*p.dt)/np.maximum(flow.sum(axis=1), 1e-15))[:, None]
        received = flow*q
        stock = np.minimum(stock+received.sum(axis=0)-flow.sum(axis=1), p.storage)
        consume = np.minimum(stock, w["demand"]*p.dt)
        stock -= consume
        np.testing.assert_allclose(sparse, stock, atol=1e-12)
        np.testing.assert_allclose(served, consume, atol=1e-12)
        self.assertAlmostEqual(loss, float((flow-received).sum()), places=12)
        self.assertLess(error, 1e-8)
        self.assertLessEqual(ratio, 1+1e-10)

    def test_only_actual_contacts_update_estimates(self):
        p = Params(n=24)
        w = make_world(993, "persistent_damage", p)
        estimates = np.full(6*p.n, p.prior)
        flow_step(np.full(p.n, .7), w["active"], estimates, w, w["qualities"][0], p)
        np.testing.assert_array_equal(estimates[~w["active"]], np.full((~w["active"]).sum(), p.prior))

    def test_local_signal_on_off_and_restart(self):
        p = Params(n=24)
        need = np.zeros(p.n, bool)
        below = np.zeros(p.n, int)
        above = np.zeros(p.n, int)
        consumers = np.ones(p.n, bool)
        consumers[0] = False
        for _ in range(9):
            update_need(need, below, above, np.full(p.n, .9), consumers, p)
            self.assertFalse(need.any())
        update_need(need, below, above, np.full(p.n, .9), consumers, p)
        self.assertEqual(need.sum(), 23)
        for _ in range(25):
            update_need(need, below, above, np.ones(p.n), consumers, p)
        self.assertFalse(need.any())
        for _ in range(10):
            update_need(need, below, above, np.full(p.n, .9), consumers, p)
        self.assertEqual(need.sum(), 23)

    def test_swaps_keep_locality_degree_and_budget(self):
        p = Params(n=96)
        w = make_world(994, "uniform", p)
        est = np.full(6*p.n, p.prior)
        stock = np.random.default_rng(995).uniform(.3, 2., p.n)
        original = stock.sum()
        total = 0
        for step in range(300):
            messages, swaps, fee = request_help(step % p.n, stock, est, w, 994, step, p)
            total += swaps
            self.assertEqual(messages, 24 + 8*swaps)
            self.assertAlmostEqual(fee, p.swap_cost*swaps)
        self.assertGreater(total, 0)
        self.assertAlmostEqual(stock.sum(), original-total*p.swap_cost, places=9)
        self.assertEqual(int(w["active"].sum()), 2*p.n)
        for a, neighbors in enumerate(w["adjacency"]):
            self.assertEqual(len(neighbors), 4)
            self.assertTrue(neighbors <= w["discovery"][a].keys())
        self.assertEqual(len(w["u"]), 6*p.n)

    def test_paired_request_control_and_end_to_end_balance(self):
        rows = run_group((96, "repeated_damage", 996, .2))
        by_policy = {r["policy"]:r for r in rows}
        self.assertEqual(by_policy["local_signal"]["requests"], by_policy["shuffled_signal"]["requests"])
        for r in rows:
            self.assertLess(r["max_mass_error"], 1e-8)
            self.assertEqual(r["active_edges"], 192)
            self.assertEqual(r["messages"], 24*r["requests"] + 8*r["swaps"])
            self.assertLessEqual(r["requests"], 20*72)


if __name__ == "__main__":
    unittest.main()
