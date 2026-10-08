import copy
import unittest
import numpy as np
import experiment_repeated_help as model
import experiment_local_support_signal as base
import experiment_help_timing as timing


class RepeatedTests(unittest.TestCase):
    def state(self):
        p = base.Params(n=24, dt=.2, swap_cost=.12)
        state = model.initial_world(24900, p)
        supply, demand = state['world']['supply'].copy(), state['world']['demand'].copy()
        for k in range(200):
            timing.advance(state, k*p.dt, 'mixed', p, supply, demand)
        node = int(np.flatnonzero(~state['world']['sources'])[0])
        return state, p, supply, demand, node

    def test_recovery_censoring_and_last_stable_segment(self):
        self.assertEqual(model.recovery([1.]*10, 1.), (1, 0.))
        self.assertEqual(model.recovery([0.]*6+[1.]*4, 1.), (1, 6.))
        self.assertEqual(model.recovery([0.]*7+[1.]*3, 1.), (0, 10.))
        self.assertEqual(model.recovery([1.]*5+[0.]+[1.]*4, 1.), (1, 6.))
        self.assertEqual(model.recovery([0.]*10, 1.), (0, 10.))

    def test_schedules_accounting_and_resource_balance(self):
        state, p, supply, demand, node = self.state()
        for policy in model.POLICIES:
            r, ev, tr = model.branch(state, 'mixed', 24900, 40, node, policy, p, supply, demand)
            if policy == 'pause':
                self.assertEqual(ev, [])
            if policy.startswith('pulse'):
                gap = int(policy[5:])
                self.assertEqual([e['delay'] for e in ev], list(range(gap, 48, gap)))
            times = [0.] + [e['delay'] for e in ev]
            self.assertTrue(all(b-a >= 8-1e-9 for a,b in zip(times,times[1:])))
            if policy == 'persistent':
                self.assertTrue(all(e['need'] for e in ev))
            self.assertEqual(r['messages'], 24*r['attempts']+8*r['swaps'])
            self.assertAlmostEqual(r['fee'], .12*r['swaps'])
            self.assertAlmostEqual(tr[:,1].sum(), r['served_units'])
            self.assertLess(r['max_mass_error'], 1e-8)
            self.assertLessEqual(r['max_flow_ratio'], 1+1e-9)

    def test_independent_branches_and_no_early_effect(self):
        state, p, supply, demand, node = self.state()
        before = copy.deepcopy(state)
        a, _, ta = model.branch(state, 'mixed', 24900, 40, node, 'pause', p, supply, demand)
        _, _, tb = model.branch(state, 'mixed', 24900, 40, node, 'pulse8', p, supply, demand)
        c, _, tc = model.branch(state, 'mixed', 24900, 40, node, 'pause', p, supply, demand)
        self.assertEqual(a,c)
        np.testing.assert_array_equal(ta,tc)
        np.testing.assert_array_equal(ta[:40],tb[:40])
        np.testing.assert_array_equal(state['stock'],before['stock'])
        np.testing.assert_array_equal(state['observer'].buffer,before['observer'].buffer)
        self.assertEqual(state['world']['adjacency'],before['world']['adjacency'])


if __name__ == '__main__':
    unittest.main()
