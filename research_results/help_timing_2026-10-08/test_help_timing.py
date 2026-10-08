import copy
import unittest
import numpy as np
import experiment_local_support_signal as base
import experiment_minimal_observer as prior
from experiment_help_timing import advance, branch, snapshot


class TimingTests(unittest.TestCase):
    def initial(self):
        p = base.Params(n=24, dt=.2, horizon=120., swap_cost=.12)
        w = base.make_world(23900, 'uniform', p)
        state = dict(world=w, stock=np.full(p.n, p.initial_stock), estimates=np.full(len(w['u']), p.prior),
                     observer=prior.Observer(p.n, p.dt))
        s, d = w['supply'].copy(), w['demand'].copy()
        for k in range(200):
            advance(state, k*p.dt, 'mixed', p, s, d)
        node = int(np.flatnonzero(~w['sources'])[0])
        return state, p, s, d, node

    def test_branches_do_not_mutate_checkpoint_or_each_other(self):
        state, p, s, d, node = self.initial()
        original = copy.deepcopy(state)
        before = branch(state, 'mixed', 23900, 40, node, 'none', p, s, d)
        branch(state, 'mixed', 23900, 40, node, 'now', p, s, d)
        after = branch(state, 'mixed', 23900, 40, node, 'none', p, s, d)
        self.assertEqual(before, after)
        np.testing.assert_array_equal(state['stock'], original['stock'])
        np.testing.assert_array_equal(state['estimates'], original['estimates'])
        np.testing.assert_array_equal(state['observer'].buffer, original['observer'].buffer)
        self.assertEqual(state['world']['adjacency'], original['world']['adjacency'])

    def test_one_attempt_and_physical_delay_are_enforced(self):
        state, p, s, d, node = self.initial()
        for policy, delay in [('none', -1), ('now', 0), ('delay4', 4), ('delay8', 8)]:
            r = branch(state, 'mixed', 23900, 40, node, policy, p, s, d)
            self.assertEqual(r['attempt_delay'], delay)
            self.assertEqual(r['attempts'], int(policy != 'none'))
            self.assertEqual(r['messages'], 24*r['attempts'] + 8*r['swaps'])
            self.assertAlmostEqual(r['fee'], .12*r['swaps'])
            self.assertLess(r['max_mass_error'], 1e-8)

    def test_identical_now_and_trend_when_already_stalled(self):
        state, p, s, d, node = self.initial()
        o = state['observer']
        o.q[node], o.below[node], o.trend[node] = .3, 3., 0.
        a = branch(state, 'mixed', 23900, 40, node, 'now', p, s, d)
        b = branch(state, 'mixed', 23900, 40, node, 'trend_rule', p, s, d)
        del a['policy'], b['policy']
        self.assertEqual(a, b)

    def test_snapshot_copies_graph_and_observer(self):
        state, p, s, d, node = self.initial()
        snap = snapshot(state['world'], state['stock'], state['estimates'], state['observer'])
        snap['world']['adjacency'][node].clear()
        snap['observer'].q[node] = -99
        self.assertEqual(len(state['world']['adjacency'][node]), 4)
        self.assertGreaterEqual(state['observer'].q[node], 0)


if __name__ == '__main__':
    unittest.main()
