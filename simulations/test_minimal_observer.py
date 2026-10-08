import unittest
import numpy as np
import experiment_local_support_signal as base
from experiment_minimal_observer import Observer, diagnostics, forcing, simulate


class ObserverTests(unittest.TestCase):
    def test_recovering_shortage_is_not_treated_as_stalled(self):
        o = Observer(2, .2)
        for _ in range(20):
            o.update(np.array([0., 1.]))
        np.testing.assert_array_equal(o.flags('two_signals'), [True, False])
        for _ in range(10):
            o.update(np.ones(2))
        self.assertTrue(o.flags('persistent')[0])
        self.assertFalse(o.flags('two_signals')[0])

    def test_full_history_required_and_no_future_input(self):
        o = Observer(1, .2)
        for _ in range(19):
            o.update(np.zeros(1))
        self.assertFalse(o.flags('two_signals')[0])
        o.update(np.zeros(1))
        self.assertTrue(o.flags('two_signals')[0])

    def test_forcing_does_not_modify_reference_world_and_mixes_regions(self):
        p = base.Params(n=24)
        w = base.make_world(21900, 'uniform', p)
        s, d = w['supply'].copy(), w['demand'].copy()
        q, s1, d1 = forcing(60, 'mixed', w, p, s, d)
        np.testing.assert_array_equal(s, w['supply'])
        np.testing.assert_array_equal(d, w['demand'])
        np.testing.assert_array_equal(s1[:16], s[:16])
        np.testing.assert_allclose(s1[16:], .45 * s[16:])
        np.testing.assert_allclose(d1[8:16], 1.8 * d[8:16])
        self.assertTrue((q <= p.good_quality).all())
        self.assertTrue((q >= p.bad_quality - 1e-12).all())

    def test_replay_preserves_request_count_and_resource_constraints(self):
        p = base.Params(n=24, horizon=120., dt=.2, swap_cost=.12)
        a, events, _, _ = simulate('two_signals', 'mixed', 21901, p)
        b, replay, _, _ = simulate('shuffled', 'mixed', 21901, p, replay=events)
        self.assertEqual(a['requests'], b['requests'])
        self.assertEqual([s for s, _ in events], [s for s, _ in replay])
        for r in (a, b):
            self.assertLess(r['max_mass_error'], 1e-8)
            self.assertLessEqual(r['max_flow_ratio'], 1 + 1e-9)
            self.assertEqual(r['messages'], 24 * r['requests'] + 8 * r['swaps'])

    def test_diagnostics_keep_predictor_separate_from_future_label(self):
        p = base.Params(n=24, horizon=120., dt=.2)
        service = np.ones((600, 2))
        flags = {'test': np.tile([True, False], (600, 1))}
        row = diagnostics(service, flags, p)[0]
        self.assertEqual(row['tp'], 0)
        self.assertGreater(row['fp'], 0)
        service[:, 0] = 0
        other = diagnostics(service, flags, p)[0]
        self.assertEqual(other['fp'], 0)
        self.assertEqual(other['tp'], row['fp'])


if __name__ == '__main__':
    unittest.main()
