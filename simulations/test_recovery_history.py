import unittest
import numpy as np
from experiment_recovery_history import ages, choose, fit, generate, predict, reset_future, shuffle_history


class HistoryTests(unittest.TestCase):
    def test_shuffle_preserves_failure_age_boundary_and_counts(self):
        h, _ = generate(200, "stable_types", 201)
        s = shuffle_history(h, 202)
        np.testing.assert_array_equal(ages(h), ages(s))
        np.testing.assert_array_equal(h.sum(axis=1), s.sum(axis=1))
        self.assertFalse(np.array_equal(h, s))
        self.assertFalse(h[:, -1].any())

    def test_predictor_does_not_observe_evaluation_future(self):
        h, f = generate(200, "finite_duration", 203)
        model = fit(h[:150], f[:150])
        for policy in ("last_contact", "ema", "failure_age", "route_history"):
            before = predict(model, h[150:], policy)
            f[150:] ^= True
            np.testing.assert_array_equal(before, predict(model, h[150:], policy))
            self.assertTrue(((before >= 0) & (before <= 1)).all())
        np.testing.assert_array_equal(predict(model, h[150:], "last_contact"), np.full(50, model["mean"]))

    def test_age_edges_and_shared_ties(self):
        h = np.ones((3, 64), dtype=bool)
        h[0, -1:] = False
        h[1, -7:] = False
        h[2, :] = False
        np.testing.assert_array_equal(ages(h), [1, 7, 64])
        self.assertEqual(choose(np.ones(4), np.array([2, 1, 3, 0])), 2)

    def test_reset_is_independent_and_generation_reproducible(self):
        np.testing.assert_array_equal(reset_future(30, 204), reset_future(30, 204))
        for x, y in zip(generate(30, "stable_types", 205), generate(30, "stable_types", 205)):
            np.testing.assert_array_equal(x, y)

    def test_renewal_complete_runs_obey_duration_bounds(self):
        for law in ("finite_duration", "stable_types"):
            h, f = generate(100, law, 206)
            for row in np.concatenate((h, f), axis=1):
                boundaries = np.flatnonzero(row[1:] != row[:-1]) + 1
                lengths = np.diff(boundaries)
                self.assertGreater(len(lengths), 0)
                if law == "finite_duration":
                    self.assertTrue(((lengths >= 4) & (lengths <= 20)).all())
                else:
                    self.assertTrue(((lengths >= 4) & (lengths <= 8)).all() or
                                    ((lengths >= 16) & (lengths <= 28)).all())


if __name__ == "__main__":
    unittest.main()
