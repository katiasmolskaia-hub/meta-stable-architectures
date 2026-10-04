import unittest

import numpy as np

from experiment_support_pulse import MessageBudget, SHARES, simulate
from experiment_local_support_signal import Params, simulate as previous


class PulseTests(unittest.TestCase):
    def test_shared_budget_bounds_any_time_window(self):
        # Adversarial: try both channels as often as permitted, with variable fees.
        for share in (0., .25, .5, .75, 1.):
            b = MessageBudget(1, 4., share)
            spends = []
            for step in range(401):
                if step:
                    b.refill(.2)
                spent = 0
                for channel in ("signal", "pulse"):
                    if b.ready(channel)[0]:
                        fee = 24 if step % 3 else 32
                        b.charge(0, channel, fee)
                        spent += fee
                spends.append(spent)
                self.assertLessEqual(b.spent[0], 32.+4.*step*.2+1e-8)
                self.assertLessEqual(b.signal_spent[0], (1.-share)*(32.+4.*step*.2)+1e-8)
            totals = np.concatenate(([0], np.cumsum(spends)))
            for start in range(len(spends)):
                for end in range(start, len(spends), 13):
                    self.assertLessEqual(totals[end+1]-totals[start], 32.+4.*(end-start)*.2+1e-8)

    def test_failed_request_still_costs_messages(self):
        b = MessageBudget(1, 1., .5)
        b.charge(0, "pulse", 24)
        self.assertEqual(b.common[0], 8.)
        b.refill(23.)
        self.assertFalse(b.ready("pulse")[0])
        b.refill(1.)
        self.assertTrue(b.ready("pulse")[0])

    def test_fixed_matches_previous_physics(self):
        for scenario in ("repeated_damage", "scarcity"):
            old, _ = previous("fixed", scenario, 11901, Params(n=96), keep_log=False)
            new = simulate("fixed", scenario, 11901, Params(n=96), rate=1.)
            self.assertEqual(old["service"], new["service"])
            self.assertEqual(old["p10"], new["p10"])
            self.assertEqual(new["messages"], 0)

    def test_end_to_end_common_cap_and_channel_controls(self):
        rows = {p: simulate(p, "repeated_damage", 11902, Params(n=96), rate=1.) for p in SHARES}
        for policy, row in rows.items():
            self.assertEqual(row["messages"], 24*row["requests"]+8*row["swaps"])
            self.assertLessEqual(row["messages"], row["allowed_message_envelope"])
            self.assertEqual(row["max_message_budget_excess"], 0.)
            self.assertEqual(row["active_edges"], 192)
            self.assertLess(row["max_mass_error"], 1e-8)
            self.assertLessEqual(row["max_flow_ratio"], 1.+1e-10)
        self.assertEqual(rows["pulse_only"]["signal_requests"], 0)
        self.assertEqual(rows["signal_only"]["pulse_requests"], 0)
        self.assertGreater(rows["hybrid_50"]["pulse_requests"], 0)
        self.assertGreater(rows["hybrid_50"]["signal_requests"], 0)


if __name__ == "__main__":
    unittest.main()
