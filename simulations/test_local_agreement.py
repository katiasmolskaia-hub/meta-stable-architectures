import unittest

from experiment_local_agreement import Bench, PROFILES


class AgreementTests(unittest.TestCase):
    def test_clean_handshake_has_expected_message_rounds(self):
        b = Bench("confirmed_retry", dict(loss=0., max_delay=1, duplicate=0.), 13900)
        r = b.run()
        self.assertEqual(r["messages"], 12)
        self.assertEqual(r["completion_time"], 3)
        self.assertEqual(r["agreement_time"], 4)
        self.assertEqual(r["mixed_view_ticks"], 1)
        self.assertEqual(r["fee_charges"], 1)

    def test_duplicates_do_not_repeat_fee_or_change(self):
        b = Bench("confirmed_retry", dict(loss=0., max_delay=1, duplicate=1.), 13901)
        r = b.run()
        self.assertEqual(r["complete"], 1)
        self.assertGreater(r["duplicates"], 0)
        self.assertEqual(b.applied, [1, 1, 1, 1])
        self.assertEqual(r["fee_charges"], 1)

    def test_abort_is_final_despite_late_prepare(self):
        b = Bench("confirmed_retry", PROFILES["temporary_partition"], 13902)
        b.enqueue(30, 0, 1, "PREPARE")
        r = b.run()
        self.assertEqual(r["decision"], "ABORT")
        self.assertEqual(r["prepared_end"], 0)
        self.assertEqual(r["complete"], 0)
        self.assertEqual(r["conflict_ticks"], 0)
        self.assertEqual(r["fee_charges"], 0)

    def test_permanent_lost_decision_blocks_instead_of_fake_rollback(self):
        r = Bench("confirmed_retry", PROFILES["final_never_arrives"], 13903).run()
        self.assertEqual(r["decision"], "COMMIT")
        self.assertEqual(r["prepared_end"], 1)
        self.assertEqual(r["partial_end"], 1)
        self.assertEqual(r["complete"], 0)
        self.assertEqual(r["conflict_ticks"], 0)

    def test_budget_and_partial_views_are_visible(self):
        r = Bench("command_once", PROFILES["final_never_arrives"], 13904).run()
        self.assertEqual(r["messages"], 3)
        self.assertEqual(r["partial_end"], 1)
        self.assertGreater(r["unavailable_changed_edge_node_ticks"], 0)
        r = Bench("confirmed_retry", PROFILES["loss_30"], 13905, limit=2).run()
        self.assertLessEqual(r["max_node_messages"], 2)
        self.assertGreater(r["budget_suppressed"], 0)


if __name__ == "__main__":
    unittest.main()
