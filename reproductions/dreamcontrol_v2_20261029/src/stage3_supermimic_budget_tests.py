"""Synthetic CPU-only collector/restore checks; no physics or policy training."""
import json
import unittest

from stage3_supermimic_budget import RealInteractionBudget


class BudgetChecks(unittest.TestCase):
    def test_native_collection_with_warmup_stops_exactly(self):
        budget = RealInteractionBudget(8912, 3000, 24)
        actual_collected = 64
        budget.record_after_step(64, actual_num_envs=8912, warmup=True)
        acc = 0.
        capped_last = None
        updates = 0
        while budget.remaining_vector_steps:
            acc += 1/.4
            requested = int(acc)
            acc -= requested
            accepted = budget.cap_collection(requested)
            for _ in range(accepted):
                # A synthetic collector step; the real adapter must call its simulator here.
                actual_collected += 1
                budget.record_after_step(actual_num_envs=8912)
            capped_last = (requested, accepted)
            updates += 1
        self.assertEqual(actual_collected, 72000)
        self.assertEqual(updates, 28775)
        self.assertEqual(capped_last, (2, 1))
        self.assertEqual(budget.real_transitions, 641664000)
        self.assertEqual(budget.progress()['completed_macro_rounds'], 3000)
        self.assertEqual(budget.cap_collection(5), 0)

    def test_resume_mid_round_has_no_double_counting(self):
        budget = RealInteractionBudget(8912, 3000, 24)
        budget.record_after_step(64, actual_num_envs=8912, warmup=True)
        budget.record_after_step(137, actual_num_envs=8912)
        restored = RealInteractionBudget.restore(json.loads(json.dumps(budget.state_dict())),
            expected_num_envs=8912, expected_macro_rounds=3000, expected_steps_per_round=24)
        self.assertEqual(restored.progress(), budget.progress())
        restored.record_after_step(7, actual_num_envs=8912)
        self.assertEqual(restored.vector_steps, 208)
        self.assertEqual(restored.warmup_vector_steps, 64)
        self.assertEqual(restored.progress()['steps_into_next_macro_round'], 16)

    def test_wrong_environment_and_overrun_rejected_without_mutation(self):
        budget = RealInteractionBudget(4, 1, 2)
        original = budget.state_dict()
        for args in [(3, 4), (1, 3), (1, True), (1, 4.0)]:
            with self.assertRaises(ValueError):
                budget.record_after_step(args[0], actual_num_envs=args[1])
            self.assertEqual(budget.state_dict(), original)

    def test_restore_rejects_changed_target_and_corrupted_counts(self):
        state = RealInteractionBudget(8912, 3000, 24).state_dict()
        with self.assertRaises(ValueError):
            RealInteractionBudget.restore(state, expected_num_envs=8912,
                expected_macro_rounds=5000, expected_steps_per_round=24)
        state['warmup_vector_steps'] = 64
        with self.assertRaises(ValueError):
            RealInteractionBudget.restore(state, expected_num_envs=8912,
                expected_macro_rounds=3000, expected_steps_per_round=24)

    def test_invalid_configuration_and_request(self):
        for args in [(True, 1, 1), (4, 0, 1), (4, 1, 1.5)]:
            with self.assertRaises(ValueError):
                RealInteractionBudget(*args)
        budget = RealInteractionBudget(4, 1, 2)
        with self.assertRaises(ValueError):
            budget.cap_collection(-1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
