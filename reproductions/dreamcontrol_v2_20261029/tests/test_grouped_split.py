import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grouped_split import assign_splits, audit_assignments


class GroupedSplitTest(unittest.TestCase):
    def test_deterministic_independent_of_order_and_preserves_groups(self):
        records = [{'sample_id': f'{i}-{j}', 'group_id': f'source-{i}'}
                   for i in range(100) for j in range(3)]
        a = assign_splits(records, seed='v1')
        b = assign_splits(list(reversed(records)), seed='v1')
        self.assertEqual({x['sample_id']: x['split'] for x in a},
                         {x['sample_id']: x['split'] for x in b})
        audit_assignments(a)
        self.assertEqual({x['split'] for x in a}, {'train', 'val', 'test'})

    def test_rejects_missing_provenance_and_split_leakage(self):
        with self.assertRaises(ValueError):
            assign_splits([{'sample_id': 'one'}], seed='v1')
        with self.assertRaises(ValueError):
            assign_splits([{'sample_id': 'x', 'group_id': 'g'}] * 2, seed='v1')
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            audit_assignments([{'sample_id': 'x', 'group_id': 'g', 'split': 'train'},
                               {'sample_id': 'y', 'group_id': 'g', 'split': 'test'}])


if __name__ == '__main__':
    unittest.main()
