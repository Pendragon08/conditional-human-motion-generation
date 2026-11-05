"""Meaningful CPU ring order, sequence, terminal-state and restore checks."""
import unittest
import torch
from stage3_supermimic_replay import RealReplay, FIELDS


def row(i, n=3):
    result = {}
    for k, (width, dtype) in FIELDS.items():
        result[k] = torch.full((n,)+(() if width is None else (width,)),
                               i if dtype != torch.bool else False, dtype=dtype)
    result['done'][0] = i == 11
    return result


class ReplayTests(unittest.TestCase):
    def test_wrapped_sequences_are_ordered(self):
        r = RealReplay(8, 3, 'cpu')
        for i in range(15): r.append(row(i))
        batch = r.sample(200, 8)
        self.assertTrue(torch.equal(batch['s'][:, :, 0], torch.arange(7., 15.).repeat(200, 1)))
        self.assertTrue(torch.equal(batch['s1'][:, :, 0], batch['s'][:, :, 0]))
        self.assertEqual(r.bytes, 8*3*1602)

    def test_exact_roundtrip_and_append(self):
        r, s = RealReplay(8, 3, 'cpu'), RealReplay(8, 3, 'cpu')
        for i in range(13): r.append(row(i))
        s.load_state_dict(r.state_dict())
        s.append(row(13)); r.append(row(13))
        for k in FIELDS: self.assertTrue(torch.equal(r.data[k], s.data[k]))
        self.assertEqual(r.position, s.position)

    def test_missing_invalid_and_oversized_samples(self):
        r = RealReplay(8, 3, 'cpu')
        with self.assertRaises(ValueError): r.sample(4, 1)
        x = row(0); del x['ref']
        with self.assertRaises(ValueError): r.append(x)
        x = row(0); x['s'][0, 1] = float('nan')
        with self.assertRaises(ValueError): r.append(x)
        self.assertEqual(r.total_rows, 0)

    def test_restored_history_corruption_rejected(self):
        r, s = RealReplay(8, 3, 'cpu'), RealReplay(8, 3, 'cpu')
        r.append(row(0)); state = r.state_dict(); state['position'] = 3
        with self.assertRaises(ValueError): s.load_state_dict(state)


if __name__ == '__main__': unittest.main()
