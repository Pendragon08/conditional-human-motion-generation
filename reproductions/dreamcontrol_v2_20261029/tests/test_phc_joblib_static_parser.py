"""Synthetic tests: no third-party pickle is executed or deserialized."""
import collections
import pickle
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from phc_joblib_static_parser import parse


class Exploit:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",))


class SymbolicParserTests(unittest.TestCase):
    def test_joblib_numeric_arrays_and_dictionary(self):
        sample = collections.defaultdict(dict)
        sample['a'] = {'issue': 'normal', 'bound': np.array([1, 20, 37], dtype=np.int64)}
        sample['b'] = {'issue': 'airborne', 'bound': np.array([-2.5, 3.0], dtype=np.float64)}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'synthetic.pkl'
            joblib.dump(sample, path, protocol=3, compress=0)
            result, stats = parse(path.read_bytes())
        self.assertEqual(result['a']['issue'], 'normal')
        self.assertEqual(result['a']['bound']['_numeric_array'], [1, 20, 37])
        self.assertEqual(result['b']['bound']['_numeric_array'], [-2.5, 3.0])
        self.assertEqual(stats['numeric_arrays'], 2)

    def test_rejects_executable_reduce_without_executing_it(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / 'marker'
            payload = pickle.dumps(Exploit(marker), protocol=3)
            with self.assertRaises(ValueError):
                parse(payload)
            self.assertFalse(marker.exists())

    def test_rejects_truncation_and_oversize(self):
        with self.assertRaises(ValueError):
            parse(pickle.dumps({'a': 3}, protocol=3)[:-1])
        with self.assertRaises(ValueError):
            parse(b'x' * 4_000_001)


if __name__ == '__main__':
    unittest.main()
