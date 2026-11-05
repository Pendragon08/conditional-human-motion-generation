import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_aliasing import high_frequency_fraction


class AliasingAuditTest(unittest.TestCase):
    def test_low_and_high_frequency_signals(self):
        t = np.arange(300) / 30.0
        low = np.sin(2 * np.pi * 2 * t)[:, None]
        high = np.sin(2 * np.pi * 12 * t)[:, None]
        self.assertLess(high_frequency_fraction(low, 30), 0.001)
        self.assertGreater(high_frequency_fraction(high, 30), 0.99)

    def test_static_and_invalid_signals(self):
        self.assertIsNone(high_frequency_fraction(np.ones((30, 2)), 30))
        with self.assertRaises(ValueError):
            high_frequency_fraction(np.array([[np.nan]] * 4), 30)


if __name__ == '__main__':
    unittest.main()
