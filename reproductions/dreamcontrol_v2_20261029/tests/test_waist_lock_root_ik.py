import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from waist_lock_root_ik import _exp_rotation, _quat_matrix, _quat_multiply, _project
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires torch and pytorch_kinematics')
class RootFitMathTest(unittest.TestCase):
    def test_rotation_and_quaternion_conventions(self):
        v = torch.tensor([[0., 0., np.pi/2]], dtype=torch.float64)
        matrix = _exp_rotation(v).numpy()[0]
        expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(matrix, expected, atol=1e-12)
        quat = torch.tensor([[0., 0., np.sin(np.pi/4), np.cos(np.pi/4)]], dtype=torch.float64)
        np.testing.assert_allclose(_quat_matrix(quat).numpy()[0], expected, atol=1e-12)
        identity = torch.tensor([[0., 0., 0., 1.]], dtype=torch.float64)
        np.testing.assert_allclose(_quat_multiply(identity, quat).numpy(), quat.numpy(), atol=1e-12)

    def test_zero_rotation_has_finite_derivative_and_projection(self):
        v = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
        _exp_rotation(v).sum().backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        vec = torch.tensor([[3., 4., 0.], [0., 0., 0.]], dtype=torch.float64)
        _project(vec, .1)
        np.testing.assert_allclose(torch.linalg.vector_norm(vec, dim=1).numpy(), [.1, 0.], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
