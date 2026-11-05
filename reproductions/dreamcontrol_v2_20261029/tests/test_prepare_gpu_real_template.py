import unittest
from pathlib import Path
from prepare_gpu_resume import generate_solver


class RealTemplateTest(unittest.TestCase):
    def test_current_mosh_solver_can_be_transformed_without_launching(self):
        path = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/mosh_000798_pyroki_fullrate_20261031/run.py')
        result = generate_solver(path.read_text())
        compile(result, '<real-gpu-resume>', 'exec')
        self.assertIn('range(len(prefix), len(starts))', result)
        self.assertIn("devices[0].platform != 'gpu'", result)


if __name__ == '__main__': unittest.main()
