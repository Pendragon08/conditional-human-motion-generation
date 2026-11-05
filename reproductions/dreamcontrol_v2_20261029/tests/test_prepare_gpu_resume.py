import unittest
from prepare_gpu_resume import generate_solver, replace_once


class ResumeTemplateTest(unittest.TestCase):
    def test_solver_transformation_is_explicit_and_compiles(self):
        template = """import json, os, jax, numpy as np
from pathlib import Path
def main():
    if any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('CPU required')
    rows, previous = [], None
    with log.open('x', encoding='utf8', buffering=1) as stream:
        for wi, start in enumerate(starts):
            pass
    report = {'source_sha256': expected, 'rows': rows}
"""
        output = generate_solver(template)
        compile(output, '<gpu-resume>', 'exec')
        self.assertIn("devices[0].platform != 'gpu'", output)
        self.assertIn('range(len(prefix), len(starts))', output)
        self.assertIn("'resumed_prefix_windows': len(prefix)", output)

    def test_unexpected_templates_rejected(self):
        with self.assertRaises(ValueError): generate_solver('pass')
        with self.assertRaises(ValueError): replace_once('xx', 'x', 'y')


if __name__ == '__main__': unittest.main()
