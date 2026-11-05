import unittest
from gpu_retarget_preflight import assess


class PreflightTest(unittest.TestCase):
    def test_busy_gpu_rejected_even_with_free_memory(self):
        result = assess(0, '0, GPU-a, 40000\n1, GPU-b, 30000', 'GPU-a, 17\nGPU-b, 18', True, 10240)
        self.assertFalse(result['gpu_retarget_may_start'])
        self.assertEqual(result['existing_compute_pids'], [17])
        self.assertIn('existing_compute_process', result['reasons'])

    def test_missing_plugin_rejected_and_idle_candidate_passes_only_this_check(self):
        self.assertEqual(assess(1, '1, GPU-b, 30000', '', False, 10240)['reasons'], ['jax_cuda_plugin_missing'])
        self.assertTrue(assess(1, '1, GPU-b, 30000', '', True, 10240)['gpu_retarget_may_start'])

    def test_sharing_requires_explicit_authorization(self):
        result = assess(0, '0, GPU-a, 40000', 'GPU-a, 17', True, 10240, allow_shared=True)
        self.assertTrue(result['gpu_retarget_may_start'])
        self.assertEqual(result['existing_compute_pids'], [17])
        self.assertTrue(result['shared_use_explicitly_allowed'])

    def test_invalid_inventory_fails_closed(self):
        with self.assertRaises(ValueError): assess(0, '0, GPU-a, 10000', 'bad', True, 1024)
        with self.assertRaises(ValueError): assess(2, '0, GPU-a, 10000', '', True, 1024)


if __name__ == '__main__': unittest.main()
