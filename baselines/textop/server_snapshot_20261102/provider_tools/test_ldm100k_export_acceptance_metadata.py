"""New isolated CPU-export acceptance metadata negatives; no production inputs."""
import argparse
import copy
import json
import pathlib
import sys

B = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'scripts'))
from ldm100k_export_acceptance import validate_export_receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    ck = dict(path='/fixture/ldm_state_100000.pth', bytes=123, sha256='a' * 64,
              manager_step=100000, actual_optimizer_updates=100000)
    binding = dict(run='user_ldm100k_prefix_export_v1', spec_sha256='b' * 64,
                   inference=dict(path='/fixture/inference.pth', bytes=111, sha256='c' * 64))
    spec = dict(executor_sha256='d' * 64)
    receipt = dict(run=binding['run'], status='USER_LDM100000_PREFIX_CPU_EXPORT_PASSED',
                   output_writers_closed=True, exact_inference_reload=True, source_sha256='d' * 64,
                   spec_sha256='b' * 64, control_receipt_sha256='e' * 64, selected_checkpoint=ck,
                   inference=binding['inference'], source_full_state_preserved=True,
                   original_interrupted_exit_retained=True, new_gpu_contexts=0, new_sdk_tasks=0,
                   training_optimizer_updates_performed=0, original300k_completion_claimed=False,
                   full_native_schedule_completed=False, semantic_quality_accepted=False,
                   cuda_rng_restored=False, cpu_full_state_loads=1, cpu_model_loads=2,
                   adam_step_min=100000, adam_step_max=100000, finite_cpu_tensors=5,
                   synthetic_cpu_reload_probe=dict(batch=2, conditional_and_unconditional=True,
                                                   forward_calls=4, exact=True, heldout_semantics=False))
    validate_export_receipt(receipt, spec, binding, ck, 'e' * 64)
    cases = [('output_writers_closed', False), ('exact_inference_reload', False),
             ('source_sha256', '0' * 64), ('spec_sha256', '0' * 64),
             ('control_receipt_sha256', '0' * 64), ('new_gpu_contexts', 1), ('new_sdk_tasks', 1),
             ('training_optimizer_updates_performed', 1), ('original300k_completion_claimed', True),
             ('full_native_schedule_completed', True), ('semantic_quality_accepted', True),
             ('cuda_rng_restored', True), ('adam_step_min', 99700), ('adam_step_max', 300000),
             ('finite_cpu_tensors', 0), ('selected_checkpoint', dict(ck, manager_step=99700)),
             ('inference', dict(binding['inference'], sha256='0' * 64)),
             ('synthetic_cpu_reload_probe', dict(receipt['synthetic_cpu_reload_probe'], heldout_semantics=True))]
    for key, value in cases:
        bad = copy.deepcopy(receipt)
        bad[key] = value
        try:
            validate_export_receipt(bad, spec, binding, ck, 'e' * 64)
        except AssertionError:
            pass
        else:
            raise AssertionError('Corrupted new export fixture accepted: ' + key)
    z = dict(status='NEW_LDM100K_EXPORT_ACCEPTANCE_METADATA_FIXTURES_PASSED', rejected_cases=len(cases),
             actual_production_checkpoint_reads=0, actual_models_loaded=0, actual_gpu_contexts=0,
             actual_production_signals=0, not_actual_export_or_verification=True)
    (out / 'SUMMARY.json').write_text(json.dumps(z, indent=2), encoding='utf-8')
    print(json.dumps(z))


if __name__ == '__main__':
    main()
