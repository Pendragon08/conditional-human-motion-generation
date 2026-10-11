"""Analyze retained, incomplete V3 measurement output; never start a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out, witnessed_exit_code, witness_path):
    assert witnessed_exit_code != 0, 'This analyzer is only for failed wrapper runs'
    state = json.loads((out / 'state.json').read_text())
    versions = json.loads((out / 'versions_before_execution.json').read_text())
    expected = ['%s_s%d' % (mode, seed) for seed in (11, 12, 13)
                for mode in ('planned', 'native')]
    expected += ['full_val_%s_s11' % mode for mode in ('planned', 'native')]
    errors = []; records = []; quality = {}; bindings = []; offsets = []
    completed_cases = 0; attempted_cases = 0
    for label in expected:
        folder = out / label
        request_path = out / (label + '_execution_request.json')
        if not request_path.exists():
            continue
        attempted_cases += 72 if label.startswith('full_val') else 12
        if not (folder / 'metadata.json').exists():
            errors.append(label + ': no completed measurement output')
            continue
        meta = json.loads((folder / 'metadata.json').read_text())
        receipt = json.loads((folder / 'receipt.json').read_text())
        rows = json.loads((folder / 'metrics.json').read_text())
        differences = meta['manual_reset_vs_first_action_state_max_difference']
        hashes_equal = meta['manual_reset_state_sha256'] == meta['first_action_state_sha256']
        valid = (meta['initial_actual_reference_xy_max_m'] < .001
                 and all(v < 1e-5 for v in differences.values()) and hashes_equal)
        if not valid:
            errors.append(label + ': seven exported kinematic tensors are not identical across resets')
        records.append(dict(label=label, cases=receipt['cases'], completed=receipt['completed'],
            initial_xy_max_m=meta['initial_actual_reference_xy_max_m'],
            full_state_max_difference=differences, exported_tensor_hashes_equal=hashes_equal,
            reference_alignment_invariants=meta['reference_alignment_invariants']))
        completed_cases += receipt['completed']
        quality[label] = dict(attempts=len(rows), completed=sum(r['completed'] for r in rows),
            **{key: sum(r[key] for r in rows) for key in
               ('condition_pass', 'physical_pass', 'tracking_pass', 'joint_pass')})
        lineage = json.loads((folder / 'lineage.json').read_text())
        request = json.loads(request_path.read_text())
        binding = dict(label=label, request_matches=lineage['request'] == request,
                       output_hashes_match=all(digest(folder / name) == value
                                              for name, value in lineage['outputs'].items()))
        bindings.append(binding)
        if not binding['request_matches'] or not binding['output_hashes_match']:
            errors.append(label + ': request/output hash binding failed')
        goals = np.array(meta['original_goal'])
        for i in range(0, len(meta['cases']), 3):
            group = meta['cases'][i:i+3]
            assert len({r['source_id'] for r in group}) == 1
            delta = goals[i:i+3] - goals[i+1]
            error = float(np.max(np.abs(delta - np.array([[-.15, 0, 0], [0, 0, 0], [.15, 0, 0]]))))
            offsets.append(dict(source_id=group[0]['source_id'], max_absolute_offset_error_m=error))
            if error > 1e-4:
                errors.append(label + ': nominal offsets are not controlled within a source')
    absent = [label for label in expected if not (out / (label + '_execution_request.json')).exists()]
    errors.append('Incomplete coverage: cross-seed, cross-mode and full validation checks were not run')
    ledger = json.loads((out / 'budget_ledger.json').read_text())
    prior = []
    for name in ('physics_guided_20261011', 'reset_alignment_20261011', 'reset_alignment_v2_20261011'):
        prior += json.loads((out.parent / name / 'budget_ledger.json').read_text())
    witness = json.loads(witness_path.read_text(encoding='utf8'))
    assert witness['final_exit_code'] == witnessed_exit_code
    assert witness['measurement_invocations'] == 1 and witness['retries'] == 0
    outer_seconds = witness['accounting']['v3_outer_wall_seconds']
    overhead = max(0, outer_seconds - sum(r['seconds'] for r in ledger))
    result = dict(study='independent, incomplete V3 measurement diagnosis',
        planned_attempts=versions['total_expected_attempts'], attempts=attempted_cases,
        completed_cases=completed_cases, started_batches=len(bindings), not_started_batches=absent,
        wrapper_exit_code=witnessed_exit_code, retained_original_state=state,
        state_note='Wrapper exited at resource guard between batches; retained RUNNING state is stale, not an active experiment',
        stop_trigger='No displacement of another workload resource assertion; no job was displaced',
        coordinate_gate_pass=False, coordinate_errors=errors, initial_state_checks=records,
        cross_seed_and_mode_checks=[], offset_checks=offsets, full_validation_quality={},
        available_fixture_quality=quality, request_output_bindings=bindings,
        joint_gate_floor=None, quality_thresholds_changed=False,
        repair_gpu_hours=(sum(r['seconds'] for r in ledger)+overhead)/3600,
        combined_charged_gpu_hours=(sum(r['seconds'] for r in ledger+prior)+overhead)/3600,
        outer_wrapper_overhead_charged_seconds=overhead,
        independent_witness_sha256=digest(witness_path),
        state_identity_scope='Seven exported kinematic tensors; excludes PhysX hidden state, contacts, RNG and policy observations',
        next_stage='STOP_MEASUREMENT_INVALID',
        qualification='Not a completed 216-case validation, not a controlled efficacy comparison; do not pool with historical results',
        evidence_sha256={str(p.relative_to(out)):digest(p) for p in
                         (out/'state.json',out/'budget_ledger.json',out/'versions_before_execution.json')},
        analyzer_sha256=digest(Path(__file__)))
    (out / 'measurement_repair_summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: result[k] for k in ('attempts','completed_cases','next_stage','combined_charged_gpu_hours')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--witnessed-exit-code', required=True, type=int)
    parser.add_argument('--witness-json', required=True)
    args = parser.parse_args(); main(Path(args.output), args.witnessed_exit_code, Path(args.witness_json))
