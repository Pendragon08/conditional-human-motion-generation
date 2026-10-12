"""Read saved evidence only; no experiment entrypoint imports or GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OLD = ROOT / 'results/task_matched_feedback_v2_20261012'
OUT = ROOT / 'results/task_matched_feedback_continue_20261012'
TRACE = ROOT / '.aris/traces/task_matched_feedback_continue_20261012/document_witness'
PREVIOUS_TRACE = ROOT / '.aris/traces/task_matched_feedback_v2_20261012/document_witness'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

old, new = read(OLD / 'versions.json'), read(OUT / 'versions.json')
runtime = read(OUT / 'runtime_witness.json')
prior_runtime, prior_witness = read(OLD / 'runtime_witness.json'), read(OLD / 'DOCUMENT_WITNESS.json')
receipt = read(TRACE / 'exec_trace.json')
assert receipt['actual_exit_code'] == 0 and receipt['invocation_count'] == 1
assert receipt['sentinel'] == 'TASK_MATCHED_CONTINUATION_CHECK_PASS'
assert receipt['sentinel'] in receipt['trace'][-1]['output']
assert runtime['status'] == 'PASS' and runtime['new_gpu_batches'] == 0
assert runtime['versions_sha256'] == sha(OUT / 'versions.json')
assert runtime['previous_actual_runtime_witness_sha256'] == sha(OLD / 'runtime_witness.json')
assert runtime['previous_document_witness_sha256'] == sha(OLD / 'DOCUMENT_WITNESS.json')
assert prior_witness['status'] == prior_runtime['status'] == 'PASS'
assert prior_witness['actual_exit_code'] == 0 and prior_witness['gpu_batches_started'] == 3
assert prior_witness['versions_sha256'] == sha(OLD / 'versions.json') == prior_runtime['versions_sha256']
assert prior_witness['runtime_witness_sha256'] == sha(OLD / 'runtime_witness.json')
old_exec = read(PREVIOUS_TRACE / 'exec_trace.json')
assert old_exec['actual_exit_code'] == old_exec['trace'][-1]['exit_code'] == 0
assert old_exec['session_id'] == prior_witness['exec_session_id']
assert old_exec['trace'][-1]['chunk_id'] == prior_witness['terminal_chunk_id']
assert 'TASK_MATCHED_FEEDBACK_CHECK_PASS' in old_exec['trace'][-1]['output']
assert sha(PREVIOUS_TRACE / 'raw_verification.json') == prior_witness['raw_verification_sha256']
prior_raw = read(PREVIOUS_TRACE / 'raw_verification.json')
assert prior_raw['status'] == 'PASS'
assert prior_runtime['actual_modes'] == ['native', 'native_zero_correction', 'replay']
assert prior_runtime['cases_per_batch'] == 72 and all(prior_runtime['exact_same_input_arrays'].values())

allowed = {'scope', 'max_new_seconds', 'carried_first_hashes', 'batch_implementation_sha256',
           'original_measurement_seconds', 'total_new_simulator_batches', 'extra_candidates_per_input', 'budget_reason'}
version_diff = [k for k in set(old) | set(new) if k != 'files' and old.get(k) != new.get(k)]
assert set(version_diff) == allowed
assert all(new['files'][k] == v for k, v in old['files'].items())
assert set(new['files']) - set(old['files']) == {'scripts/task_matched_feedback_continue_20261012.py',
    'docs/task_matched_feedback_continue_20261012.md', 'docs/task_matched_feedback_20261012.md'}
assert all(sha(ROOT / k) == v for k, v in new['files'].items())
assert old['max_new_seconds'] == 1800 and new['max_new_seconds'] == 3600
assert new['total_new_simulator_batches'] == 20 and new['extra_candidates_per_input'] == 0
assert new['candidates_per_method'] == 3 and new['seeds'] == [11, 12, 13] and new['inputs_per_method'] == 216
for name, key in [('source_manifest.json', 'source_manifest_sha256'), ('frozen_thresholds.json', 'thresholds_sha256')]:
    assert sha(OUT / name) == sha(OLD / name) == old[key] == new[key]
manifest = read(OUT / 'source_manifest.json')
val_rows = [x for x in manifest['rows'] if x['split'] == 'val']
assert len(val_rows) == len({x['source_id'] for x in val_rows}) == 24
config = read(ROOT / 'configs/physics_guided_20261011.json')
assert config['offsets_m'] == [-0.15, 0.0, 0.15]
assert config['generator_frozen'] and config['controller_frozen_during_generation']

source = (ROOT / 'scripts/task_matched_feedback_20261012.py').read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'batch')
original_function = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
old_expression = "1800 - sum(r['seconds'] for r in ledger)"
assert original_function.count(old_expression) == 1
assembled = original_function.replace(old_expression, "3600 - sum(r['seconds'] for r in ledger)")
assembled_hash = hashlib.sha256(json.dumps(assembled, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert assembled_hash == new['batch_implementation_sha256']
for expression in ['min(180,', '72 * 3600', "stage_cap = 4 if stage == 'measurement' else 12",
                   '20 * 1024**3', '15 * 1024**3', 'Failed charged invocation retained']:
    assert expression in assembled
for p, digest in new['carried_first_hashes'].items():
    assert sha(ROOT / p) == digest

ledger = read(OLD / 'budget_ledger.json')
assert len(ledger) == 3 and all(x['exit_code'] == 0 and x['stage'] == 'measurement' for x in ledger)
assert read(OUT / 'budget_ledger.json') == []
assert sum(x['seconds'] for x in ledger) == new['original_measurement_seconds']
batches = []
for entry, previous in zip(ledger, prior_raw['batches']):
    label = entry['label']; path = OLD / 'raw' / label; request_path = OLD / 'requests' / (label + '.json')
    request = read(request_path); lineage = read(path / 'lineage.json'); complete = read(path / 'receipt.json')
    data, weights = read(path / 'generation_input_binding.json'), read(path / 'loaded_checkpoint_binding.json')
    meta = read(path / 'metadata.json')
    assert entry['request_sha256'] == sha(request_path) == data['request_sha256'] == weights['request_sha256']
    assert request['versions'] == sha(OLD / 'versions.json') == data['versions_sha256'] == weights['versions_sha256']
    assert lineage['certification'] == 'bound_request_at_execution' and lineage['request'] == request
    assert set(lineage['outputs']) == {'metadata.json', 'metrics.json', 'trajectories.npz'}
    assert all(sha(path / k) == v for k, v in lineage['outputs'].items())
    assert complete['status'] == 'complete' and complete['cases'] == len(read(path / 'metrics.json')) == 72
    assert complete['trajectories_sha256'] == lineage['outputs']['trajectories.npz']
    assert data['exact_loaded_dataset_bytes'] == old['external_dataset_files']
    assert data['loaded_correction_object_sha256'] == request.get('correction')
    assert weights['controller_state'] == 't2m'
    assert sorted(weights['actual_loaded_weight_bytes'].values()) == sorted([old['generator_weights_sha256'], old['controller_states']['t2m']['sha256']])
    assert sha(path / 'initial_states.npz') == meta['initial_state_archive_sha256']
    assert sha(path / 'observation_trace.npz') == meta['observation_trace_sha256']
    assert meta['initial_reset_gate_pass'] and not any(meta['initial_rigid_cache_consistency'].values())
    assert previous['label'] == label and sha(OLD / (label + '.log')) == previous['log_sha256']
    batches.append({'label': label, 'cases': 72, 'actual_exit_code': entry['exit_code'], 'charged_seconds': entry['seconds'],
                    'original_request_sha256': sha(request_path), 'original_versions_sha256': sha(OLD / 'versions.json')})
first_request = read(OLD / 'requests/val_shared_s11_r0.json')
assert (first_request['mode'], first_request['seed'], first_request['round'], first_request['noise_round']) == ('native', 11, 0, 0)

snapshot_path = sorted((ROOT / 'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1]
resource = read(snapshot_path)
assert str(OLD.relative_to(ROOT)) in resource['directories']
assert resource['total_cap_hours'] == 72 and resource['stage_caps_hours']['measurement'] == 4 and resource['stage_caps_hours']['comparison'] == 12
assert resource['output_cap_bytes'] == 15 * 1024**3 and resource['data_free_floor_bytes'] == 20 * 1024**3
assert resource['status'] == 'CAPS_PASS' and all(resource['cap_checks'].values())
spec = read(ROOT / 'configs/env-closd-baseline-20261010.json')
spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert spec_hash == '81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780'

verification = {'status': 'PASS', 'new_gpu_batches': 0, 'versions_sha256': sha(OUT / 'versions.json'),
    'runtime_witness_sha256': sha(OUT / 'runtime_witness.json'), 'carried_hash_count': len(new['carried_first_hashes']),
    'carried_first_original_request_sha256': sha(OLD / 'requests/val_shared_s11_r0.json'),
    'carried_first_original_versions_sha256': sha(OLD / 'versions.json'), 'previous_actual_three_batch_witness': batches,
    'same_input_arrays_preceding_witness': len(prior_runtime['exact_same_input_arrays']),
    'scientific_controls_unchanged': True, 'assembly_sha256': assembled_hash, 'env_spec_sha256': spec_hash,
    'original_measurement_seconds': new['original_measurement_seconds'], 'new_dispatch_seconds': 3600,
    'new_dispatch_gpu_batch_limit': 20, 'original_user_caps': {'total_gpu_hours': 72, 'measurement_hours': 4,
        'comparison_hours': 12, 'output_gib': 15, 'data_free_floor_gib': 20},
    'resource_snapshot': str(snapshot_path.relative_to(ROOT)), 'resource_snapshot_sha256': sha(snapshot_path),
    'resource_snapshot_status': resource['status'], 'version_diff_keys': sorted(version_diff),
    'hidden_physx_identity_established': False, 'candidate_efficacy_test': False}
with (TRACE / 'binding_verification.json').open('x') as f:
    json.dump(verification, f, indent=2)
witness = dict(verification, witness_agent='/root/task_matched_continuation_document_witness',
    observed_utc=datetime.now(timezone.utc).isoformat(), document='docs/task_matched_feedback_continue_20261012.md',
    referenced_documents=['docs/task_matched_feedback_20261012.md', 'docs/task_matched_feedback_v2_20261012.md'],
    environment_ledger='.aris/compute/closd-baseline-20261010.md', actual_documented_command=receipt['documented_command'],
    actual_ssh_command=receipt['actual_ssh_command'], actual_exit_code=0, exec_session_id=receipt['session_id'],
    terminal_chunk_id=receipt['trace'][-1]['chunk_id'], sentinel=receipt['sentinel'], invocations=1, retries=0,
    same_family=True, provisional=True, independence='Fresh agent follows documentation; same-family/provisional, not external scientific review',
    binding_verification_sha256=sha(TRACE / 'binding_verification.json'), actual_tool_trace_sha256=sha(TRACE / 'exec_trace.json'),
    previous_document_witness_sha256=sha(OLD / 'DOCUMENT_WITNESS.json'), previous_runtime_witness_sha256=sha(OLD / 'runtime_witness.json'),
    preceding_actual_exit_code=0, preceding_exec_session_id=old_exec['session_id'], preceding_terminal_chunk_id=old_exec['trace'][-1]['chunk_id'],
    source_or_threshold_changes=False, installs=False, prepare_invocations=0, run_invocations=0,
    innovation_verdict_issued=False, semantic_correctness_certified=False, overleaf_results_sync=False,
    scope='CPU dispatch binding and carried original execution binding only; actual GPU interfaces rely on preceding independently witnessed three batches',
    trace_directory=str(TRACE.relative_to(ROOT)) + '/')
with (OUT / 'DOCUMENT_WITNESS.json').open('x') as f:
    json.dump(witness, f, indent=2)
markdown = '''# Independent continuation document witness

Status: PASS. Fresh document-following agent; same-family/provisional.

The documented CPU --check was executed exactly once over SSH, with actual terminal exit 0 and TASK_MATCHED_CONTINUATION_CHECK_PASS. Tool session 73316; launch chunk d257b9; terminal chunk 405beb. No retry, prepare, run, installation, source edit, threshold edit, or Overleaf action occurred in this witness.

```bash
''' + receipt['documented_command'] + '\n```\n\n' + f'''Frozen versions SHA256: {witness['versions_sha256']}.
Runtime witness SHA256: {witness['runtime_witness_sha256']}.
Assembled dispatcher SHA256: {assembled_hash}.

All 14 carried hashes remain bound. Seed 11 raw first candidate retains its original request and versions; it is neither regenerated nor relabeled as a new execution. Original versions SHA256: {sha(OLD / 'versions.json')}. Original request SHA256: {sha(OLD / 'requests/val_shared_s11_r0.json')}.

The preceding v2 independent witness has actual exit 0, session {old_exec['session_id']}, terminal chunk {old_exec['trace'][-1]['chunk_id']}, three first-attempt GPU batches with 72 cases each and 0 exit codes, and {new['original_measurement_seconds']:.6f} charged seconds. Its 28 exact-repeat exported array checks and fixed-reference replay bindings remain the existing evidence. This CPU witness creates new_gpu_batches=0; it does not newly certify GPU operation or hidden PhysX identity.

All old frozen scientific fields and source bindings match. Manifest, targets, thresholds, generator and t2m tracker, 24 validation sources, three offsets, seeds 11/12/13, three candidates per method and 216 inputs per method remain fixed. The assembled batch function differs by exactly the allocation expression 1800 -> 3600 seconds. This is a prospective separate dispatch allocation for 20 additional batches; the 275.798321 seconds already measured remain separately charged, and the carried first candidate is not charged twice. The original 72 GPU-hour total, 4-hour measurement, 12-hour comparison, 15 GiB output and 20 GiB free-disk gates remain. The resource snapshot includes the v2 measurement directory and has CAPS_PASS.

No efficacy, innovation, semantic correctness, unseen-source generalization, robot benefit, downstream readiness, or hidden simulator-state identity is certified. C2/C3/G1 and downstream gates remain unchanged. The result establishes dispatch and provenance bindings only.

Full trace: {TRACE.relative_to(ROOT)}/. Only small JSON/Markdown/log files may be copied locally; NPZ archives and weights remain remote.
'''
with (OUT / 'DOCUMENT_WITNESS.md').open('x') as f:
    f.write(markdown)
for name in ['DOCUMENT_WITNESS.json', 'DOCUMENT_WITNESS.md']:
    (TRACE / name).write_bytes((OUT / name).read_bytes())
print(json.dumps({'status': 'PASS', 'versions_sha256': witness['versions_sha256'],
                  'runtime_witness_sha256': witness['runtime_witness_sha256'], 'new_gpu_batches': 0,
                  'document_witness_sha256': sha(OUT / 'DOCUMENT_WITNESS.json')}))
