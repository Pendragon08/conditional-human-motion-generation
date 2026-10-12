"""Append-only V53 remediation: execution-lineage check and configured bootstrap.

Original V51/V53 sources, files, scores and audit findings remain unchanged.
This is a CPU-only analysis amendment, not a new simulation or method.
"""
import hashlib
import json
from pathlib import Path
import time
from verify_execution_lineage_v39_20261012 import guard_batch, sha
from physics_guided_lineage import motion_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/paired_feedback_validation_v51_20261012'
DEST = ROOT/'results/paired_feedback_analysis_v54_20261012'
parent = ROOT/'scripts/analyze_paired_feedback_v51_20261012.py'
assert sha(parent) == '8f99e1901ca091003dc431f902c525eaa877877736982447311a677278be8eec'
assert not DEST.exists(), 'Prior attempt retained; no automatic rerun'
summary = json.loads((OUT/'summary.json').read_bytes())
versions = json.loads((OUT/'versions.json').read_bytes())
config = json.loads((ROOT/'configs/physics_guided_20261011.json').read_bytes())
assert sha(ROOT/'configs/physics_guided_20261011.json') == versions['files']['configs/physics_guided_20261011.json']
assert sha(OUT/'source_manifest.json') == versions['source_manifest_sha256']
assert sha(OUT/'frozen_thresholds.json') == versions['thresholds_sha256']
ledger = json.loads((OUT/'budget_ledger.json').read_bytes())
assert len(ledger) == 21
labels = {row['label'] for row in ledger}
assert len(labels) == 21 and labels == {p.name for p in (OUT/'raw').iterdir() if p.is_dir()}
checks = {}
for label in sorted(labels):
    path = OUT/'raw'/label
    request = json.loads((OUT/'requests'/(label+'.json')).read_bytes())
    assert request['manifest'] == versions['source_manifest_sha256']
    assert request['thresholds'] == versions['thresholds_sha256']
    assert request['versions'] == sha(OUT/'versions.json')
    assert request['split'] == 'val'
    row = next(x for x in ledger if x['label'] == label)
    assert row['request_sha256'] == sha(OUT/'requests'/(label+'.json'))
    record = guard_batch(path, request)
    metadata = json.loads((path/'metadata.json').read_bytes())
    assert sha(path/'observation_trace.npz') == metadata['observation_trace_sha256']
    if '_replay_' in label:
        arm = 'feedback' if '_feedback_' in label else 'random'
        seed = request['seed']
        selection_label = '%s_s%d' % (arm,seed)
        assert record == summary['replay_records'][selection_label]
        selection = OUT/'selected'/(selection_label+'.json')
        selection_lineage = json.loads((OUT/'selected'/(selection_label+'.lineage.json')).read_bytes())
        assert sha(selection) == selection_lineage['selection_json']
        assert motion_hash(OUT/'selected'/(selection_label+'.npz')) == selection_lineage['selected_motion'] == request['replay_motion']
        candidates = ['val_shared_s%d_r0'%seed]+['val_%s_s%d_r%d'%(arm,seed,r) for r in (1,2)]
        assert len(selection_lineage['candidate_outputs']) == 3
        for name,binding in zip(candidates,selection_lineage['candidate_outputs']):
            assert binding == json.loads((OUT/'raw'/name/'lineage.json').read_bytes())['outputs']
            for field,digest in binding.items():
                assert sha(OUT/'raw'/name/field) == digest
    checks[label] = record
assert sum('_replay_' in x for x in labels) == 6
assert versions['construction_batches'] == 15 and versions['independent_replay_batches'] == 6
assert versions['sources'] == 24 and versions['inputs_per_seed'] == 72
# Record current consistency only, not retrospective proof of all resume guards.
certificate = dict(status='CURRENT_EXECUTION_BINDINGS_MATCH', batches=checks,
    source_manifest_sha256=sha(OUT/'source_manifest.json'), thresholds_sha256=sha(OUT/'frozen_thresholds.json'),
    version_sha256=sha(OUT/'versions.json'), summary_sha256=sha(OUT/'summary.json'),
    source_sha256=sha(Path(__file__)), parent_analysis_source_sha256=sha(parent),
    corrected_scope=dict(split='val',sources=24,seeds=[11,12,13],inputs_per_method=216,
        construction_batches=15,replay_batches=6,test_completed=False,innovation_confirmed=False),
    legacy_metadata_warning='Original versions retains V28 test/44/30 fields. Preserved; not evidence of V51 test completion.',
    original_resume_guard_warning='V51 existing-directory manifest/threshold guard incompleteness preserved; this check confirms current files against execution records only.',
    original_tracking_label_warning='Original synthetic-reference labeling inadequacy preserved; explicit supplementary evaluation provenance required.',
    original_analysis_warning='V53 snapshot-only input bindings and 10000 bootstrap repetitions preserved; V54 verifies execution binding and reads configured 5000.',
    bootstrap_repetitions=config['bootstrap_repetitions'], min_gain_percentage_points=config['min_gain_percentage_points'],
    generation_calls=0,simulation_calls=0,original_files_changed=False,time=time.time())
DEST.mkdir()
(DEST/'execution_binding_certificate.json').write_text(json.dumps(certificate,indent=2))
source = parent.read_text()
replacements = {
    "'results/paired_feedback_analysis_v53_20261012'":"'results/paired_feedback_analysis_v54_20261012'",
    'size=(10000, 24)':'size=(config["bootstrap_repetitions"], 24)',
    'bootstrap_resamples=10000':'bootstrap_resamples=config["bootstrap_repetitions"]',
    'gain >= 10':'gain >= config["min_gain_percentage_points"]',
}
for old,new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old,new)
old='bound = {}'
assert source.count(old) == 1
source=source.replace(old,"bound = {str((DEST/'execution_binding_certificate.json').relative_to(ROOT)): sha(DEST/'execution_binding_certificate.json'), str(parent.relative_to(ROOT)): sha(parent)}")
exec(compile(source,str(parent),'exec'),globals())
