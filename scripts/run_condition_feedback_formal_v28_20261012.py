"""Complete original C1 comparison after calibrated COM/runtime repair, no new algorithm."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import run_reset_alignment_checks_v4 as runtime

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/condition_feedback_formal_v28_20261012'
V24=ROOT/'results/contact_com_validation_v24_20261012'
V27=ROOT/'results/fixed_reference_com_v27_20261012'
OLD=ROOT/'results/physics_guided_20261011'
DOC=ROOT/'docs/condition_feedback_formal_v28_20261012.md'


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')


def prepare():
    assert json.loads((V27/'replay_summary.json').read_text())['status']=='COMPLETE_FIXED_REFERENCE_CALIBRATION'
    assert json.loads((ROOT/'results/contact_com_runtime_v26_20261012/run_receipt.json').read_text())['status']=='PASS'
    assert not json.loads((V24/'validation_summary.json').read_text())['full_quality_floor'], 'Stop floor route; no threshold relaxation'
    if OUT.exists():
        versions=json.loads((OUT/'versions.json').read_text())
        assert all(runtime.sha(ROOT/name)==digest for name,digest in versions['files'].items())
        return
    OUT.mkdir()
    shutil.copyfile(OLD/'source_manifest.json',OUT/'source_manifest.json')
    shutil.copyfile(V24/'frozen_thresholds.json',OUT/'frozen_thresholds.json')
    shutil.copyfile(ROOT/'results/contact_measurement_review_20261011/budget_snapshot_20261012.json',OUT/'prior_budget_snapshot.json')
    version=json.loads((V24/'versions.json').read_text())
    version['manifest_sha256']=runtime.sha(OUT/'source_manifest.json')
    for name in ('scripts/contact_com_collect_v28_20261012.py','scripts/physics_guided_run_v28_20261012.sh',
                 'scripts/run_condition_feedback_formal_v28_20261012.py','scripts/physics_guided_analyze.py','scripts/physics_guided_semantics.py',str(DOC.relative_to(ROOT))):
        version['files'][name]=runtime.sha(ROOT/name)
    version.update(role='C1 complete equal-budget comparison and four ablations after measurement repair; no success claimed.',
                   prior_budget_snapshot_sha256=runtime.sha(OUT/'prior_budget_snapshot.json'),
                   fixed_reference_calibration_sha256=runtime.sha(V27/'replay_summary.json'),
                   test_inputs_per_arm=432, generation_seeds=[11,12,13], test_sources=48,
                   output_retention='Dense manifolds omitted; COM, states, original targets, all attempts and proxy metrics retained. Exact array equality witness required.',
                   confirmatory=False, protocol_history='Retrospective measurement repairs and previous exposure retained. Fresh output is complete paired test, not a pristine preregistered confirmatory study.',
                   expected_construction_batches=44, expected_independent_replay_batches=30,
                   innovation_confirmed=False, overleaf_results_sync=False)
    save(OUT/'versions.json',version)
    save(OUT/'budget_ledger.json',[])


def main(check,inside):
    if not inside:
        os.execve(str(runtime.PYTHON),[str(runtime.PYTHON),'-u',str(Path(__file__).resolve()),
                                    '--check' if check else '--run','--inside'],runtime.environment())
    import numpy as np
    assert np.__version__=='1.23.5'
    prepare()
    # Reuse the existing algorithms verbatim; only resolve the calibrated
    # verifier, cumulative-output guard and V27 document-witness prerequisite.
    original=ROOT/'scripts/physics_guided_queue.py'
    source=original.read_text().split("if __name__=='__main__':")[0]
    old='from physics_guided_lineage import request, certify_cached, file_hash, motion_hash, object_hash'
    assert source.count(old)==1
    source=source.replace(old,'from physics_guided_com_lineage_v24_20261012 import request, certify_cached, file_hash, motion_hash, object_hash')
    old="sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())"
    assert source.count(old)==2
    source=source.replace(old,'cumulative_output_bytes(self.out)')
    begin="            witness=json.loads((out/'witness_doc_v4/receipt.json').read_text())"
    end="            frozen=out/'frozen_thresholds.json'"
    start=source.index(begin); finish=source.index(end,start)
    source=source[:start]+"            witness=json.loads((out/'runtime_witness.json').read_text())\n            assert witness['status']=='PASS' and witness['exact_arrays_unchanged']\n"+source[finish:]
    namespace={'__file__':str(original),'__name__':'formal_v28_queue'}
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    prior=json.loads((OUT/'prior_budget_snapshot.json').read_text())
    prior_entries=[r for r in prior['entries'] if r['directory']!=OUT.name]
    def output_bytes(out):
        return prior['output_bytes']+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    namespace['cumulative_output_bytes']=output_bytes
    namespace['SHELL']=ROOT/'scripts/physics_guided_run_v28_20261012.sh'
    Base=namespace['Queue']
    class WarmQueue(Base):
        def __init__(self,out):
            self.out=out;self.ledger_path=out/'budget_ledger.json'
            self.ledger=json.loads(self.ledger_path.read_text())
        def seconds(self,stage=None):
            return sum(float(r['seconds']) for r in prior_entries+self.ledger if stage is None or r['stage']==stage)
        def preflight(self,stage):
            runtime.OUT=self.out
            runtime.resource_guard()
            assert shutil.disk_usage('/data').free>=20*1024**3
            assert output_bytes(self.out)<15*1024**3
            assert self.seconds()<72*3600
            assert self.seconds(stage)<namespace['CONFIG']['budget_gpu_hours'][stage]*3600
    namespace['Queue']=WarmQueue
    if check:
        import fcntl
        with (OUT/'queue.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            q=WarmQueue(OUT)
            path=q.run('val_planned_s11_r0','measurement','val','planned',11)
            old=V24/'raw/val_planned_s11'
            exact={}
            for filename in ('trajectories.npz','sensor_diagnostic.npz'):
                with np.load(path/filename,allow_pickle=False) as current,np.load(old/filename,allow_pickle=False) as previous:
                    assert current.files==previous.files
                    for key in current.files:
                        exact[filename+':'+key]=bool(np.array_equal(current[key],previous[key]))
            assert all(exact.values()), 'Removing passive diagnostic reads must not change simulation'
            assert json.loads((path/'metrics.json').read_text())==json.loads((old/'metrics.json').read_text())
            meta=json.loads((path/'metadata.json').read_text())
            assert meta['dense_manifold_recording'] is False
            assert meta['original_goal_sha256']==json.loads((old/'metadata.json').read_text())['original_goal_sha256']
            save(OUT/'runtime_witness.json',dict(status='PASS',exact_arrays_unchanged=True,exact_fields=exact,
                 collector_sha256=runtime.sha(ROOT/'scripts/contact_com_collect_v28_20261012.py'),
                 metrics_unchanged=True,thresholds_unchanged=True,innovation_confirmed=False))
            q.state('WITNESS_PASS')
            print('CONDITION_FEEDBACK_FORMAL_DOCUMENT_CHECK_PASS',flush=True)
    else:
        assert json.loads((OUT/'runtime_witness.json').read_text())['status']=='PASS'
        namespace['main'](OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();main(args.check,args.inside)
