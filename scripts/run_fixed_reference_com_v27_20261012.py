"""Fixed-reference replay calibration, preserving failed construction denominators."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import run_reset_alignment_checks_v4 as runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fixed_reference_com_v27_20261012'
V24 = ROOT / 'results/contact_com_validation_v24_20261012'
V26 = ROOT / 'results/contact_com_runtime_v26_20261012'
DOC = ROOT / 'docs/fixed_reference_com_v27_20261012.md'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def prepare():
    assert json.loads((V26 / 'run_receipt.json').read_text())['status'] == 'PASS'
    full = json.loads((V24 / 'validation_summary.json').read_text())
    assert full['attempts'] == 432
    if OUT.exists():
        version = json.loads((OUT / 'versions.json').read_text())
        assert all(runtime.sha(ROOT / name) == digest for name, digest in version['files'].items())
        return
    OUT.mkdir()
    for name in ('source_manifest.json', 'frozen_thresholds.json'):
        shutil.copyfile(V24 / name, OUT / name)
    version = json.loads((V24 / 'versions.json').read_text())
    for name in (str(Path(__file__).relative_to(ROOT)), str(DOC.relative_to(ROOT))):
        version['files'][name] = runtime.sha(ROOT / name)
    base = V24 / 'raw/val_planned_s11'
    version.update(role='Fixed generated vs physically refined reference replay calibration, not candidate efficacy.',
                   parent_outputs={name: runtime.sha(base/name) for name in ('metadata.json','metrics.json','trajectories.npz')},
                   expected_construction_batches=0, expected_independent_replay_batches=2,
                   qualification='All 24 preselected validation sources and 3 offsets retained. Construction-terminated cases remain failures in combined success. No generator sampling on replay, no data selection by outcome.',
                   frozen_targets=True, overleaf_results_sync=False, innovation_confirmed=False)
    save(OUT / 'versions.json', version)
    save(OUT / 'budget_ledger.json', [])
    (OUT / 'raw').mkdir()
    (OUT / 'requests').mkdir()


def main(check, inside):
    if not inside:
        os.execve(str(runtime.PYTHON), [str(runtime.PYTHON), '-u', str(Path(__file__).resolve()),
                  '--check' if check else '--run', '--inside'], runtime.environment())
    import fcntl
    import numpy as np
    assert np.__version__ == '1.23.5'
    prepare()
    import run_contact_com_validation_v24_20261012 as queue
    from physics_guided_queue import select
    queue.OUT = OUT
    runtime.OUT = OUT
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            base = V24 / 'raw/val_planned_s11'
            raw = select(OUT, [base], 'raw_reference', raw=True)
            queue.batch('val_raw_replay_s11', 'replay', 11, replay_file=raw)
            if check:
                save(OUT/'state.json', dict(status='WITNESS_PASS', scope='fixed raw-reference interface only'))
                print('FIXED_REFERENCE_COM_DOCUMENT_CHECK_PASS', flush=True)
                return
            refined = select(OUT, [base], 'refined_reference', raw=False)
            queue.batch('val_refined_replay_s11', 'replay', 11, replay_file=refined)
            entries=[]
            for name in ('raw', 'refined'):
                folder = OUT / ('raw/val_%s_replay_s11' % name)
                rows = json.loads((folder/'metrics.json').read_text())
                choices = json.loads((OUT/'selected'/('%s_reference.json'%name)).read_text())['choices']
                assert len(rows) == len(choices) == 72
                combined = [bool(row['joint_pass'] and choice['construction_complete']) for row, choice in zip(rows, choices)]
                entries.append(dict(reference=name, attempted=72,
                    **{key:sum(row.get(key,False) for row in rows) for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},
                    combined_construction_replay_joint_pass=sum(combined),
                    construction_terminated=sum(not c['construction_complete'] for c in choices),
                    metadata_sha256=runtime.sha(folder/'metadata.json'), metrics_sha256=runtime.sha(folder/'metrics.json')))
            result=dict(status='COMPLETE_FIXED_REFERENCE_CALIBRATION', entries=entries,
                        measurement_scope='Frozen reference clock and original-goal consistency under replay; calibrated COM surface proxy. Not ground-truth contact force or arbitrary rolling-contact truth.',
                        diagnostic_only=True, generation_seeds=[11], source_count=24,
                        full_physical_quality_floor=all(e['physical_pass']==0 for e in entries),
                        thresholds_changed=False, overleaf_results_sync=False, innovation_confirmed=False)
            save(OUT/'replay_summary.json', result)
            save(OUT/'state.json', dict(status=result['status']))
            print('FIXED_REFERENCE_COM_CALIBRATION_COMPLETE', json.dumps(result), flush=True)
        except Exception as error:
            save(OUT/'state.json', dict(status='FAILED_RETAINED', error=repr(error)))
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args=parser.parse_args()
    main(args.check, args.inside)
