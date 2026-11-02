"""Actual immutable library closure acceptance, separate from Stage3."""
import json
from step4_train_only_protocol import ROOT, owned, read, sha, scope


def accepted_library(binding):
    run = 'user_step4_train_only_fk_library_v1'; parent = ROOT/'step34_numeric_v1'
    rp = owned(parent/run/'receipt.json'); rec = read(rp)
    assert sha(rp) == binding['receipt_sha256']
    assert rec['status'] == 'STAGE4_ORIGINAL_TRAIN_ISAAC_REFERENCES_FINISHED' and rec['output_writers_closed']
    assert rec['exported_clips'] == 5551 and rec['control_intervals'] == 2835665
    assert rec['generated_training_clips'] == rec['original_VAL_motions_in_training'] == 0
    log = parent/(run+'_supervision')
    for key, path in dict(process=log/'process_exit.json', exit=log/'exit.json',
        outer=ROOT/'receipts'/('step4_train_only_'+run+'_outer_exit.json')).items():
        assert sha(owned(path)) == binding['closure_sha256'][key]
        value = read(path)
        if key == 'outer': assert value['actual_returncode'] == 0
        else: assert value['supervision_passed'] and value['child_exit_code'] == 0 and value['child_reaped'] and not value['timed_out'] and value['sample_errors'] == []
    mp = owned(rec['private_manifest']['path']); assert sha(mp) == rec['private_manifest']['sha256'] == binding['manifest_sha256']
    clips, sources, names, source_binding = scope(); assert source_binding == rec['input_binding']
    rows = [json.loads(x) for x in mp.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(clips) == 5551
    for clip, row in zip(clips, rows):
        assert all(row[k] == v for k, v in clip.items())
        path = owned(row['file']); assert path.parent == parent/run
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    actual = dict(status='STAGE4_ORIGINAL_TRAIN_LIBRARY_ACCEPTED', actual_original_train_clips=5551,
        actual_original_train_families=len(sources), actual_generated_clips=0, original_VAL_motions_in_training=0,
        numerical_failures_unaccepted=0, source_binding=source_binding, manifest_sha256=sha(mp))
    return actual, rows
