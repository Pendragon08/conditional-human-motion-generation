"""Actual PRIVATE heldout metadata scope after closed100k CPU/fresh acceptance.

Deserializes signed JSON metadata only; upstream acceptance also checks immutable
checkpoint/inference file SHA without loading models. Source array hashes are
bound here and must be rechecked by the later model executor.
"""
import argparse
import json
import os
import pathlib
import shutil
import time
import traceback
from ldm100k_verified_input import ROOT, read, sha, owned, accepted_verified_input
from ldm100k_heldout_protocol import make_segment_plan
from ldm100k_scope_contract import RUN, validate_policy, validate_scope_receipt

SHM = pathlib.Path('/dev/shm/ouyanghao_textop_20261025')


def own_input(path):
    p = pathlib.Path(path)
    roots = [x for x in [ROOT, SHM] if p == x or x in p.parents]
    assert len(roots) == 1
    return owned(p, roots[0])


def private_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def write_private(path, data):
    raw = private_bytes(data)
    with path.open('xb') as handle:
        os.chmod(path, 0o600)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(path=str(path), bytes=len(raw), sha256=sha(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    assert a.run == RUN and os.getuid() == 1013
    assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3} and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sp = owned(ROOT / 'configs/ldm100k-heldout-scope-spec.json')
    s = read(sp)
    validate_policy(s)
    assert sha(__file__) == s['executor_sha256']
    for n, h in s['bound_sources'].items():
        assert sha(ROOT / n) == h, n
    verified = accepted_verified_input(s['verified_model'])
    source = verified['export']['stopped']['source_spec']
    cs = read(ROOT / 'configs/filtered15000-corpus-spec.json')
    assert sha(ROOT / 'configs/filtered15000-corpus-spec.json') == s['corpus_spec_sha256']
    assert cs['candidate_view'] == source['corpus_dir']
    assert cs['strict_audit_receipt_sha256'] == source['strict_audit_receipt_sha256']
    accepted_path = owned(ROOT / 'receipts/filtered15000_data_acceptance_v1.json')
    assert sha(accepted_path) == source['data_acceptance_sha256']
    accepted_data = read(accepted_path)
    assert accepted_data['actual_strict_filtered_training_data_accepted']
    assert accepted_data['corpus_receipt_sha256'] == source['corpus_receipt_sha256']
    view = own_input(cs['candidate_view'])
    vp = own_input(view / 'view_receipt.json')
    assert sha(vp) == source['corpus_receipt_sha256']
    meta = read(vp)
    assert meta['fit_split'] == 'train' and meta['validation_variants_removed'] == 0
    assert meta['source_sequences_in_view'] == 7135 and meta['motion_variants'] == 14270
    assert meta['normalization_scope'] == 'ACTUAL_STRICT15000_PASSES_TRAIN_ONLY_REFIT'
    out = ROOT / 'ldm100k_user_handoff_20261028' / a.run
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    rec = dict(status='STARTING_LDM100K_HELDOUT_SCOPE', run=a.run, source_sha256=sha(__file__),
               spec_sha256=sha(sp), output_writers_closed=False, actual_array_deserializations=0,
               actual_model_loads=0, actual_gpu_contexts=0, actual_generation_runs=0,
               semantic_metrics_computed=False, optimizer_updates=0,
               verified_model_receipt_sha256=verified['receipt_sha256'])

    def save():
        rec['elapsed_seconds'] = time.monotonic() - started
        temp = out / 'receipt.partial'
        with temp.open('w', encoding='utf-8') as handle:
            json.dump(rec, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(out / 'receipt.json')
        fd = os.open(str(out), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    save()
    try:
        from filtered15000_inputs import load_selection, select_prepared_rows
        audit, selected = load_selection(cs)  # existing closed decisions, no trace replay
        rows, manifests = [], {}
        for name in cs['prepared_manifests']:
            path = own_input(name)
            assert sha(path) == cs['prepared_manifest_sha256'][name]
            manifests[name] = sha(path)
            rows.extend(json.loads(x) for x in path.read_text(encoding='utf-8').splitlines())
        assert manifests == meta['prepared_manifest_hashes']
        rows = select_prepared_rows(rows, selected)
        qp = own_input(view / 'cross_split_quarantine.private.json')
        assert sha(qp) == meta['artifacts'][qp.name]['sha256']
        quarantine = read(qp)
        assert all(x['reason'] == 'exact_float32_pose_overlap_with_validation' for x in quarantine)
        quarantined = {x['canonical_source'] for x in quarantine}
        train = {x['canonical_source'] for x in rows if x['split'] == 'train'} - quarantined
        val = [dict(x, mirrored=False, physical_filter_passed=True, fps=50) for x in rows if x['split'] == 'val']
        assert quarantined <= {x['canonical_source'] for x in rows if x['split'] == 'train'}
        assert len(train) + len(val) == meta['source_sequences_in_view']
        witnesses = accepted_data['actual_loader_witnesses']
        assert len(witnesses) == 2 and {w['mode'] for w in witnesses} == {'build', 'verify'}
        for witness in witnesses:
            assert witness['actual_exit_code'] == 0 and witness['source_sequences'] == len(selected)
            checks = witness['loader_checks']
            assert [(x['split'], x['weighted']) for x in checks] == [('train', False), ('val', False), ('train', True)]
            assert checks[0]['raw_sequences'] == checks[2]['raw_sequences'] == len(train) * 2
            assert checks[1]['raw_sequences'] == len(val) * 2
        assert len(quarantined) == meta['quarantined_train_source_families']
        plan = read(ROOT / 'configs/ldm100k-heldout-plan.json')
        assert sha(ROOT / 'configs/ldm100k-heldout-plan.json') == s['heldout_plan_sha256']
        assert plan['scope_build_authorized'] is False and plan['generation_authorized'] is False
        diagnostic = make_segment_plan(val, train, plan['first_diagnostic_family_limit'])
        complete = make_segment_plan(val, train, None)
        index = []
        for row in val:
            path = own_input(row['output_file'])
            assert path.is_file()
            index.append(dict(canonical_source=row['canonical_source'], split='val', mirrored=False,
                              frames50=row['frames50'], output_file=str(path), output_sha256=row['output_sha256']))
        size = sum(len(private_bytes(x)) for x in [diagnostic, complete, index])
        assert size <= s['metadata_upper_bound_bytes'], 'Retain real input; do not silently truncate evaluation scope'
        assert shutil.disk_usage(out).free >= s['reserve_bytes'] + s['metadata_upper_bound_bytes'] + s['log_allowance_bytes']
        artifacts = {}
        for name, data in [('diagnostic_scope.private.json', diagnostic),
                           ('validation_scope.private.json', complete), ('source_index.private.json', index)]:
            artifacts[name] = write_private(out / name, data)
        fd = os.open(str(out), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        assert sha(vp) == source['corpus_receipt_sha256']
        rec.update(status='USER_LDM100K_HELDOUT_METADATA_SCOPE_BUILT', output_writers_closed=True,
                   artifacts=artifacts, diagnostic_summary=diagnostic['summary'], validation_summary=complete['summary'],
                   actual_validation_original_families=len(val), actual_train_families=len(train),
                   actual_quarantined_train_families=len(quarantined), prepared_manifest_sha256=manifests,
                   source_corpus_receipt_sha256=source['corpus_receipt_sha256'],
                   data_acceptance_sha256=source['data_acceptance_sha256'],
                   strict_audit_receipt_sha256=source['strict_audit_receipt_sha256'],
                   normalization_sha256_bound=meta['artifacts']['meanstd.pkl']['sha256'],
                   array_bytes_revalidated_here=False, source_array_hash_recheck_required_before_generation=True,
                   training_data_or_model_modified=False, generation_inputs_numerically_accepted=False,
                   private_metadata_only=True)
        validate_scope_receipt(rec, s, sha(sp))
        save()
        print(json.dumps(rec), flush=True)
    except BaseException as error:
        rec.update(status='FAILED_LDM100K_HELDOUT_SCOPE', error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
