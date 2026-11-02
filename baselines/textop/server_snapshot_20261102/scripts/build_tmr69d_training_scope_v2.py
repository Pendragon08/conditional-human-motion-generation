"""Signed PRIVATE TRAIN-only BABEL annotation scope, without array/model loading."""
import argparse
import decimal
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback
from ldm100k_verified_input import ROOT, read, sha, owned
from filtered15000_inputs import load_selection, select_prepared_rows

SHM = pathlib.Path('/dev/shm/ouyanghao_textop_20261025')


def own_input(path):
    p = pathlib.Path(path)
    roots = [x for x in [ROOT, SHM] if p == x or x in p.parents]
    assert len(roots) == 1
    return owned(p, roots[0])


def frame_window(start_s, end_s, features):
    first, last = decimal.Decimal(str(start_s)), decimal.Decimal(str(end_s))
    assert first.is_finite() and last.is_finite() and 0 <= first < last
    assert last <= decimal.Decimal(features + 1) / 50
    return max(0, int((first * 50).to_integral_value(rounding=decimal.ROUND_FLOOR))), min(features, int((last * 50).to_integral_value(rounding=decimal.ROUND_CEILING)))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
assert sha(__file__) == s['executor_sha256']
assert sha(ROOT / 'scripts/filtered15000_inputs.py') == s['inputs_helper_sha256']
assert sha(ROOT / 'configs/filtered15000-corpus-spec.json') == s['corpus_spec_sha256']
cs = read(ROOT / 'configs/filtered15000-corpus-spec.json')
view = own_input(cs['candidate_view'])
vp = own_input(view / 'view_receipt.json')
assert sha(vp) == s['corpus_receipt_sha256']
meta = read(vp)
assert meta['fit_split'] == 'train' and meta['source_sequences_in_view'] == 7135 and meta['motion_variants'] == 14270
assert meta['normalization_scope'] == 'ACTUAL_STRICT15000_PASSES_TRAIN_ONLY_REFIT'
ap = owned(ROOT / 'receipts/filtered15000_data_acceptance_v1.json')
assert sha(ap) == s['data_acceptance_sha256']
accept = read(ap)
assert accept['actual_strict_filtered_training_data_accepted'] and accept['corpus_receipt_sha256'] == sha(vp)
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
rec = dict(status='STARTING_TMR69D_TRAINING_METADATA_SCOPE', run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    output_writers_closed=False, actual_model_loads=0, actual_array_deserializations=0, optimizer_updates=0,
    training_authorized=False, original_validation_families_used_for_training=False)


def save():
    rec['elapsed_seconds'] = time.monotonic() - started
    tmp = out / 'receipt.partial'
    with tmp.open('w') as f:
        json.dump(rec, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out / 'receipt.json')


save()
try:
    audit, selection = load_selection(cs)
    assert len(selection) == 7137 and audit['strict_result_accepted']
    raw, manifests = [], {}
    for name in cs['prepared_manifests']:
        path = own_input(name)
        assert sha(path) == cs['prepared_manifest_sha256'][name]
        manifests[name] = sha(path)
        raw.extend(json.loads(x) for x in path.read_text(encoding='utf-8').splitlines())
    assert manifests == meta['prepared_manifest_hashes']
    rows = select_prepared_rows(raw, selection)
    qp = own_input(view / 'cross_split_quarantine.private.json')
    assert sha(qp) == meta['artifacts'][qp.name]['sha256']
    quarantine = read(qp)
    assert len(quarantine) == 4 and len({x['canonical_source'] for x in quarantine}) == meta['quarantined_train_source_families'] == 2 and all(x['reason'] == 'exact_float32_pose_overlap_with_validation' for x in quarantine)
    excluded = {x['canonical_source'] for x in quarantine}
    train = [r for r in rows if r['split'] == 'train' and r['canonical_source'] not in excluded]
    original_val = {r['canonical_source'] for r in rows if r['split'] == 'val'}
    assert len(train) == 5361 and len(original_val) == 1774
    train_set = {r['canonical_source'] for r in train}
    assert len(train_set) == len(train) and not train_set & original_val and not train_set & excluded
    for witness in accept['actual_loader_witnesses']:
        assert witness['actual_exit_code'] == 0
        checks = witness['loader_checks']
        assert checks[0]['raw_sequences'] == checks[2]['raw_sequences'] == len(train) * 2
        assert checks[1]['raw_sequences'] == len(original_val) * 2
    order = sorted(train, key=lambda r: hashlib.sha256(('tmr69d_internal_split_v1\0' + r['canonical_source']).encode()).hexdigest())
    internal_holdout = {r['canonical_source'] for r in order[:268]}
    assert len(internal_holdout) == 268
    samples, omissions, texts, sources = [], [], {}, []
    for r in order:
        assert r['dof'] == 29 and r['features'] == 69 and type(r['frames50']) is int and r['frames50'] >= 2
        path = own_input(r['output_file'])
        split = 'internal_validation' if r['canonical_source'] in internal_holdout else 'gradient_train'
        sources.append(dict(canonical_source=r['canonical_source'], split=split, original_split='train', mirrored=False,
            output_file=str(path), output_sha256=r['output_sha256'], frames50=r['frames50']))
        for ai, ann in enumerate(r['annotations']):
            assert isinstance(ann['text'], str) and ann['text'].strip()
            first, last = frame_window(ann['start_s'], ann['end_s'], r['frames50'] - 1)
            if first >= last:
                omissions.append(dict(canonical_source=r['canonical_source'], annotation_index=ai, reason='empty_T_minus_1_annotation_window', physical_failure_claimed=False))
                continue
            text_id = hashlib.sha256(ann['text'].encode()).hexdigest()
            if text_id in texts:
                assert texts[text_id] == ann['text']
            texts[text_id] = ann['text']
            for first_chunk in range(first, last, 200):
                end = min(first_chunk + 200, last)
                key = dict(canonical_source=r['canonical_source'], annotation_index=ai, start_feature=first_chunk,
                    end_feature_exclusive=end, text_id=text_id)
                samples.append(dict(key, sample_id=hashlib.sha256(canonical(key)).hexdigest(), split=split,
                    feature_frames=end-first_chunk, original_split='train', mirrored=False))
    assert samples and len(samples) == len({x['sample_id'] for x in samples})
    assert all(1 <= x['feature_frames'] <= 200 and x['canonical_source'] in train_set for x in samples)
    assert all(x['canonical_source'] not in original_val and x['canonical_source'] not in excluded for x in samples)
    norm = own_input(view / 'meanstd.pkl')
    assert sha(norm) == meta['artifacts']['meanstd.pkl']['sha256']
    summary = dict(original_train_families=5361, gradient_train_families=5093, internal_validation_families=268,
        original_validation_families_excluded=1774, quarantined_train_families_excluded=2,
        sources_without_any_nonempty_annotation=len(train_set - {x['canonical_source'] for x in samples}),
        samples=len(samples), gradient_train_samples=sum(x['split']=='gradient_train' for x in samples),
        internal_validation_samples=sum(x['split']=='internal_validation' for x in samples), unique_texts=len(texts),
        feature_frames=sum(x['feature_frames'] for x in samples), omissions=len(omissions), fps=50, feature_dim=69,
        source_arrays_deserialized=0, source_array_bytes_rehash_required_before_numeric_use=True,
        mirror_augmentation_used=False, mirror_note='Original TRAIN sources only; generator mirror data retained unchanged',
        normalization_scope='frozen accepted generator TRAIN-only transform, including original TRAIN families reserved for internal checks',
        internal_validation_use='No generated-val tuning or early stopping authorized; fixed training policy still requires separate acceptance',
        feature_boundaries='floor(start_s*50),ceil(end_s*50),0<=start<end<=T-1; chunks<=200',
        overlapping_annotations='preserved independently, no invented transition order')
    data = dict(summary=summary, sources=sources, samples=samples, texts=texts, omissions=omissions,
        normalization=dict(path=str(norm),sha256=sha(norm)), prepared_manifest_sha256=manifests)
    blob = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode()
    assert len(blob) < s['output_upper_bytes']
    assert shutil.disk_usage(out).free > 2 * 1024**3 + 128 * 1024**2 + len(blob)
    artifact = out / 'training_scope.private.json'
    with artifact.open('xb') as f:
        os.chmod(artifact, 0o600)
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    assert sha(vp) == s['corpus_receipt_sha256'] and sha(ap) == s['data_acceptance_sha256']
    rec.update(status='TMR69D_TRAINING_METADATA_SCOPE_BUILT', output_writers_closed=True,
        summary=summary, private_artifact=dict(path=str(artifact),bytes=len(blob),sha256=sha(artifact)),
        accepted_data_sha256=sha(ap), corpus_receipt_sha256=sha(vp), normalization_sha256=sha(norm), prepared_manifest_sha256=manifests)
    save()
except BaseException as error:
    rec.update(status='FAILED_TMR69D_TRAINING_METADATA_SCOPE_ALL_OUTPUTS_RETAINED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise
