"""Pure saved-counter analysis, then SHA-bound original-source enumeration.

No policy training, new physical evaluation, deletion or corpus acceptance.
"""
import hashlib
import json
import math
import pathlib


def summarize(rows, counters, expected_count=8278):
    assert len(rows) == len({r['canonical_source'] for r in rows}) == expected_count
    keys = ['raw_failed', 'raw_success', 'failed_motion_count', 'success_motion_count']
    assert set(counters) == set(keys) and all(len(counters[k]) == expected_count for k in keys)
    counts = dict(eligible_for_strict=0, above_relaxed_threshold=0, unattempted=0, invalid_denominator=0)
    decisions, by_split, by_dataset, rates, raw_rates, attempts = [], {}, {}, [], [], []
    for i, row in enumerate(rows):
        vals = [float(counters[k][i]) for k in keys]
        assert all(math.isfinite(x) and x >= 0 for x in vals)
        failed, success, ef, es = vals
        assert failed.is_integer() and success.is_integer(), 'Raw attempts must be integers'
        failed, success = int(failed), int(success)
        assert row['split'] in ('train', 'val') and type(row['frames']) is int and row['frames'] > 0
        assert math.isfinite(ef + es)
        rate = ef / (ef + es) if ef + es > 0 else None
        decision = ('unattempted' if failed + success == 0 else 'invalid_denominator' if rate is None
                    else 'above_relaxed_threshold' if rate > .05 else 'eligible_for_strict')
        counts[decision] += 1
        split = by_split.setdefault(row['split'], {k: 0 for k in counts}); split[decision] += 1
        dataset = row.get('dataset', row['canonical_source'].split('/')[0])
        group = by_dataset.setdefault(dataset, {k: 0 for k in counts}); group[decision] += 1
        attempts.append(failed + success)
        if rate is not None:
            rates.append(rate)
        if failed + success:
            raw_rates.append(failed / (failed + success))
        decisions.append(dict(global_index=i, canonical_source=row['canonical_source'], reference_sha256=row['sha256'],
            reference_frames=row['frames'], split=row['split'], raw_failed=failed, raw_success=success,
            ema_failed=ef, ema_success=es, relaxed_failure_probability=rate, decision=decision))
    def distribution(values):
        ordered = sorted(values)
        if not ordered:
            return None
        def quantile(q):
            p = (len(ordered) - 1) * q; lo = int(p); hi = min(lo + 1, len(ordered) - 1)
            return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)
        return dict(minimum=ordered[0], median=quantile(.5), p95=quantile(.95), maximum=ordered[-1], count=len(ordered))
    summary = dict(source_count=len(rows), counts=counts, by_split=by_split, by_dataset=by_dataset,
        relaxed_threshold=.05, relaxed_statistic='saved failed_motion_count / (failed_motion_count + success_motion_count)',
        ema_failure_distribution=distribution(rates), raw_failure_distribution=distribution(raw_rates),
        attempts_distribution=distribution(attempts), raw_attempts_total=sum(attempts),
        eligible_original_frames=sum(x['reference_frames'] for x in decisions if x['decision'] == 'eligible_for_strict'),
        strict_evaluation_completed=False, dataset_accepted=False, semantic_quality_accepted=False,
        original_assets_deleted=0, new_physics_steps=0, new_optimizer_updates=0)
    return summary, decisions


def reference_rows(root, expected_fingerprint, sha):
    plan = json.loads((pathlib.Path(root) / 'configs/filter-training-plan.json').read_text(encoding='utf-8'))
    rows = []
    for name in plan['input_reference_exports']:
        folder = pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references') / name
        receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
        assert receipt['status'] == 'TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
        assert sha(folder / 'manifest.private.jsonl') == receipt['private_manifest_sha256']
        rows.extend(r for r in map(json.loads, (folder / 'manifest.private.jsonl').read_text(encoding='utf-8').splitlines()) if not r['mirrored'])
    assert len(rows) == len({r['canonical_source'] for r in rows}) == 8278
    fingerprint = hashlib.sha256(json.dumps([{'sha256': r['sha256'], 'frames': r['frames']} for r in rows],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert fingerprint == expected_fingerprint
    return rows
