"""Trace observed construction quality into selected-reference replays.

Post-hoc diagnostic only. No alternative selection, controller, scoring,
generation, simulation, candidate-bank efficacy or calibration on test labels.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/condition_feedback_formal_v28_20261012'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    certificates = sorted((ROOT / 'results/execution_lineage_guard_v39_20261012').glob('certificate_*.json'))
    assert certificates
    certificate = read(certificates[-1])
    assert certificate['status'] == 'PASS' and certificate['checked_batches'] == 74
    raw = BASE / 'public/raw'
    if not raw.exists():
        raw = BASE / 'raw'
    bindings, construction_positive, rows = {}, [], []
    totals = dict(validation=dict(attempts=0, completed=0, joint_pass=0),
                  test_construction=dict(attempts=0, completed=0, joint_pass=0))
    for entry in certificate['entries']:
        path = raw / entry['label']
        bound = {}
        for name in ('metadata.json', 'metrics.json'):
            assert sha(path / name) == entry['output_hashes'][name]
            bound[name] = sha(path / name)
        bindings[entry['label']] = bound
        if '_eval_' in entry['label']:
            continue
        scores = read(path / 'metrics.json')
        group = 'validation' if entry['label'].startswith('val_') else 'test_construction'
        for score in scores:
            for key in ('completed', 'joint_pass'):
                totals[group][key] += int(score.get(key, False))
            totals[group]['attempts'] += 1
            if score.get('joint_pass', False):
                assert score['completed']
                construction_positive.append(dict(batch=entry['label'], case_id=score['case_id'],
                    source_id=score['source_id'], family=score['family'], offset_m=score['offset_m']))
    for selected_path in sorted((BASE / 'selected').glob('*.json')):
        if selected_path.name.endswith('.lineage.json'):
            continue
        selected = read(selected_path)
        stem = selected_path.stem
        arm, seed = stem.rsplit('_s', 1)
        replay_path = raw / ('test_eval_' + stem) / 'metrics.json'
        replay = read(replay_path)
        assert len(replay) == len(selected['choices']) == 144
        for choice, executed in zip(selected['choices'], replay):
            before = choice['metrics']
            assert choice['case_id'] == before['case_id'] == executed['case_id']
            if choice['construction_complete'] and before.get('joint_pass', False):
                batch = Path(choice['path']).name
                original = [r for r in read(raw / batch / 'metrics.json') if r['case_id'] == choice['case_id']]
                assert len(original) == 1 and original[0] == before
                rows.append(dict(arm=arm, seed=int(seed), case_id=choice['case_id'], batch=batch,
                    source_type=selected['source_type'], construction_joint_pass=True,
                    replay_joint_pass=executed['joint_pass'], replay_completed=executed['completed'],
                    replay_condition_pass=executed['condition_pass'], replay_physical_pass=executed['physical_pass'],
                    replay_tracking_pass=executed['tracking_pass'],
                    construction_endpoint_error_m=before['endpoint_error_m'], replay_endpoint_error_m=executed['endpoint_error_m'],
                    construction_slip_proxy_fraction=before['surface_slip_proxy_fraction'],
                    replay_slip_proxy_fraction=executed['surface_slip_proxy_fraction'],
                    construction_jerk_p95_m_s3=before['jerk_p95_m_s3'], replay_jerk_p95_m_s3=executed['jerk_p95_m_s3']))
        bindings[str(selected_path.relative_to(BASE))] = sha(selected_path)
    result = dict(status='COMPLETE_READ_ONLY_CONSTRUCTION_REPLAY_TRANSFER', time=time.time(),
        source_sha256=sha(Path(__file__)), certificate_sha256=sha(certificates[-1]), input_bindings=bindings,
        totals=totals, construction_joint_cases=construction_positive,
        selected_construction_joint_attempts=len(rows), selected_joint_lost_on_replay=sum(not row['replay_joint_pass'] for row in rows),
        selected_transfers=rows, causal_failure_attribution=False,
        simulation_calls=0, generation_calls=0, historical_scores_changed=False,
        c2_efficacy_tested=False, c3_efficacy_tested=False, innovation_confirmed=False, overleaf_results_sync=False,
        scope='Observed selection/replay transfer only; test labels cannot calibrate a proposed allocator')
    out = ROOT / 'results/construction_replay_transfer_v40_20261012'
    out.mkdir(exist_ok=True)
    target = out / ('diagnostic_%d.json' % time.time_ns())
    with target.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({key: result[key] for key in ('totals', 'selected_construction_joint_attempts', 'selected_joint_lost_on_replay', 'selected_transfers')}, indent=2))
    print(target)


if __name__ == '__main__':
    main()
