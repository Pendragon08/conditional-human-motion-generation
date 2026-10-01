"""Aggregate every preregistered trial; do not discard failures or retune criteria."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/mujoco_robust_contact'


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((ROOT/'configs/mujoco_robustness_contact_protocol.json').read_text())
    conditions = [c['id'] for c in protocol['conditions']]
    summary = {'protocol': protocol, 'tasks': {}, 'videos': []}
    for task, count in protocol['references'].items():
        parts = [json.loads((OUT/f'{task}_shard_{s}.json').read_text()) for s in [0, 1]]
        assert all(p['status'] == 'COMPLETE' and p['protocol'] == protocol for p in parts)
        assert parts[0]['provenance'] == parts[1]['provenance']
        rows = [r for p in parts for r in p['trials']]
        assert len(rows) == count*len(conditions)
        lookup = {(r['condition'], r['reference_index']): r for r in rows}
        assert len(lookup) == len(rows)
        assert set(lookup) == {(c, i) for c in conditions for i in range(count)}
        for row in rows:
            assert all(math.isfinite(v) for v in row.values() if isinstance(v, float))
            if row['valid_initialization']:
                assert 1 <= row['executed_steps'] <= 500
                assert not (row['fallen'] and row['survived_horizon'])
                assert not row['stable_goal_success'] or (row['virtual_goal_reached'] and row['survived_horizon'])
                assert not row['stable_strike_success'] or (row['valid_strike_10n'] and row['survived_horizon'])
                assert not row['valid_strike_20n'] or row['valid_strike_10n']
                assert not row['valid_strike_10n'] or row['valid_strike_5n']
                intended = next(c.get('floor_friction', 1.) for c in protocol['conditions'] if c['id'] == row['condition'])
                assert all(abs(v-intended) < 1e-8 for v in row['observed_floor_contact_friction'])
        table = []
        for c in conditions:
            rs = [lookup[c, i] for i in range(count)]
            base = 'nominal_mass_constants_refresh' if c in ['mass_low', 'mass_high'] else 'nominal'
            item = {'condition': c, 'n': count, 'paired_control': base}
            for metric in ['valid_initialization', 'virtual_goal_reached', 'stable_goal_success', 'fallen',
                           'survived_horizon', 'push_exposed', 'any_contact', 'correct_effector_contact',
                           'window_correct_contact', 'window_front_contact', 'valid_strike_5n',
                           'valid_strike_10n', 'valid_strike_20n', 'stable_strike_success']:
                item[metric] = sum(bool(r.get(metric, False)) for r in rs)
            item['stable_goal_rate'] = item['stable_goal_success']/count
            item['paired_stable_goal_lost'] = sum(lookup[base, i]['stable_goal_success'] and not lookup[c, i]['stable_goal_success'] for i in range(count))
            item['paired_stable_goal_gained'] = sum(not lookup[base, i]['stable_goal_success'] and lookup[c, i]['stable_goal_success'] for i in range(count))
            item['push_impulse_norm_ns_unique'] = sorted(set(round(sum(v*v for v in r.get('push_impulse_world_ns', [0,0,0]))**.5, 5) for r in rs))
            item['observed_floor_contact_friction'] = sorted(set(v for r in rs for v in r.get('observed_floor_contact_friction', [])))
            positive = [r for r in rs if r.get('correct_effector_contact')]
            item['correct_contact_peak_force_median_n'] = statistics.median([r['peak_correct_normal_force_n'] for r in positive]) if positive else None
            item['correct_contact_impulse_median_ns'] = statistics.median([r['correct_impulse_ns'] for r in positive]) if positive else None
            item['force_stat_denominator_correct_contact'] = len(positive)
            table.append(item)
        nominal = [lookup['nominal', i] for i in range(count)]
        previous = json.loads((ROOT/f'results/sim2sim_mujoco/{task}_zero_shot.json').read_text())
        assert previous['provenance']['checkpoint_sha256'] == parts[0]['provenance']['checkpoint_sha256']
        previous_rows = {r['reference_index']: r for r in previous['trials']}
        nominal_disagreements = []
        for row in nominal:
            old = previous_rows[row['reference_index']]
            if (row['virtual_goal_reached'] != old['task_success_virtual_target'] or
                row['fallen'] != old['fallen'] or
                row['survived_horizon'] != (old['executed_steps'] == 500 and not old['fallen'])):
                nominal_disagreements.append(row['reference_index'])
        summary['tasks'][task] = {'n_references': count, 'trials': len(rows), 'conditions': table,
                                 'provenance': parts[0]['provenance'],
                                 'nominal_disagreements_with_previous': nominal_disagreements,
                                 'reference_npz_sha256': previous['reference_npz_sha256'],
                                 'policy_sha256': previous['policy_sha256'],
                                 'sum_worker_wall_s': sum(p['elapsed_wall_s'] for p in parts),
                                 'previous_nominal_result_sha256': file_sha(ROOT/f'results/sim2sim_mujoco/{task}_zero_shot.json')}
        for path in sorted((ROOT/'media/mujoco_robust_contact').glob(f'{task}_*.mp4')):
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                'format=duration:stream=width,height,nb_frames', '-of', 'json', str(path)]))
            assert float(probe['format']['duration']) > 0 and path.stat().st_size > 0
            summary['videos'].append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                                     'sha256': file_sha(path), 'ffprobe': probe})
    assert len(summary['videos']) == 12, len(summary['videos'])
    summary['status'] = 'COMPLETE'
    summary['total_trials'] = sum(t['trials'] for t in summary['tasks'].values())
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({task: data['conditions'] for task, data in summary['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
