"""Apply acceptance checks to fresh actual simulator recordings, not generation.

The 71D native reference states use dataset root/joint states; body-position
targets are captured commands. No motion/semantic generator is invoked.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reimplementation'))
from physics_prefix import Motion, Execution, assess_execution


def main(capture, output):
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((ROOT/'configs/prefix_diagnostic_plan.json').read_text())
    metadata = json.loads((capture/'feedback_metadata.json').read_text())
    metrics = json.loads((capture/'metrics.json').read_text())
    assert metrics['checkpoint_sha256'] == plan['parent_sha256']
    assert metrics['num_envs'] == plan['num_envs'] and metrics['seed'] == plan['seed']
    assert Path(metrics['motion_file']).name == Path(plan['clip']).name
    for filename, key in [('raw_pre_reset.npz','raw_sha256'),('initial_state.npz','initial_state_sha256')]:
        assert hashlib.sha256((capture/filename).read_bytes()).hexdigest() == metadata[key]
    with np.load(capture/'raw_pre_reset.npz', allow_pickle=False) as d:
        raw = {k:d[k] for k in d.files}
    with np.load(capture/'initial_state.npz', allow_pickle=False) as d:
        initial = {k:d[k] for k in d.files}
    with np.load(ROOT/plan['clip'], allow_pickle=False) as d:
        reference = {k:d[k] for k in d.files}
    assert hashlib.sha256((ROOT/plan['clip']).read_bytes()).hexdigest() == metrics['motion_sha256']
    root_index = reference['body_names'].tolist().index('pelvis')
    body_indices = [reference['body_names'].tolist().index(name) for name in metadata['body_names']]
    count = plan['window_frames']
    state_ref = np.concatenate([reference['body_pos_w'][:count,root_index],
        reference['body_quat_w'][:count,root_index], reference['body_lin_vel_w'][:count,root_index],
        reference['body_ang_vel_w'][:count,root_index], reference['joint_pos'][:count],
        reference['joint_vel'][:count]], axis=-1)
    def motion(states, positions):
        return Motion(states, positions, metadata['fps'], metadata['schema'], tuple(metadata['body_names']))
    records = []
    for eid in range(plan['num_envs']):
        select = (raw['env_id']==eid) & (raw['frame']<count)
        frame = raw['frame'][select]
        np.testing.assert_array_equal(frame, np.arange(len(frame)))
        prefix = motion(initial['native_state'][eid:eid+1], initial['body_positions'][eid:eid+1])
        candidate = prefix.append(motion(state_ref, reference['body_pos_w'][:count,body_indices]))
        # Ensure the command target was really the dataset sequence, before acceptance.
        np.testing.assert_allclose(raw['reference_body_pos_local'][select], candidate.body_positions[1:len(frame)+1], atol=2e-5, rtol=0)
        actual = motion(raw['native_state'][select], raw['robot_body_pos_local'][select])
        execution = Execution(actual, raw['terminated'][select], prefix.states[-1], candidate.digest())
        decision = assess_execution(candidate, prefix, execution, plan['tolerance_m'])
        decision['env_id'] = eid
        # This is offline prefix construction, not a second generated/physical window.
        decision['next_prefix_sha256'] = prefix.append(actual).digest() if decision['accepted'] else prefix.digest()
        records.append(decision)
    report = {'status':'COMPLETED_FEEDBACK_GATE_DIAGNOSTIC', 'evaluation_type':'simulation_only',
        'source':'fresh dataset-reference replay; 32 randomized first episodes of one training clip',
        'plan':plan, 'generated_candidates':False, 'online_rolling_simulator_adapter':False,
        'terminal_suffix_tested_in_simulation':False, 'ppo_updates':0,
        'episodes':len(records), 'accepted':sum(r['accepted'] for r in records),
        'rejected':sum(not r['accepted'] for r in records), 'records':records,
        'capture_metadata_sha256':hashlib.sha256((capture/'feedback_metadata.json').read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'reimplementation/physics_prefix.py', ROOT/'configs/prefix_diagnostic_plan.json']}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    main(args.capture,args.output)
