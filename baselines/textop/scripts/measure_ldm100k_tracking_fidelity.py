"""New position-fidelity metrics from immutable recorded poses; zero dynamics.

Reconstructs executed link poses with the same real Isaac articulation as the
reference FK. No policy/model is loaded, no trajectory is replayed physically.
The declared target timing is initial0, then consumed targets0..steps-1.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback
from isaaclab.app import AppLauncher

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
HOME = pathlib.Path('/home/ouyanghao')
p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_bytes())


def owned(path, root):
    p = pathlib.Path(path)
    assert p.is_absolute() and p.resolve() == p and (root == p or root in p.parents)
    assert p.stat().st_uid == 1013 and not p.is_symlink()
    return p


assert os.getuid() == 1013 and a.headless and a.device == 'cuda:1'
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
request = owned(a.request, ROOT)
s = read(request)
assert sha(__file__) == s['executor_sha256']
for name, digest in s['bound_sources'].items():
    assert sha(owned(ROOT / name, ROOT)) == digest, name
for item in s['physics_binding'].values():
    assert sha(item['path']) == item['sha256']
    proof = read(item['path'])
    assert proof.get('actual_returncode', 0) == proof.get('child_exit_code', 0) == 0
    assert proof.get('supervision_passed', True)
prp = owned(s['physics_binding']['receipt']['path'], HOME)
pr = read(prp)
assert pr['status'] == 'USER_LDM100K_GENERATED_STRICT_PHYSICS_FINISHED' and pr['output_writers_closed']
index = owned(prp.parent / 'executed_traces.private.json', HOME)
assert sha(index) == pr['private_executed_trace_index_sha256']
traces = read(index)
assert len(traces) == 1880
fkpath = owned(s['fk_receipt']['path'], HOME)
assert sha(fkpath) == s['fk_receipt']['sha256']
fk = read(fkpath)
mp = owned(fk['private_manifest']['path'], HOME)
assert sha(mp) == fk['private_manifest']['sha256']
references = {(x['segment_id'], x['seed'], x['mode']): x for x in map(json.loads, mp.read_text().splitlines())}
assert len(references) == 1880
warm = read(ROOT / 'configs/tracker-env-spec.json')
assert sha(warm['robot_urdf']) == warm['robot_urdf_sha256']
out = owned(s['output_root'], HOME) / s['run']
out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
rec = dict(status='STARTING_RECORDED_TRACKING_FIDELITY', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(request), physics_receipt_sha256=sha(prp), processed_trials=0, declared_trials=1880,
    new_physics_steps=0, new_policy_rollouts=0, neural_model_loads=0, optimizer_updates=0,
    output_writers_closed=False, threshold_m=.3, semantic_metrics_computed=False)
app = sim = None


def save():
    rec['elapsed_seconds'] = time.monotonic() - started
    temp = out / 'receipt.partial'
    with temp.open('w') as f:
        json.dump(rec, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(out / 'receipt.json')


save()
try:
    assert shutil.disk_usage(out).free > s['reserve_bytes'] + s['log_allowance_bytes'] + s['output_upper_bytes']
    app = AppLauncher(a).app
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import yaw_quat, quat_mul, quat_inv, quat_apply
    from textop_tracker.robots.g1 import G1_CYLINDER_CFG
    from textop_tracker.tasks.tracking.config.g1.flat_env_cfg import G1FlatPrivPrivObsEnvCfg
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.65, torch.device(a.device))
    robot_cfg = G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    robot_cfg.spawn.asset_path = warm['robot_urdf']
    robot_cfg.spawn.usd_dir = str(out / 'usd')
    robot_cfg.spawn.usd_file_name = 'g1.usd'

    @configclass
    class Scene(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=a.device))
    scene = InteractiveScene(Scene(num_envs=128, env_spacing=2.5))
    sim.reset()
    robot = scene['robot']
    assert len(robot.joint_names) == 29 and len(robot.body_names) == 30
    key_names = list(G1FlatPrivPrivObsEnvCfg().commands.motion.body_names)
    assert len(key_names) == 14 and key_names[0] == 'pelvis'
    key_ids = [robot.body_names.index(n) for n in key_names]
    pelvis = robot.body_names.index('pelvis')
    rec.update(all_link_names=list(robot.body_names), tracked_link_names=key_names,
               output_position_units='mm', threshold_group='14_tracked_links_including_pelvis',
               global_alignment='fixed initial pelvis yaw and XY; reference initial height retained',
               target_timing='initial0 excluded from aggregates; subsequent state k compares consumed reference k-1')
    result_rows = []
    poses = 0
    endpoint_max_difference = 0.
    for trace in traces:
        assert shutil.disk_usage(out).free > s['reserve_bytes'] + s['log_allowance_bytes']
        key = trace['segment_id'], trace['seed'], trace['mode']
        row = references[key]
        folder = owned(trace['folder'], HOME)
        meta = read(folder / 'trace.private.json')
        source = owned(folder / 'qpos.private.bin', HOME)
        assert sha(source) == meta['binary_sha256'] == trace['binary_sha256']
        assert meta['status'] == 'FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind'] == 'isaaclab_physics'
        q = np.fromfile(source, dtype='<f4').reshape(meta['samples'], 36)
        assert len(q) == meta['trial']['control_steps'] + 1 and np.isfinite(q).all()
        order = [meta['joint_names'].index(n) for n in robot.joint_names]
        positions = []
        for start in range(0, len(q), 128):
            n = min(128, len(q) - start)
            indices = np.minimum(np.arange(128) + start, len(q) - 1)
            roots = robot.data.default_root_state.clone()
            roots[:, :3] = torch.as_tensor(q[indices, :3], device=a.device) + scene.env_origins
            roots[:, 3:7] = torch.as_tensor(q[indices, 3:7], device=a.device)
            roots[:, 7:] = 0
            jp = torch.as_tensor(q[indices, 7:][:, order], device=a.device)
            robot.write_root_state_to_sim(roots)
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            sim.forward()
            scene.update(.005)
            assert torch.allclose(robot.data.joint_pos, jp, atol=1e-5)
            positions.append((robot.data.body_pos_w - scene.env_origins[:, None, :])[:n].cpu())
        actual = torch.cat(positions)
        assert torch.isfinite(actual).all()
        ref_file = owned(row['file'], HOME)
        assert sha(ref_file) == row['sha256']
        assert row['robot_body_names'] == list(robot.body_names)
        with np.load(ref_file, allow_pickle=False) as data:
            reference = torch.from_numpy(data['body_pos_w'].copy())
            initial_quat = torch.from_numpy(data['body_quat_w'][0, pelvis].copy())
        rotation = yaw_quat(quat_mul(torch.from_numpy(q[0, 3:7].copy()), quat_inv(initial_quat)))
        reference_anchor = reference[0, pelvis].clone()
        actual_anchor = actual[0, pelvis].clone()
        actual_anchor[2] = reference_anchor[2]
        target = actual_anchor + quat_apply(rotation.expand(reference.numel() // 3, 4),
            (reference - reference_anchor).reshape(-1, 3)).reshape_as(reference)
        # Match every independently recorded endpoint metric before accepting
        # a different per-link aggregation or any paper-inspired success rate.
        last_error = torch.linalg.vector_norm(actual[-1, key_ids] - target[-1, key_ids], dim=-1).mean().item()
        expected_endpoint = trace['final_reference_endpoint_position']['mean_14_body_m']
        difference = abs(last_error - expected_endpoint)
        endpoint_max_difference = max(endpoint_max_difference, difference)
        assert difference < 1e-4, (key, last_error, expected_endpoint)
        steps = len(q) - 1
        target = target[:steps]  # executed states1..steps consumed targets0..steps-1
        observed = actual[1:]
        groups = {}
        for label, ids in [('tracked14', key_ids), ('all30', list(range(30)))]:
            a0, b0 = observed[:, ids], target[:, ids]
            global_error = torch.linalg.vector_norm(a0 - b0, dim=-1)
            rel = torch.linalg.vector_norm((a0 - observed[:, pelvis:pelvis+1]) -
                                          (b0 - target[:, pelvis:pelvis+1]), dim=-1)
            maximum = rel.mean(dim=1).max().item()
            groups[label] = dict(observed_global_mpjpe_mm=global_error.mean().item()*1000,
                observed_root_relative_mpjpe_mm=rel.mean().item()*1000,
                max_frame_root_relative_mean_m=maximum, observed_prefix_within_threshold=maximum <= .3,
                full_coverage_and_within_threshold=bool(meta['trial']['complete'] and maximum <= .3))
        result_rows.append(dict(segment_id=key[0], seed=key[1], mode=key[2], executed_control_steps=steps,
            reference_frames=row['frames'], full_T_minus_1_coverage=meta['trial']['complete'],
            physical_failure=meta['trial']['failed'], groups=groups))
        poses += len(q)
        rec['processed_trials'] += 1
        if rec['processed_trials'] % 100 == 0:
            save()
    assert len(result_rows) == 1880 and poses == 195105
    aggregates = {}
    for mode in ['conditional_cfg5', 'unconditional']:
        rr = [x for x in result_rows if x['mode'] == mode]
        assert len(rr) == 940
        aggregates[mode] = {}
        for label in ['tracked14', 'all30']:
            values = [x['groups'][label] for x in rr]
            aggregates[mode][label] = dict(trials=940,
                full_coverage_successes=sum(x['full_coverage_and_within_threshold'] for x in values),
                full_coverage_success_rate=sum(x['full_coverage_and_within_threshold'] for x in values)/940,
                observed_prefix_only_within_threshold_count=sum(x['observed_prefix_within_threshold'] for x in values),
                trial_equal_global_mpjpe_mm=float(np.mean([x['observed_global_mpjpe_mm'] for x in values])),
                trial_equal_root_relative_mpjpe_mm=float(np.mean([x['observed_root_relative_mpjpe_mm'] for x in values])),
                observed_frame_weighted_global_mpjpe_mm=float(np.average([x['observed_global_mpjpe_mm'] for x in values],weights=[x['executed_control_steps'] for x in rr])),
                observed_frame_weighted_root_relative_mpjpe_mm=float(np.average([x['observed_root_relative_mpjpe_mm'] for x in values],weights=[x['executed_control_steps'] for x in rr])))
    pp = out / 'fidelity_trials.private.json'
    with pp.open('x') as f:
        os.chmod(pp, 0o600)
        json.dump(result_rows, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    assert sha(index) == pr['private_executed_trace_index_sha256'] and sha(mp) == fk['private_manifest']['sha256']
    rec.update(status='USER_LDM100K_RECORDED_TRACKING_FIDELITY_FINISHED', output_writers_closed=True,
        actual_recorded_pose_samples=poses, endpoint_max_difference_from_independent_capture_m=endpoint_max_difference,
        aggregate=aggregates, private_artifact=dict(path=str(pp), bytes=pp.stat().st_size, sha256=sha(pp)),
        original_strict_decisions_unchanged=True, early_failures_keep_observed_prefix_only=True,
        unobserved_frames_imputed=False, full_validation_or_online_quality_accepted=False,
        note='Position-only fidelity on the existing32-family diagnostic. Full-coverage plus0.3m root-relative threshold is separate from the original termination-only decision; fourteen tracked and all30 Isaac links are both explicit.')
    save()
except BaseException as error:
    rec.update(status='FAILED_RECORDED_TRACKING_FIDELITY', error=repr(error), traceback=traceback.format_exc())
    save()
    raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app, rec, save, out, simulation=sim,
              successful_statuses=('USER_LDM100K_RECORDED_TRACKING_FIDELITY_FINISHED',))
