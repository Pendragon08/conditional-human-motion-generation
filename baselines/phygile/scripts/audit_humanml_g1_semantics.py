"""Supplement the legacy consistency audit with all-frame MuJoCo semantics.

This checks an explicit reconstruction, not author-exact features or robot GT.
The nominal HumanML playback clock is not a proven source-capture clock.
"""
import argparse
import datetime
import json
from pathlib import Path, PurePosixPath
import sys
import traceback
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'reimplementation'))
from codec262 import G1Codec, SELECTED, FEET
from humanml_pairing import sha, quality
from audit_humanml_g1_dataset import audit as legacy_audit


def difference(a, fps):
    d = (a[1:] - a[:-1]) * fps
    return np.concatenate((d, d[-1:]), axis=0)


def root_rotation(native):
    return Rotation.from_quat(native[:, [4, 5, 6, 3]]).as_matrix()


def opposite(name):
    if name.startswith('left_'): return 'right_' + name[5:]
    if name.startswith('right_'): return 'left_' + name[6:]
    return name


def world_fk(model, data, native, names, addresses):
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
    if any(i < 0 for i in ids.values()): raise ValueError('Missing MuJoCo body')
    pos = {n: np.empty((len(native), 3)) for n in names}
    rot = {n: np.empty((len(native), 3, 3)) for n in names}
    for t in range(len(native)):
        data.qpos[:7] = native[t, :7]; data.qpos[addresses] = native[t, 13:42]
        mujoco.mj_forward(model, data)
        for n, i in ids.items():
            pos[n][t] = data.xpos[i]; rot[n][t] = data.xmat[i].reshape(3, 3)
    return pos, rot


def independent_features(native, pos, rot, codec, fps):
    r = root_rotation(native); xyz = native[:, :3]
    yaw = np.arctan2(r[0, 1, 0], r[0, 0, 0])
    c = Rotation.from_euler('z', -yaw).as_matrix()
    offset = xyz[0].copy(); offset[2] = 0
    canonical = (xyz-offset) @ c.T; cr = c @ r
    local = np.stack([np.einsum('tji,tj->ti', r, pos[n]-xyz) for n in SELECTED], 1)
    vel = np.stack([np.einsum('tji,tj->ti', r, difference(pos[n], fps))
                    for n in ('pelvis',)+SELECTED], 1)
    angular = Rotation.from_matrix(cr[1:] @ cr[:-1].transpose(0, 2, 1)).as_rotvec()*fps
    angular = np.concatenate((angular, angular[-1:]), 0)
    relative = np.stack([rot[codec.movable[n]['parent']].transpose(0, 2, 1)
                          @ rot[codec.movable[n]['child']] for n in codec.joint_names], 1)
    six = np.concatenate((relative[:, :, :, 0], relative[:, :, :, 1]), axis=-1)
    feet = np.stack([(pos[n][:, 2] < .05) &
         (np.linalg.norm(difference(pos[n], fps)[:, :2], axis=-1) < .01) for n in FEET], 1)
    hands = np.stack([pos[s+'_rubber_hand'][:, 2] < .10 for s in ('left', 'right')], 1)
    return np.concatenate((angular, difference(canonical, fps), canonical[:, 2:3],
        local.reshape(len(native), 36), six.reshape(len(native), 174),
        vel.reshape(len(native), 39), feet, hands), 1)


def check(a):
    inventory = json.loads(a.inventory.read_text())
    actual = {p.relative_to(a.dataset).as_posix() for p in a.dataset.rglob('*') if p.is_file()}
    if actual != {r['path'] for r in inventory['files']}:
        raise ValueError('Complete payload inventory differs')
    for row in inventory['files']:
        path = a.dataset / row['path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Payload bytes differ from fixed full inventory')
    core = a.output.with_name(a.output.stem+'_legacy_core.json')
    legacy_audit(argparse.Namespace(**{**vars(a), 'output': core}))
    codec = G1Codec(a.skeleton)
    accepted = json.loads((a.dataset/'accepted.json').read_text())
    official = {}
    for split in ('train', 'val', 'test'):
        for mid in (a.public/'HumanML3D'/(split+'.txt')).read_text().splitlines():
            official[mid] = split
    overlap = {x['source_group'] for x in accepted
        if len({y['official_split'] for y in accepted if y['source_group']==x['source_group']}) > 1}
    for kind in ('official', 'source_disjoint'):
        manifest = json.loads((a.dataset/('manifest_'+kind+'.json')).read_text())
        for split in ('train', 'val', 'test'):
            expected = [x for x in accepted if x['official_split']==split and
                        (kind=='official' or x['source_group'] not in overlap)]
            if manifest['splits'][split] != expected:
                raise ValueError('Manifest differs from accepted entries / official buckets')
            if any(official[x['motion_id']] != split for x in expected):
                raise ValueError('Manifest disagrees with official split')
    model = mujoco.MjModel.from_xml_path(str(a.gmr_root/'assets/unitree_g1/g1_mocap_29dof.xml'))
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                 for n in codec.joint_names]
    names = sorted(set(codec.body_names) | set(SELECTED) | {'pelvis'} |
        {j[k] for j in codec.movable.values() for k in ('parent','child')})
    fps = 20.; channel_error = np.zeros(262); ric = 0.; cache = {}; canonical_frames = {}; test_frames = {}
    config = json.loads((a.dataset/'config.json').read_text())
    for entry in accepted:
        with np.load(a.dataset/entry['native'], allow_pickle=False) as d:
            native = d['native'].astype(float); ids = d['reference_raw_frames']
        x = np.load(a.dataset/entry['features'], allow_pickle=False)
        p, r = world_fk(model, data, native, names, addresses)
        independent = independent_features(native, p, r, codec, fps)
        channel_error = np.maximum(channel_error, np.abs(independent-x).max(0))
        if not np.array_equal(independent[:, 256:], x[:, 256:]):
            raise ValueError('Independent MuJoCo contact semantics differ')
        metadata = json.loads((a.dataset/entry['codec_metadata']).read_text())
        _, _, decoded = codec.decode(x, metadata)
        ric = max(ric, decoded['ric_redundancy_residual_m'])
        screening = quality(codec, native, fps, config['thresholds'])
        if screening['passed'] != entry['screening']['passed']:
            raise ValueError('Recomputed screening decision differs')
        for k, v in screening.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if abs(v-entry['screening'][k]) > 1e-4:
                    raise ValueError('Recomputed screening metric differs')
        key = (entry['motion_id'], tuple(ids.tolist()))
        if key in cache: raise ValueError('Duplicate motion/frame interval')
        cache[key] = (native, p, r)
        canonical_frames.setdefault(entry['source_group'], set()).update(ids.tolist())
        if entry['official_split']=='test':
            test_frames.setdefault(entry['source_group'], set()).update(ids.tolist())
    if channel_error[:256].max() > 1e-4 or ric > 2e-5:
        raise ValueError(('Full-channel/RIC tolerance exceeded', channel_error.max(), ric))
    s = np.diag([1., -1., 1.]); mirror_errors = {'root_position_m':0., 'root_rotation':0.,
        'joint_rad':0., 'derivative_nominal_units':0., 'body_position_m':0., 'body_rotation':0.}
    pairs = 0
    for (mid, ids), (m, mp, mr) in cache.items():
        if not mid.startswith('M'): continue
        b, bp, br = cache[(mid[1:], ids)]; pairs += 1
        mirror_errors['root_position_m'] = max(mirror_errors['root_position_m'], float(np.abs(m[:,:3]-b[:,:3]@s).max()))
        mirror_errors['root_rotation'] = max(mirror_errors['root_rotation'], float(np.abs(root_rotation(m)-s@root_rotation(b)@s).max()))
        for j, n in enumerate(codec.joint_names):
            partner = opposite(n); k = codec.joint_names.index(partner)
            sign = float(np.asarray(codec.movable[n]['axis']) @ (-s @ np.asarray(codec.movable[partner]['axis'])))
            mirror_errors['joint_rad'] = max(mirror_errors['joint_rad'], float(np.abs(m[:,13+j]-sign*b[:,13+k]).max()))
            mirror_errors['derivative_nominal_units'] = max(mirror_errors['derivative_nominal_units'], float(np.abs(m[:,42+j]-sign*b[:,42+k]).max()))
        mirror_errors['derivative_nominal_units'] = max(mirror_errors['derivative_nominal_units'],
            float(np.abs(m[:,7:10]-b[:,7:10]@s).max()), float(np.abs(m[:,10:13]+b[:,10:13]@s).max()))
        for n in codec.body_names:
            partner = opposite(n)
            mirror_errors['body_position_m'] = max(mirror_errors['body_position_m'], float(np.abs(mp[n]-bp[partner]@s).max()))
            mirror_errors['body_rotation'] = max(mirror_errors['body_rotation'], float(np.abs(mr[n]-s@br[partner]@s).max()))
    if mirror_errors['body_position_m'] > 2e-5 or max(v for k,v in mirror_errors.items() if k!='body_position_m') > 2e-5:
        raise ValueError(('Artifact reflection exceeds tolerance', mirror_errors))
    heights = []; ambiguous = []
    for path in sorted((a.dataset/'source_receipts').glob('*.json')):
        record = json.loads(path.read_text())
        with np.load(record['smplx_file'], allow_pickle=False) as d:
            height = float(1.66+.1*d['betas'].reshape(-1)[0])
        heights.append({'source_group': record['source_group'], 'height_heuristic_m':height,
            'observation_status':'post_run_recomputation_not_execution_time_receipt'})
        if abs(record['timeline']['reference_fps']-record['timeline']['smplx_header_fps']) > 1:
            ambiguous.append(record['source_group'])
    tree = {p.relative_to(a.gmr_root).as_posix():sha(p)
        for p in sorted((a.gmr_root/'general_motion_retargeting').rglob('*'))
        if p.is_file() and p.suffix in ('.py','.json')}
    groups = {split:{x['source_group'] for x in accepted if x['official_split']==split}
              for split in ('train','val','test')}
    subjects = {k:{PurePosixPath(g).parts[-2] for g in v} for k,v in groups.items()}
    return {'status':'SUPPLEMENTAL_SEMANTICS_CHECKS_PASSED',
        'evaluation_type':'mixed_real_source_algorithmic_retarget_proxy_consistency',
        'complete_payload_inventory_files_checked':len(inventory['files']),
        'complete_payload_inventory_sha256':sha(a.inventory),
        'legacy_core_receipt':str(core),'legacy_core_receipt_sha256':sha(core),
        'all_manifest_entries_and_official_buckets_equal':True,
        'all_frames_all_262_channels_recomputed_from_mujoco_and_scipy':True,
        'channel_max_abs_errors':channel_error.tolist(),'continuous_channel_tolerance':1e-4,
        'contact_channels_exact_match':True,'ric_mean_residual_max_m':ric,'ric_tolerance_m':2e-5,
        'kinematic_screening_recomputed':True,'mirror_artifact_pairs_checked':pairs,
        'mirror_errors':mirror_errors,'mirror_position_tolerance_m':2e-5,
        'unique_canonical_frames':sum(map(len,canonical_frames.values())),
        'unique_canonical_test_frames':sum(map(len,test_frames.values())),
        'canonical_test_recordings':len(test_frames),'subject_session_identifiers':sorted(set.union(*subjects.values())),
        'train_test_subject_session_overlap':sorted(subjects['train'] & subjects['test']),
        'subject_disjoint':False,'source_physical_timebase_ambiguous_groups':ambiguous,
        'feature_timebase':'nominal HumanML indexing / chosen 20-tick playback; source physical timing unverified for conflicting headers',
        'speed_metrics_are_nominal_playback_units':True,'height_formula':'1.66 + 0.1 * beta[0], heuristic',
        'post_run_height_observations':heights,'post_run_gmr_solver_source_sha256':tree,
        'observed_source_tree_is_not_execution_time_proof':True,
        'retargeting_fidelity_or_author_exact_codec_validated':False,
        'physical_tracking_evaluated':False,'generator_training_started':False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('dataset','public','skeleton','gmr-root','inventory','output'):p.add_argument('--'+n,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(); code = 0
    try: result = check(a)
    except Exception as exc:
        code = 1; result = {'status':'FAILED','error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()}
    result.update({'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'exit_code':code,'source_sha256':sha(Path(__file__)),
        'accepted_sha256':sha(a.dataset/'accepted.json'),'all_data_retained':True})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp = a.output.with_suffix(a.output.suffix+'.writing')
    if temp.exists():raise FileExistsError(temp)
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.rename(a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('channel_max_abs_errors','post_run_height_observations','post_run_gmr_solver_source_sha256')}))
    sys.exit(code)
