"""Build a four-record, NON-TRAINING technical preview of the data interface.

Official HumanML splits remain unchanged; the auxiliary assignment uses exact
source paths only and is deliberately unsuitable as a validated evaluation.
The existing training entry must still fail closed on this preview.
"""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from audit_robot_features import audit
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK
from training_stats import training_stats

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    assignments = {'000008': 'train', '000069': 'train', '013303': 'val', '003017': 'test'}
    alignment = ROOT / 'ssm_id_alignment_20261030/ids.jsonl'
    rows = {r['motion_id']: r for r in map(json.loads, alignment.read_text().splitlines())}
    if sha(alignment) != '575c9c008298dbf30d21317f258346a3dd11cc7efc6858d242a26c11c6333b46':
        raise ValueError('source mapping changed')
    splits = {name: set((HUMANML / (name + '.txt')).read_text().splitlines()) for name in ('train','val','test')}
    archive = HUMANML / 'texts.zip'; archive_sha = sha(archive)
    human_batch = ROOT / 'ssm_three_human_candidates_20261031'
    human_report = json.loads((human_batch / 'report.json').read_text())
    human_audit = json.loads((human_batch / 'independent_audit.json').read_text())
    if sha(human_batch / 'report.json') != human_audit['witness_report_sha256']:
        raise ValueError('human source audit mismatch')
    model = URDFFK(URDF)
    records, sources, data_for_stats = [], [], []
    for name, assigned in assignments.items():
        row = rows[name]
        if row['mirrored'] or not row['nominal_crop_in_bounds']:
            raise ValueError('unsupported source')
        official = [s for s, ids in splits.items() if name in ids]
        if len(official) != 1: raise ValueError('official split ambiguous')
        if name == '000008':
            prior = ROOT / 'ssm_000008_feature_candidate_20261031'
            receipt = json.loads((prior / 'report.json').read_text())
            prior_audit = json.loads((prior / 'audit_report.json').read_text())
            if sha(prior / 'report.json') != prior_audit['candidate_report_sha256'] or sha(prior / 'candidate.npz') != receipt['output_sha256']:
                raise ValueError('prior feature candidate changed')
            with np.load(prior / 'candidate.npz', allow_pickle=False) as d:
                features = d['features'].copy(); grid = d['time_grid'].copy()
            trajectory_hash = receipt['source_trajectory_sha256']
            tail = receipt['tail_not_sampled_s']
            limit_violation = 0.0
        else:
            solve = ROOT / 'ssm_three_global_20261031' / name
            receipt = json.loads((solve / 'report.json').read_text())
            check = json.loads((solve / 'audit_report.json').read_text())
            traj = solve / 'trajectory.npz'
            if (sha(solve / 'report.json') != check['report_sha256'] or
                    sha(traj) != check['trajectory_sha256'] or
                    sha(traj) != receipt['artifact_sha256'] or
                    sha(URDF) != receipt['urdf_sha256'] or check['training_ready'] is not False):
                raise ValueError('trajectory audit mismatch')
            with np.load(traj, allow_pickle=False) as d:
                t = d['source_times'].astype(np.float64); root = d['root_wxyz_xyz'].astype(np.float64)
                q = d['joint_pos'].astype(np.float64); names = [str(x) for x in d['joint_names']]
            count = int(np.floor((t[-1] - t[0]) * 20)) + 1
            grid = t[0] + np.arange(count) / 20
            if count < 3 or grid[-1] > t[-1]: raise ValueError('invalid temporal grid')
            pos = np.column_stack([np.interp(grid, t, root[:,i]) for i in (4,5,6)])
            joints = np.column_stack([np.interp(grid, t, q[:,i]) for i in range(27)])
            rot = Slerp(t, Rotation.from_quat(root[:,[1,2,3,0]]))(grid).as_quat()
            points = model.forward(pos, rot, joints, names)
            features = encode(points, foot_velocity_squared_threshold=.002)
            with torch.no_grad(): decoded = recover_from_ric(torch.from_numpy(features),22).numpy()
            if np.linalg.norm(decoded - canonicalize_z_up(points)[:-1],axis=-1).max() > 1e-4:
                raise ValueError('feature inverse mismatch')
            trajectory_hash = sha(traj); tail = float(t[-1] - grid[-1])
            limit_violation = check['max_joint_limit_violation_rad']
        with zipfile.ZipFile(archive) as z: text_bytes = z.read('texts/' + name + '.txt')
        entries = [line.split('#') for line in text_bytes.decode('utf8').splitlines()]
        if any(len(x) != 4 for x in entries): raise ValueError('unexpected caption fields')
        whole = [x[0] for x in entries if x[0] and x[2:] == ['0.0','0.0']]
        if not whole: raise ValueError('no full-sequence caption')
        out = HERE / (name + '.npz')
        with out.open('xb') as f: np.savez_compressed(f,features=features,time_grid=grid)
        rec = {'sample_id': name, 'group_id': row['source_path'], 'source_id': row['source_path'],
               'split': assigned, 'official_split': official[0], 'caption': whole[0],
               'caption_source': 'HumanML texts.zip: texts/' + name + '.txt',
               'feature_path': out.name, 'feature_sha256': sha(out), 'fps': 20,
               'training_ready': False}
        records.append(rec); data_for_stats.append((assigned,features))
        sources.append({'id': name, 'official_split': official[0], 'auxiliary_split': assigned,
                        'raw_tags': [x[2:] for x in entries], 'whole_caption_count': len(whole),
                        'other_caption_count': len(entries)-len(whole),
                        'caption_member_sha256': hashlib.sha256(text_bytes).hexdigest(),
                        'source_trajectory_sha256': trajectory_hash,
                        'feature_sha256': rec['feature_sha256'], 'frames': len(features),
                        'tail_not_sampled_s': tail, 'max_original_joint_violation_rad': limit_violation})
    if sha(archive) != archive_sha: raise RuntimeError('caption archive changed during preview build')
    stats = training_stats(data_for_stats,min_std=1e-3)
    train_joints = []
    for split, x in data_for_stats:
        if split == 'train':
            with torch.no_grad(): p = recover_from_ric(torch.from_numpy(x),22).numpy()
            train_joints.append(p.reshape(len(x),66).astype(np.float64))
    spatial = np.concatenate(train_joints)
    values = {'feature_mean': stats['mean'], 'feature_std': stats['std'],
              'spatial_mean': spatial.mean(axis=0), 'spatial_std': np.maximum(spatial.std(axis=0),1e-3)}
    with (HERE / 'experimental_stats_not_for_training.npz').open('xb') as f:
        np.savez_compressed(f,**values)
    manifest = {'records':records, 'training_ready':False,
        'purpose':'technical preview only; auxiliary split differs from official split',
        'limitations':['four short same-dataset clips','exact-path grouping is incomplete',
                       'source model and license unverified','no retarget or physics acceptance']}
    with (HERE / 'experimental_manifest_not_for_training.json').open('x',encoding='utf8') as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    checked = audit(HERE,records,values,min_std=1e-3)
    if checked['training_ready'] is not False: raise ValueError('training gate must remain closed')
    result = {'script_sha256':sha(__file__),'source_mapping_sha256':sha(alignment),
        'human_batch_report_sha256':sha(human_batch/'report.json'),
        'human_batch_audit_sha256':sha(human_batch/'independent_audit.json'),
        'caption_archive_sha256':archive_sha,'urdf_sha256':sha(URDF),'samples':sources,
        'manifest_sha256':sha(HERE/'experimental_manifest_not_for_training.json'),
        'stats_sha256':sha(HERE/'experimental_stats_not_for_training.npz'),
        'train_frames':stats['frame_count'],'clamped_feature_dimensions':len(stats['clamped_channels']),
        'feature_audit':checked,'training_ready':False,
        'limitations':['technical interface preview, no approved training set',
            '013303 is moved from official train to auxiliary val; official split not modified',
            'exact source paths are not a full recording/subject independence audit',
            'timed captions excluded from this preview; whole-caption semantics still unverified',
            'samples 003017 and 013303 have recorded nonzero joint-limit violations; not clipped or accepted',
            'source model, license, physical/retarget quality and paper method unaccepted']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
