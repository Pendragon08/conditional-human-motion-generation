"""Real licensed motions -> GMR/G1 -> real caption intervals -> screening -> 262D.

The public metadata/provenance, cross-export marker identity, explicit names,
and immutable output directory are required. No input or failed artifact is deleted.
"""
import argparse
import collections
import contextlib
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile
import numpy as np
import torch
import smplx
import mujoco

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'reimplementation'))
from codec262 import G1Codec, quat_matrix
from features import BlockNormalizer
from humanml_pairing import (sha, verify_timeline, native_from_qpos, mirror_native,
                            mirror_fk_witness, quality)
from plan_humanml_amass import original_frame_indices, parse_caption, source_parts


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def run(a):
    if a.output.exists():
        raise FileExistsError('Existing run retained: ' + str(a.output))
    config = json.loads(a.config.read_text())
    spec = json.loads(a.spec.read_text())
    versions = {name: importlib.metadata.version(name) for name in spec['required_versions']}
    if versions != spec['required_versions']:
        raise ValueError('Environment differs from pinned spec')
    torch.manual_seed(config['seed']); torch.set_num_threads(config['cpu_threads'])
    codec = G1Codec(a.skeleton)
    witness = mirror_fk_witness(codec)
    public = a.public
    provenance = json.loads(a.provenance.read_text())
    public_hashes = {row['path']: row['sha256'] for row in provenance['files']}
    for name in ['index.csv', 'HumanML3D/texts.zip'] + ['HumanML3D/' + s + '.txt' for s in ['all','train','val','test']]:
        if sha(public / name) != public_hashes[name]:
            raise ValueError('Pinned public metadata differs: ' + name)
    rows = list(csv.DictReader((public / 'index.csv').open()))
    index = {Path(row['new_name']).stem: row for row in rows}
    splits = {}
    for split in ['train','val','test']:
        for mid in (public / 'HumanML3D' / (split + '.txt')).read_text().splitlines():
            if mid in splits:
                raise ValueError('Duplicate official ID across splits')
            splits[mid] = split
    jobs = json.loads(a.jobs.read_text())['jobs']
    if a.max_sources:
        jobs = jobs[:a.max_sources]
    sys.path.insert(0, str(a.gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.smpl import get_smplx_data
    from general_motion_retargeting.params import IK_CONFIG_DICT
    a.output.mkdir(parents=True, exist_ok=False)
    save(a.output / 'config.json', config)
    save(a.output / 'mirror_witness.json', witness)
    accepted, rejected, source_receipts = [], [], []
    models = {}
    started = time.monotonic()
    with zipfile.ZipFile(public / 'HumanML3D/texts.zip') as captions:
        caption_members = {Path(n).stem: n for n in captions.namelist() if n.endswith('.txt')}
        for number, job in enumerate(jobs, 1):
            key = hashlib.sha256(job['source_group'].encode()).hexdigest()[:16]
            try:
                target_path, reference_path = Path(job['smplx_file']), Path(job['smplh_reference_file'])
                with np.load(target_path, allow_pickle=False) as x, np.load(reference_path, allow_pickle=False) as h:
                    timeline = verify_timeline(h, x)
                    ref_n, ref_fps = len(h['trans']), float(h['mocap_framerate'])
                    frame_ids = np.arange(0, ref_n, int(ref_fps / 20))
                    raw = {n: np.asarray(x[n]) for n in ['trans','root_orient','pose_body','pose_hand','pose_jaw','pose_eye','betas']}
                    gender = str(x['gender'])
                if gender not in models:
                    models[gender] = smplx.create(str(a.model_root), 'smplx', ext='npz',
                        gender=gender, num_betas=16, use_pca=False).cpu()
                model = models[gender]
                height = float(1.66 + .1 * raw['betas'].reshape(-1)[0])
                with contextlib.redirect_stdout(sys.stderr):
                    gmr = GeneralMotionRetargeting(src_human='smplx', tgt_robot='unitree_g1',
                        actual_human_height=height, verbose=False)
                addresses = []
                for name in codec.joint_names:
                    jid = mujoco.mj_name2id(gmr.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid < 0:
                        raise ValueError('GMR robot missing codec joint: ' + name)
                    addresses.append(int(gmr.model.jnt_qposadr[jid]))
                qposes = []
                for first in range(0, len(frame_ids), config['smplx_batch_size']):
                    ids = frame_ids[first:first + config['smplx_batch_size']]
                    count = len(ids)
                    tensor = lambda x: torch.from_numpy(np.array(x, copy=True)).float()
                    with torch.no_grad():
                        body = model(betas=tensor(raw['betas'].reshape(1,16)).expand(count,-1),
                            global_orient=tensor(raw['root_orient'][ids]),
                            body_pose=tensor(raw['pose_body'][ids]), transl=tensor(raw['trans'][ids]),
                            left_hand_pose=tensor(raw['pose_hand'][ids,:45]),
                            right_hand_pose=tensor(raw['pose_hand'][ids,45:]),
                            jaw_pose=tensor(raw['pose_jaw'][ids]),
                            leye_pose=tensor(raw['pose_eye'][ids,:3]),
                            reye_pose=tensor(raw['pose_eye'][ids,3:]),
                            expression=torch.zeros(count, model.num_expression_coeffs),
                            return_full_pose=True, return_verts=False)
                    for local_frame in range(count):
                        human = get_smplx_data({}, model, body, local_frame)
                        qpos = gmr.retarget(human)
                        qposes.append(np.r_[qpos[:7], qpos[addresses]])
                native = native_from_qpos(codec, np.stack(qposes), config['fps'])
                source_out = a.output / 'retargeted_sources' / (key + '.npz')
                source_out.parent.mkdir(exist_ok=True)
                np.savez_compressed(source_out, native=native.astype(np.float32), reference_raw_frames=frame_ids)
                source_record = {'source_group': job['source_group'], 'smplx_file': str(target_path),
                    'smplx_sha256': sha(target_path), 'reference_file': str(reference_path),
                    'reference_sha256': sha(reference_path), 'timeline': timeline,
                    'retargeted_frames': len(native), 'retargeted_source': str(source_out.relative_to(a.output)),
                    'retargeted_source_sha256': sha(source_out), 'human_model_gender': gender,
                    'human_model_sha256': sha(a.model_root/'smplx'/('SMPLX_'+gender.upper()+'.npz')),
                    'unavailable_expression_coefficients': 'Explicit zero expression; AMASS file has no expression field',
                    'gmr_ik_config_sha256': sha(IK_CONFIG_DICT['smplx']['unitree_g1']),
                    'gmr_robot_xml_sha256': sha(gmr.xml_file), 'success': True}
                raw_to_grid = {int(v): k for k,v in enumerate(frame_ids)}
                for mid in sorted(job['motion_ids']):
                    row = index[mid[1:] if mid.startswith('M') else mid]
                    groups = collections.defaultdict(list)
                    for line_no, line in enumerate(captions.read(caption_members[mid]).decode().splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            caption = parse_caption(line)
                            raw_ids = original_frame_indices(row, ref_fps, ref_n, caption)
                            groups[tuple(raw_ids)].append({'text':line.split('#')[0], 'line':line_no,
                                'kind':caption['kind'], 'start_20':caption['start_20'], 'end_20':caption['end_20']})
                        except (ValueError, OverflowError) as exc:
                            rejected.append({'motion_id':mid,'caption_line':line_no,'reason':str(exc), 'stage':'caption_alignment'})
                    for interval, (raw_ids, texts) in enumerate(groups.items()):
                        grid_ids = np.array([raw_to_grid[v] for v in raw_ids])
                        cropped = native[grid_ids].copy()
                        if mid.startswith('M'):
                            cropped = mirror_native(codec, cropped, config['fps'])
                        # Recompute boundary derivatives after the exact caption crop.
                        qpos = np.c_[cropped[:,:7],cropped[:,13:42]]
                        cropped = native_from_qpos(codec,qpos,config['fps'])
                        screening = quality(codec,cropped,config['fps'],config['thresholds'])
                        entry = {'motion_id':mid, 'interval':interval,'source_group':job['source_group'],
                            'official_split':splits[mid], 'mirrored':mid.startswith('M'),
                            'reference_raw_start':int(raw_ids[0]),'reference_raw_last':int(raw_ids[-1]),
                            'reference_raw_stride':timeline['reference_stride'], 'frames':len(raw_ids),
                            'captions':texts,'texts':[t['text'] for t in texts],
                            'annotation_source':'HumanML3D@'+provenance['pinned_commit']+':'+mid,
                            'screening':screening, 'tags':[], 'dynamics_validated':False}
                        if not screening['passed']:
                            rejected.append({**entry,'stage':'kinematic_screening'});continue
                        features, metadata = codec.encode(cropped, config['fps'])
                        recovered, _, check = codec.decode(features,metadata)
                        pos_error=float(np.abs(recovered[:,:3]-cropped[:,:3]).max())
                        q_error=float(np.abs(recovered[:,13:42]-cropped[:,13:42]).max())
                        rot_error=float(np.abs(quat_matrix(recovered[:,3:7])-quat_matrix(cropped[:,3:7])).max())
                        if max(pos_error,q_error,rot_error)>2e-4:
                            raise ValueError('262D round-trip exceeded tolerance')
                        stem=mid+'_'+str(interval).zfill(2)
                        folder=a.output/'features'/splits[mid];folder.mkdir(parents=True,exist_ok=True)
                        feature_file=folder/(stem+'.npy');np.save(feature_file,features)
                        native_file=folder/(stem+'_native.npz')
                        np.savez_compressed(native_file,native=cropped.astype(np.float32),reference_raw_frames=np.array(raw_ids))
                        save(folder/(stem+'_metadata.json'),metadata)
                        entry.update({'features':str(feature_file.relative_to(a.output)), 'sha256':sha(feature_file),
                            'native':str(native_file.relative_to(a.output)), 'native_sha256':sha(native_file),
                            'codec_metadata':str((folder/(stem+'_metadata.json')).relative_to(a.output)),
                            'roundtrip':{'position_max_abs_m':pos_error,'joint_max_abs_rad':q_error,
                                         'rotation_matrix_max_abs':rot_error,**check}})
                        accepted.append(entry)
                source_receipts.append(source_record)
                save(a.output/'source_receipts'/ (key+'.json'),source_record)
                print(json.dumps({'processed_source':number,'total_sources':len(jobs),
                                  'accepted_sequences':len(accepted),'rejected_records':len(rejected)}),flush=True)
            except Exception as exc:
                error={'source_group':job['source_group'],'success':False,'error':type(exc).__name__+': '+str(exc)}
                source_receipts.append(error);save(a.output/'source_errors'/(key+'.json'),error)
                (a.output/'source_errors'/(key+'_traceback.txt')).write_text(traceback.format_exc())
                print(json.dumps(error),flush=True)
    groups=collections.defaultdict(set)
    for entry in accepted:groups[entry['source_group']].add(entry['official_split'])
    overlapping={name for name,parts in groups.items() if len(parts)>1}
    common={'format':'phygile262-explicit-v1','purpose':'research_training',
        'feature_semantics':'Pinned reconstruction codec; canonical 20Hz G1 FK with measured-marker verified HumanML alignment',
        'source_and_license':'Official licensed AMASS SSM SMPL-X / SMPL+H reference, paired with official HumanML3D text; no redistribution of motion arrays',
        'mapping_status':'explicit_reconstruction', 'codec_schema':metadata['schema'] if accepted else None,
        'codec_skeleton_sha256':sha(a.skeleton), 'physical_tracking_validated':False,
        'author_exact_codec_verified':False}
    for kind in ['official','source_disjoint']:
        manifest={**common,'protocol':kind,'excluded_cross_split_groups':sorted(overlapping) if kind=='source_disjoint' else [],
            'splits':{s:[x for x in accepted if x['official_split']==s and (kind=='official' or x['source_group'] not in overlapping)] for s in ['train','val','test']}}
        save(a.output/('manifest_'+kind+'.json'),manifest)
        if kind=='source_disjoint':
            training=[np.load(a.output/x['features'],allow_pickle=False) for x in manifest['splits']['train']]
            if training:
                normalizer=BlockNormalizer.fit(training)
                np.savez(a.output/'normalization_train_only.npz',mean=normalizer.mean,std=normalizer.std)
    save(a.output/'accepted.json',accepted);save(a.output/'rejected.json',rejected)
    summary={'status':'COMPLETED' if all(x['success'] for x in source_receipts) else 'COMPLETED_WITH_SOURCE_ERRORS',
        'sources_requested':len(jobs),'sources_retargeted':sum(x['success'] for x in source_receipts),
        'accepted_sequences':len(accepted),'accepted_motion_ids':len({x['motion_id'] for x in accepted}),
        'accepted_frames':sum(x['frames'] for x in accepted), 'accepted_caption_lines':sum(len(x['texts']) for x in accepted),
        'accepted_mirrored_sequences':sum(x['mirrored'] for x in accepted),
        'rejected_records':len(rejected),'cross_split_recording_groups':sorted(overlapping),
        'official_split_sequences':dict(collections.Counter(x['official_split'] for x in accepted)),
        'source_disjoint_split_sequences':{s:len(v) for s,v in manifest['splits'].items()},
        'device':'cpu','gpu_used':False,'elapsed_seconds':time.monotonic()-started,
        'versions':versions,'environment_spec_hash':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8],
        'source_hashes':{str(p.relative_to(BASE)):sha(p) for p in [Path(__file__),BASE/'reimplementation/humanml_pairing.py',BASE/'reimplementation/codec262.py',BASE/'scripts/plan_humanml_amass.py']},
        'public_provenance_sha256':sha(a.provenance),'config_sha256':sha(a.config),
        'dynamics_validated':False,'generator_training_started':False,'all_humanml3d_processed':False,
        'data_deleted':False,'author_data_or_performance_reproduced':False}
    save(a.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['public','provenance','jobs','gmr-root','model-root','spec','skeleton','config','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-sources',type=int)
    run(p.parse_args())
