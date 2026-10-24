"""Read real dataset artifacts and verify data consistency; no tracking claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import mujoco

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'reimplementation'))
from codec262 import G1Codec,quat_matrix
from features import NORMALIZED
from humanml_pairing import sha,verify_timeline
from plan_humanml_amass import original_frame_indices,parse_caption


def audit(a):
    if a.output.exists():raise FileExistsError(a.output)
    dataset=a.dataset
    accepted=json.loads((dataset/'accepted.json').read_text())
    summary=json.loads((dataset/'summary.json').read_text())
    if not accepted or summary['accepted_sequences']!=len(accepted):raise ValueError('Missing/miscounted accepted data')
    codec=G1Codec(a.skeleton)
    rows={Path(x['new_name']).stem:x for x in csv.DictReader((a.public/'index.csv').open())}
    source_records={}
    for path in (dataset/'source_receipts').glob('*.json'):
        record=json.loads(path.read_text());source_records[record['source_group']]=record
        if sha(record['smplx_file'])!=record['smplx_sha256'] or sha(record['reference_file'])!=record['reference_sha256']:
            raise ValueError('Raw source hash differs')
        with np.load(record['reference_file'],allow_pickle=False) as h,np.load(record['smplx_file'],allow_pickle=False) as x:
            timeline=verify_timeline(h,x)
        if timeline!=record['timeline']:raise ValueError('Marker timeline differs from recorded proof')
    mjmodel=mujoco.MjModel.from_xml_path(str(a.gmr_root/'assets/unitree_g1/g1_mocap_29dof.xml'))
    mjdata=mujoco.MjData(mjmodel)
    addresses=[mjmodel.jnt_qposadr[mujoco.mj_name2id(mjmodel,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in codec.joint_names]
    mj_error=pos_error=q_error=rot_error=0.;caption_count=frames=0
    official_ids={}
    for split in ['train','val','test']:
        for mid in (a.public/'HumanML3D'/(split+'.txt')).read_text().splitlines():official_ids[mid]=split
    with zipfile.ZipFile(a.public/'HumanML3D/texts.zip') as z:
        names={Path(p).stem:p for p in z.namelist() if p.endswith('.txt')}
        for entry in accepted:
            mid=entry['motion_id'];source=source_records[entry['source_group']]
            if official_ids[mid]!=entry['official_split']:raise ValueError('Official split changed')
            reference=rows[mid[1:] if mid.startswith('M') else mid]
            if reference['source_path']!=entry['source_group']:raise ValueError('HumanML source identity mismatch')
            x=np.load(dataset/entry['features'],allow_pickle=False)
            with np.load(dataset/entry['native'],allow_pickle=False) as d:
                native=d['native'];raw_frames=d['reference_raw_frames']
            if (sha(dataset/entry['features'])!=entry['sha256'] or sha(dataset/entry['native'])!=entry['native_sha256']):
                raise ValueError('Prepared file hash differs')
            if x.shape!=(entry['frames'],262) or not np.isfinite(x).all() or not np.isin(x[:,256:],[0,1]).all():
                raise ValueError('Invalid encoded features')
            if np.max(np.abs(x[:,43:217]))>1.0001:raise ValueError('Invalid 6D rotations')
            lines=z.read(names[mid]).decode().splitlines()
            for caption in entry['captions']:
                line=lines[caption['line']-1]
                if caption['text']!=line.split('#')[0]:raise ValueError('Caption text changed')
                expected=original_frame_indices(reference,source['timeline']['reference_fps'],
                    source['timeline']['reference_frames'],parse_caption(line))
                if not np.array_equal(raw_frames,expected):raise ValueError('Caption/raw-frame interval mismatch')
                caption_count+=1
            metadata=json.loads((dataset/entry['codec_metadata']).read_text())
            recovered,_,_=codec.decode(x,metadata)
            pos_error=max(pos_error,float(np.abs(recovered[:,:3]-native[:,:3]).max()))
            q_error=max(q_error,float(np.abs(recovered[:,13:42]-native[:,13:42]).max()))
            rot_error=max(rot_error,float(np.abs(quat_matrix(recovered[:,3:7])-quat_matrix(native[:,3:7])).max()))
            body_pos,_=codec.fk(native[:,:3],native[:,3:7],native[:,13:42])
            for t in sorted({0,len(native)//2,len(native)-1}):
                mjdata.qpos[:7]=native[t,:7];mjdata.qpos[addresses]=native[t,13:42]
                mujoco.mj_forward(mjmodel,mjdata)
                for body in codec.body_names:
                    bid=mujoco.mj_name2id(mjmodel,mujoco.mjtObj.mjOBJ_BODY,body)
                    if bid<0:raise ValueError('Missing MuJoCo FK body')
                    mj_error=max(mj_error,float(np.abs(mjdata.xpos[bid]-body_pos[body][t]).max()))
            frames+=len(x)
    if max(pos_error,q_error,rot_error)>2e-4 or mj_error>1e-5:raise ValueError('Codec/MuJoCo consistency failed')
    manifest=json.loads((dataset/'manifest_source_disjoint.json').read_text())
    groups={s:{x['source_group'] for x in items} for s,items in manifest['splits'].items()}
    for left in groups:
        for right in groups:
            if left<right and groups[left]&groups[right]:raise ValueError('Recording leakage')
    training=np.concatenate([np.load(dataset/e['features'],allow_pickle=False).astype(np.float64) for e in manifest['splits']['train']])
    mean=np.zeros(262);std=np.ones(262)
    mean[NORMALIZED]=training[:,NORMALIZED].mean(0)
    std[NORMALIZED]=np.maximum(training[:,NORMALIZED].std(0),1e-6)
    with np.load(dataset/'normalization_train_only.npz') as normalization:
        if not np.allclose(normalization['mean'],mean,rtol=1e-6,atol=1e-6) or not np.allclose(normalization['std'],std,rtol=1e-6,atol=1e-6):
            raise ValueError('Normalization does not match training-only statistics')
    from train import load_manifest
    _,loader_training=load_manifest(dataset/'manifest_source_disjoint.json','research_training')
    result={'status':'DATASET_CONSISTENCY_CHECKS_PASSED','evaluation_type':'data_preparation_consistency',
        'sequences_checked':len(accepted),'frames_checked':frames,'caption_lines_checked':caption_count,
        'raw_sources_rechecked':len(source_records),'train_loader_sequences':len(loader_training),
        'position_max_abs_m':pos_error,'joint_max_abs_rad':q_error,'rotation_matrix_max_abs':rot_error,
        'mujoco_vs_codec_fk_position_max_abs_m':mj_error,'raw_source_and_derived_hashes_checked':True,
        'all_caption_frame_ranges_recomputed':True,'recording_disjoint_manifest_verified':True,
        'train_only_normalization_verified':True,'motion_tracking_success_evaluated':False,
        'author_exact_representation_verified':False,'generator_training_started':False,
        'audit_source_sha256':sha(Path(__file__))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','public','skeleton','gmr-root','output']:p.add_argument('--'+name,type=Path,required=True)
    audit(p.parse_args())
