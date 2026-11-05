"""Compare FIR approximation with Slerp on 13 real SSM recordings."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanml_time_mapping import aligned_grid
from humanml_rotation_resample import apply_grid
from humanml_fir_resample import fir_resample

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
IDS=BASE/'ssm_id_alignment_20261030/ids.jsonl'
RECEIPTS=BASE/'ssm_receipt_audit_20261030/rows.jsonl'
OUT=BASE/'ssm_fir_real_comparison_20261030'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ids=list(map(json.loads,IDS.open())); receipts={x['source_group']:x for x in map(json.loads,RECEIPTS.open())}
    selected=[x for x in ids if x['status']=='prior_marker_aligned_non20_requires_reprocessing' and not x['mirrored']]
    if len(selected)!=13 or len({x['source_path'] for x in selected})!=13:raise ValueError('unexpected sample selection')
    rows=[]
    for item in selected:
        record=receipts[item['source_path']]; rec=json.loads(Path(record['receipt']).read_text())
        src=Path(rec['smplx_file']);ref=Path(rec['reference_file'])
        if sha(src)!=rec['smplx_sha256'] or sha(ref)!=rec['reference_sha256']:raise ValueError('input hash differs')
        grid=aligned_grid(item['reference_raw_frame_count'],item['reference_fps'],item['start'],item['stop'])
        with np.load(src,allow_pickle=False) as x:
            trans=np.asarray(x['trans'],dtype=np.float64)
            pose=np.concatenate([np.asarray(x['root_orient']),np.asarray(x['pose_body'])],axis=1)
        src_ids=np.asarray(grid['source_indices'])
        ordinary_p,ordinary_r=apply_grid(trans,pose,grid)
        fir_p,fir_r,meta=fir_resample(trans[src_ids],pose[src_ids],grid['actual_reference_sample_fps'])
        if ordinary_p.shape!=fir_p.shape or ordinary_r.shape!=fir_r.shape:raise ValueError('different output grids')
        delta=np.linalg.norm(fir_p-ordinary_p,axis=1)
        a=Rotation.from_rotvec(ordinary_r.reshape(-1,3));b=Rotation.from_rotvec(fir_r.reshape(-1,3))
        angle=(a.inv()*b).magnitude().reshape(len(fir_p),22)
        rows.append({'motion_id':item['motion_id'],'source':item['source_path'],'frames':len(fir_p),
                     'position_difference_mean_m':float(delta.mean()),'position_difference_max_m':float(delta.max()),
                     'joint_angle_difference_mean_rad':float(angle.mean()),'joint_angle_difference_max_rad':float(angle.max()),
                     'source_sha256':rec['smplx_sha256'],'reference_sha256':rec['reference_sha256'], 'timing':meta})
    OUT.mkdir(mode=0o700)
    with (OUT/'rows.jsonl').open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    summary={'recordings':len(rows), 'max_position_difference_m':max(x['position_difference_max_m'] for x in rows),
             'max_joint_angle_difference_rad':max(x['joint_angle_difference_max_rad'] for x in rows),
             'mean_of_recording_position_means_m':float(np.mean([x['position_difference_mean_m'] for x in rows])),
             'mean_of_recording_joint_angle_means_rad':float(np.mean([x['joint_angle_difference_mean_rad'] for x in rows])),
             'limitations':['FIR component filtering on quaternion is approximate.',
                            'Method differences do not quantify error relative to truth.',
                            'No foot contact, collision, task object, dynamics, annotation or G1 IK validated.',
                            '13 non-mirrored SSM paths only; no training data produced.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
