"""Read-only candidate time alignment when the SMPL+H export lacks markers.

This produces diagnostics, not approved paired training data. Inputs arrive over
an SSH stdin pipe; only metadata is returned and no motion file is written.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass').resolve()


def check(q):
    p=Path(q['smplx_file']).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Source outside authorized asset root')
    with np.load(p,allow_pickle=False) as x:
        target_trans=x['trans'];target_poses=np.c_[x['root_orient'],x['pose_body']]
        target_fps=float(x['mocap_frame_rate'])
    idx=np.asarray(q['sample_indices'],int);ref_t=np.asarray(q['reference_trans'],float)
    ref_p=np.asarray(q['reference_poses66'],float)
    n=q['reference_frames']
    if n!=len(target_trans):
        return {'source_group':q['source_group'],'status':'FRAME_COUNT_MISMATCH',
                'reference_frames':n,'target_frames':len(target_trans)}
    scores=[]
    for lag in range(-12,13):
        valid=(idx+lag>=0)&(idx+lag<n)
        ids=idx[valid];t=target_trans[ids+lag];r=ref_t[valid]
        centered=(t-r)-(t-r).mean(0)
        trans_rmse=float(np.sqrt(np.mean(centered**2)))
        a=Rotation.from_rotvec(ref_p[valid].reshape(-1,3))
        b=Rotation.from_rotvec(target_poses[ids+lag].reshape(-1,3))
        angles=(a.inv()*b).magnitude()
        pose_rmse=float(np.sqrt(np.mean(angles**2)))
        scores.append({'lag_raw_frames':lag,'centered_trans_rmse_m':trans_rmse,
                       'body_rotation_rmse_rad':pose_rmse,'diagnostic_score':trans_rmse+.1*pose_rmse})
    ranked=sorted(scores,key=lambda r:r['diagnostic_score'])
    return {'source_group':q['source_group'],'status':'FIT_TRAJECTORY_CANDIDATES_ONLY',
        'reference_frames':n,'target_frames':len(target_trans),
        'reference_fps':q['reference_fps'],'target_header_fps':target_fps,
        'best_lag_raw_frames':ranked[0]['lag_raw_frames'],
        'zero_lag':next(x for x in scores if x['lag_raw_frames']==0),
        'all_lag_diagnostics':scores,'measurement_marker_identity_verified':False,
        'approved_for_training':False,'data_written_on_server':False}


for line in sys.stdin:
    try:result=check(json.loads(line))
    except Exception as exc:result={'status':'FAILED','error':type(exc).__name__+': '+str(exc)}
    print(json.dumps(result,allow_nan=False),flush=True)
