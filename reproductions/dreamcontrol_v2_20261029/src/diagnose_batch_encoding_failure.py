"""Locate decoding inconsistency across all frames of the quarantined clip."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up,encode
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_gpu_batch24_20261101';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sample='sub8_largebox_006'; records={r['sample_id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    r=records[sample];p=SOURCE/r['state20'];f=SOURCE/r['feature'];assert sha(p)==r['state20_sha256'] and sha(f)==r['feature_sha256']
    with np.load(p,allow_pickle=False) as d:points=d['points_z_up'];root=d['root_wxyz_xyz'];times=d['sample_times']
    with np.load(f,allow_pickle=False) as d:features=d['features']
    assert np.array_equal(encode(points,foot_velocity_squared_threshold=.002),features)
    canonical=canonicalize_z_up(points)[:-1]
    with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
    delta=np.linalg.norm(canonical-decoded,axis=-1);per_frame=delta.max(axis=1);bad=np.flatnonzero(per_frame>1e-4)
    rot=Rotation.from_quat(root[:,[1,2,3,0]]);angle=(rot[:-1].inv()*rot[1:]).magnitude()
    peak=np.unravel_index(delta.argmax(),delta.shape)
    result={'sample_id':sample,'state_sha256':sha(p),'feature_sha256':sha(f),'script_sha256':sha(Path(__file__)),
        'feature_rows':len(features),'rows_above_1e_4_m':bad.tolist(),'max_error_m':float(delta.max()),
        'worst_row':int(peak[0]),'worst_joint':int(peak[1]),'worst_nominal_time_s':float(times[peak[0]]),
        'max_root_relative_angle_rad':float(angle.max()),'max_root_angle_interval':int(angle.argmax()),
        'root_rotation_feature_max_abs':float(np.abs(features[:,0]).max()),
        'limitations':['Error is between canonical project FK and its own experimental decoder, not physical ground truth.',
                       'Root angle is not identical to the encoder skeleton-facing rotation.',
                       'No automatic correction or source training acceptance.'],'training_ready':False}
    with (HERE/'report.json').open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
