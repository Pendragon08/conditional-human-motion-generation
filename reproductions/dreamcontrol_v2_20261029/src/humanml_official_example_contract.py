"""Check one upstream HumanML example without claiming raw-data identity."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PUBLIC=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public')
HERE=Path(__file__).resolve().parent;COMMIT='9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);j=HERE/'joints.npy';v=HERE/'features.npy';notebook=HERE/'motion_representation.ipynb'
    assert sha(j)=='739df5e2e025d21ae5ce01ea7e0a8d3038448a46e88865d95a638e6242ee8b6b'
    assert sha(v)=='6e6fc0204ea9e48abdf94f84e8be22b19a1f1e955c7b98e025b1d702f6fe96a0'
    index=PUBLIC/'index.csv';assert sha(index)=='80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065'
    with index.open(newline='',encoding='utf8') as f:matches=[r for r in csv.DictReader(f) if r['new_name']=='012314.npy']
    assert len(matches)==1;row=matches[0];train=PUBLIC/'HumanML3D/train.txt';assert '012314' in set(train.read_text().splitlines())
    code='\n'.join(''.join(c.get('source',[])) for c in json.loads(notebook.read_text(encoding='utf8'))['cells'] if c.get('cell_type')=='code')
    required=["process_file(source_data, 0.002)","recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)","np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())","np.save(pjoin(save_dir2, source_file), data)"]
    assert all(s in code for s in required)
    joints=np.load(j,allow_pickle=False);features=np.load(v,allow_pickle=False);assert joints.shape==(170,22,3) and features.shape==(170,263)
    assert np.isfinite(joints).all() and np.isfinite(features).all()
    actual=recover_from_ric(torch.from_numpy(features).float(),22).numpy();absolute=np.abs(actual-joints)
    d=np.diff(joints[:,[7,10,8,11]],axis=0);foot=((d*d).sum(axis=2)<.002).astype(int)
    published=features[:-1,-4:];foot_difference=np.argwhere(foot!=published)
    out={'repository':'https://github.com/EricGuo5513/HumanML3D','commit':COMMIT,'joints_url':f'https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{COMMIT}/HumanML3D/new_joints/012314.npy','features_url':f'https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{COMMIT}/HumanML3D/new_joint_vecs/012314.npy','notebook_url':f'https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{COMMIT}/motion_representation.ipynb','joints_sha256':sha(j),'features_sha256':sha(v),'notebook_sha256':sha(notebook),'index_sha256':sha(index),'train_sha256':sha(train),'script_sha256':sha(Path(__file__)),'index_record':row,'published_split':'train','joints_shape':list(joints.shape),'features_shape':list(features.shape),'torch_recovery_max_abs_m':float(absolute.max()),'torch_recovery_exact':bool(np.array_equal(actual,joints)),'foot_threshold':.002,'foot_decisions_checked':int(foot.size),'foot_differences':int(len(foot_difference)),'foot_difference_indices':foot_difference.tolist(),'last_feature_velocity_and_foot_row_checked':False,'limitations':'One published example; notebook saves new_joints by recovering new_joint_vecs, so matching is not independent raw motion validation. Nominal index slice 171 poses assumes shown version and availability; original Eyes Japan intermediate, real FPS and SMPL+H source unavailable. Publication identity does not establish license for training. No robot/physics validation.'}
    with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
if __name__=='__main__':main()
