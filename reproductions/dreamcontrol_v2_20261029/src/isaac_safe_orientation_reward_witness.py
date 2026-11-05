"""Compare isolated fallback prototype with the extracted old Pick function."""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import torch
from isaac_safe_orientation_reward import safe_target_orientation_error

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1); old=list(sys.argv);sys.argv=[str(Path(__file__).with_name('isaac_pick_reward_singularity_witness.py')),str(Path(sys.argv[1]).with_name('original_retest.json'))]
try:w=runpy.run_path(sys.argv[0])
finally:sys.argv=old
gen=torch.Generator().manual_seed(428);regular=torch.rand((100,3),generator=gen)*2-1;regular[:,0]+=.2
singular=torch.tensor([[0.,0.,0.],[0.,0.,1.],[1e-8,0.,1.],[1e-5,0.,1.]])
target=torch.cat([regular,singular]);results={}
for closed in (False,True):
    env=w['Obj']();env.scene=w['Scene'](len(target));env.device='cpu';env.episode_length_buf=torch.zeros(len(target));env.step_dt=.02;env.start_motion_times=torch.full((len(target),),2.);env.motion_ids=torch.arange(len(target));env.motion_lib=w['Library'](target,torch.full((len(target),),closed,dtype=torch.bool))
    base=w['fn'](env,w['Entity']());safe,valid=safe_target_orientation_error(env,w['Entity'](),math_utils=w['math_utils'],slerp=w['slerp'])
    assert torch.isfinite(base[:100]).all() and torch.isfinite(safe).all() and valid[:100].all()
    err=float(torch.max(torch.abs(base[:100]-safe[:100])));assert err<1e-6
    assert valid[100:].tolist()==[False,False,False,True]
    results[str(closed).lower()]={'regular_max_abs_difference':err,'singular_base_finite':torch.isfinite(base[100:]).tolist(),'safe_finite':torch.isfinite(safe[100:]).tolist(),'geometry_valid':valid[100:].tolist(),'safe_values':safe[100:].tolist()}
result={'random_seed':428,'regular_vectors':100,'singular_vectors':singular.tolist(),'epsilon_m':1e-6,'results':results,'baseline_retest_sha256':sha(Path(sys.argv[1]).with_name('original_retest.json')),'prototype_sha256':sha(Path(__file__).with_name('isaac_safe_orientation_reward.py')),'script_sha256':sha(Path(__file__)),'limitations':'Synthetic CPU mock, original math and quaternion functions, assumed fallback; not integrated into Isaac, no task or PhysX validation, v2 reward unspecified. Source target orientations are not independently verified.'}
with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
