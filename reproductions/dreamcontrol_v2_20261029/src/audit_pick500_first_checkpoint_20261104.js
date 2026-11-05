const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const script=String.raw`import json,hashlib
from pathlib import Path
import torch
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
run=S/'runs/pick500_8912_3000_v1';path=run/'ppo/model_0.pt'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
before=sha(path);saved=torch.load(path,map_location='cpu',weights_only=True);prov=saved['infos']['provenance']
assert saved['iter']==0 and prov['reference_count']==500 and prov['num_envs']==8912 and prov['max_iterations']==3000
assert prov['reference_bank_sha256']=='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572'
assert len(prov['body_joint_names'])==29 and prov['policy_actions']==31 and prov['policy_observations']==102 and not prov['actor_reference_input']
assert prov['object_pose_policy']=='per_reference_fixed' and saved['optimizer_state_dict']['state']
assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
weights={k:list(v.shape) for k,v in saved['model_state_dict'].items() if k.startswith('actor.') and k.endswith('.weight')}
assert weights['actor.0.weight'][1]==102
last=sorted(weights,key=lambda k:int(k.split('.')[1]))[-1];assert weights[last][0]==31
assert sha(path)==before
report={'passed':True,'checkpoint':str(path),'checkpoint_sha256':before,'saved_iteration_index':0,'actual_completed_updates_in_checkpoint':1,
 'reference_count':500,'num_envs':8912,'body_dof':29,'actions':31,'actor_obs':102,'actor_reference_input':False,
 'finite_model_and_nonempty_real_optimizer_state':True,'actor_weight_shapes':weights,'provenance':prov,
 'scope':'read-only audit of actual first formal update; not task success or final training completion'}
out=O/'completion_controller_v1/first_checkpoint_audit.json';out.open('x').write(json.dumps(report,indent=2));print(json.dumps(report))
`;
const out=execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','env CUDA_VISIBLE_DEVICES= /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python -'],{input:script,encoding:'utf8',timeout:30000});
const D=path.resolve(__dirname,'../reports/pick500_formal_start_20261104');fs.mkdirSync(D,{recursive:true});fs.writeFileSync(path.join(D,'first_checkpoint_audit.json'),out);const j=JSON.parse(out);console.log(JSON.stringify({passed:j.passed,reference_count:j.reference_count,checkpoint_sha256:j.checkpoint_sha256,actor_weight_shapes:j.actor_weight_shapes}));
