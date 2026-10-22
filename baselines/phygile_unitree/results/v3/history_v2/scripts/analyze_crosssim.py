"""Recompute raw errors and actual-observation policy parity for both engines."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import torch
from frozen_policy import FrozenPhyGile
p=argparse.ArgumentParser();p.add_argument('--local-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
assets=a.local_root/'private_assets';c=json.loads((assets/'contract.json').read_text(encoding='utf-8'))
policy=FrozenPhyGile(assets/'parent_checkpoint.pt',c,assets/'encoder_actor_critic.py');rows=[];manifest=[];all_parity=True
for i in range(4):
 pair=[];io=[]
 for engine,folder in [('isaac',a.local_root/'isaac_evaluations'/f'lab_eval_v1_clip{i:03d}'),('mjlab',a.local_root/f'mjlab_eval_v2_clip{i:03d}')]:
  r=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
  if r['status']!='FIRST_EPISODE_EVALUATED':raise RuntimeError('Incomplete receipt')
  cp=r.get('checkpoint_sha256',r.get('parent_sha256'))
  if cp!=c['checkpoint_sha256']:raise RuntimeError('Checkpoint mismatch')
  with np.load(folder/'raw_pre_reset.npz',allow_pickle=False) as d:
   if not np.array_equal(d['frame'],np.arange(r['frames_evaluated'])):raise RuntimeError('Frame continuity mismatch')
   metrics={'mpjpe_m_until_termination':float(np.linalg.norm(d['reference_body_pos']-d['robot_body_pos'],axis=-1).mean()),
    'mpjae_rad_until_termination':float(np.abs(d['reference_joint_pos']-d['robot_joint_pos']).mean()),
    'mpjve_rad_s_until_termination':float(np.abs(d['reference_joint_vel']-d['robot_joint_vel']).mean())}
   for k,v in metrics.items():
    if abs(r['metrics'][k]-v)>1e-10:raise RuntimeError('Metric recomputation mismatch')
   success=bool(len(d['frame'])==r['motion_frames'] and not d['terminated'].any())
   if success!=r['full_clip_success']:raise RuntimeError('Success mismatch')
  with np.load(folder/'policy_io_private.npz',allow_pickle=False) as d:
   actual=policy(torch.tensor(d['observations'])).numpy();expected=d['actions'];io.append(d['observations'].copy())
   parity={'max_abs_action_error':float(np.abs(actual-expected).max()),'allclose':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-5))}
   all_parity &= parity['allclose']
  pair.append({'engine':engine,'clip':i,'motion_sha256':r['motion_sha256'],'motion_frames':r['motion_frames'],'frames_evaluated':r['frames_evaluated'],
   'seconds_until_termination':r['frames_evaluated']*.02,'full_clip_success':success,'metrics':metrics,'actual_observation_policy_parity':parity,'reference_type':'dataset-derived kinematic reference','evaluation_type':'simulation_only'})
  for name in ['receipt.json','raw_pre_reset.npz','policy_io_private.npz']:
   path=folder/name;manifest.append({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'publish_arrays':False})
 if pair[0]['motion_sha256']!=pair[1]['motion_sha256']:raise RuntimeError('Motion mismatch across engines')
 blocks={};start=0
 for name,size in [('command',520),('anchor_orientation',6),('base_angular_velocity',3),('joint_position',29),('joint_velocity',29),('previous_action',29)]:
  blocks[name]=float(np.abs(io[0][:,start:start+size]-io[1][:,start:start+size]).max());start+=size
 rows.append({'clip':i,'engines':pair,'initial_observation_max_abs_difference_by_block':blocks})
result={'status':'RAW_METRICS_RECOMPUTED_AND_ACTUAL_INPUT_PARITY_PASSED' if all_parity else 'ACTUAL_INPUT_PARITY_FAILED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'checkpoint_sha256':c['checkpoint_sha256'],'seed':2026,'trials_per_clip_per_engine':1,'optimizer_steps':0,'control_dt':.02,'physics_dt':.005,'policy_observation_dims':616,'actions':29,
 'success_counts':{e:sum(x['engines'][j]['full_clip_success'] for x in rows) for j,e in enumerate(['isaac','mjlab'])},'clips':rows,
 'scope':'Four existing short held-out clips, one deterministic episode each. Failure-censored errors must be read with duration/success. This is a simulation-only framework/transfer pilot, not paper-scale validation, equivalence, or real-robot performance.'}
(a.output/'crosssim_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(a.output/'private_raw_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
if not all_parity:raise SystemExit(1)
