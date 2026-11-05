"""Independent file/noise/clock/FK/limit/native-proof checks for each cohort."""
import argparse,hashlib,json,pathlib,sys,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';sys.path[:0]=[str(B/'src'),str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_supported_task_conditions import build,PROMPTS
from stage3_pick29_core import BODY_JOINTS
from g1_fk import URDFFK
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['raw','robot','native'],required=True);p.add_argument('--producer',type=pathlib.Path,required=True);p.add_argument('--robot-records',type=pathlib.Path);p.add_argument('--native-dir',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert not a.output.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rows=lambda p:[json.loads(x) for x in pathlib.Path(p).read_text().splitlines()]
raw=rows(a.producer/'records.jsonl');ids=[r['candidate_id'] for r in raw];assert len(ids)==len(set(ids))
rawmap={r['candidate_id']:r for r in raw};report={'mode':a.mode,'candidates':len(raw),'final_training_ready':False}
urdf=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf');fk=URDFFK(urdf);rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
limits={j.get('name'):j.find('limit') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')=='revolute'};low=np.array([float(limits[k].get('lower')) for k in BODY_JOINTS]);high=np.array([float(limits[k].get('upper')) for k in BODY_JOINTS])
if a.mode=='raw':
 import torch
 torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.05,0)
 q=np.zeros((1,29))
 for side in ('left','right'):
  for part,value in [('hip_pitch',-.2),('knee',.42),('ankle_pitch',-.23),('elbow',.87),('shoulder_pitch',.35)]:q[0,BODY_JOINTS.index(side+'_'+part+'_joint')]=value
 q[0,BODY_JOINTS.index('left_shoulder_roll_joint')]=.16;q[0,BODY_JOINTS.index('right_shoulder_roll_joint')]=-.16
 initial,rot=fk.forward(np.array([[0,0,.84]]),np.array([[0,0,0,1.]]),q,BODY_JOINTS,return_rotations=True)
 with np.load(raw[0]['path'],allow_pickle=False) as z:task=json.loads(z['metadata_utf8'].tobytes())['task']
 expected=build(task,ids,20261104,initial[0],rot[0]);seen_noise=set();seen_features=set()
 for k,r in enumerate(raw):
  assert sha(r['path'])==r['sha256']
  with np.load(r['path'],allow_pickle=False) as z:
   m=json.loads(z['metadata_utf8'].tobytes());assert m['task']==task and m['caption']==PROMPTS[task] and m['fps']==20 and m['frames']==180
   assert m['checkpoint_sha256']=='109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9'
   assert m['task_conditions_source_sha256']==sha(B/'src/stage3_supported_task_conditions.py') and m['scene_sha256']==expected['scene_sha256']
   assert z['features'].shape==(1,180,263) and np.isfinite(z['features']).all() and np.isfinite(z['canonical_points']).all()
   assert np.allclose(z['target_points'],expected['world'][k]@rotation,atol=1e-7,rtol=0) and np.array_equal(z['target_mask'][0],expected['mask'][k])
   for key in ['desired_object_positions','desired_wrist_quat_wxyz','planned_hand_closed']:assert np.allclose(z[key],expected[key][k],atol=1e-10,rtol=0)
   f=hashlib.sha256(z['features'][0].tobytes()).hexdigest();assert f==r['feature_content_sha256']==m['feature_content_sha256']
   noise=torch.randn((1,263,1,180),generator=torch.Generator(device='cuda:0').manual_seed(20261104+r['candidate_id']),device='cuda:0');h=hashlib.sha256(noise[0].cpu().numpy().tobytes()).hexdigest();assert h==r['initial_noise_sha256']==m['initial_noise_sha256']
   assert f not in seen_features and h not in seen_noise;seen_features.add(f);seen_noise.add(h)
 report.update(status='passed_raw_noise_and_condition_audit',task=task)
else:
 robots=rows(a.robot_records);assert {r['candidate_id'] for r in robots}==set(ids);fingerprints={};passed=[]
 for r in robots:
  source=rawmap[r['candidate_id']];assert sha(r['sample'])==source['sha256']==r['sample_sha256']
  rawstate=pathlib.Path(r.get('output',str(pathlib.Path(a.robot_records).parent/'unused'))).parent/'solver_raw.npz'
  if r['status']=='generated_candidate_rejected':
   assert not r['passed_requested_kinematic_checks']
   choices=list((B/'bulk_v1').glob('**/candidate_'+str(r['candidate_id'])+'/solver_raw.npz'))
   match=next(x for x in choices if sha(x)==r['solver_raw_sha256'])
   with np.load(match,allow_pickle=False) as z:
    if r['rejection_reason']=='nonfinite_raw_solver_state':assert not (np.isfinite(z['q']).all() and np.isfinite(z['root']).all())
    else:
     correction=float(np.max(np.abs(np.clip(z['q'].astype(float),low+1e-6,high-1e-6).astype(np.float32)-z['q'])))
     assert r['rejection_reason']=='bound_correction_at_least1mrad' and correction>=.001 and abs(correction-r['rejection_metrics']['required_projection_rad'])<1e-12
   continue
  assert sha(r['output'])==r['output_sha256'] and sha(rawstate)==r['solver_raw_sha256']
  d=pathlib.Path(r['output']).parent;initial=next(x for x in [d/'initial_stage.npz',d/'warm_stage.npz'] if x.exists() and sha(x)==r['initial_state_sha256'])
  with np.load(initial,allow_pickle=False) as z:iq=np.clip(z['q'].astype(float),low+1e-6,high-1e-6).astype(np.float32);assert hashlib.sha256(iq.tobytes()+z['root'].tobytes()).hexdigest()==r['initial_values_sha256']
  with np.load(r['output'],allow_pickle=False) as z,np.load(r['sample'],allow_pickle=False) as src,np.load(rawstate,allow_pickle=False) as qraw:
   assert z['q'].shape==(180,29) and z['joint_names'].tolist()==BODY_JOINTS and float(z['fps'])==20 and np.allclose(z['sample_times'],np.arange(180)/20)
   assert np.array_equal(z['q'],np.clip(qraw['q'].astype(float),low+1e-6,high-1e-6).astype(np.float32)) and np.all((z['q']>=low)&(z['q']<=high))
   kp,rot=fk.forward(z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'],BODY_JOINTS,return_rotations=True);assert np.max(np.abs(kp-z['keypoints']))<1e-5
   active=[1] if r['task']=='pour' else [0,1];goals=src['target_points']@rotation.T
   err=np.linalg.norm(kp[:,20:22]-goals[:,20:22],axis=-1)[:,active];angle=(Rotation.from_matrix(rot[:,20:22].reshape(-1,3,3)).inv()*Rotation.from_quat(src['desired_wrist_quat_wxyz'].reshape(-1,4)[:,[1,2,3,0]])).magnitude().reshape(180,2)[:,active]
   assert abs(err.max()-r['max_hand_error_m'])<1e-7 and abs(angle.max()-r['max_wrist_angle_rad'])<1e-7
   assert r['checks']['hand_target']==bool(err.max()<.03) and r['checks']['wrist_orientation']==bool(angle.max()<.20)
   assert np.array_equal(z['hand_closed'],src['planned_hand_closed']) and np.allclose(z['planned_object_positions'],src['desired_object_positions'])
   assert r['passed_requested_kinematic_checks']==all(r['checks'].values())
   fp=hashlib.sha256(b''.join(np.ascontiguousarray(z[k]).tobytes() for k in ['q','root_pos','root_quat','hand_closed'])).hexdigest();fingerprints[str(r['candidate_id'])]=fp
  if r['passed_requested_kinematic_checks']:passed.append(r['candidate_id'])
 report.update(status='passed_robot_state_and_target_audit',kinematic_passed_ids=passed,state_fingerprints=fingerprints)
 if a.mode=='native':
  n=json.loads((a.native_dir/'report.json').read_text());assert n['status']=='native_task_replay_completed' and n['candidates']==len(passed)
  bank=json.loads((a.native_dir/'reference_bank.json').read_text());accepted=[];robotmap={r['candidate_id']:r for r in robots}
  for ref in bank['references']:
   r=robotmap[ref['candidate_id']];proof=pathlib.Path(ref['native_proof']);j=json.loads(proof.read_text())
   assert sha(proof)==ref['native_proof_sha256'] and sha(ref['path'])==ref['sha256']==r['output_sha256'] and r['candidate_id'] in passed
   assert j['eligible_for_calibrated_scene_reference'] and all(j['native_checks'].values()) and j['native_replay_frames']==180 and j['source_fps']==20
   assert j['source_state_sha256']==ref['sha256'] and j['scene_sha256']==r['scene_sha256'] and j['kinematic_report_sha256']==sha(ref['kinematic_report'])
   accepted.append({**ref,'state_fingerprint':fingerprints[str(r['candidate_id'])],'initial_noise_sha256':rawmap[r['candidate_id']]['initial_noise_sha256'],'feature_content_sha256':rawmap[r['candidate_id']]['feature_content_sha256']})
  assert len(accepted)==n['admitted_references'];report.update(status='passed_native_admission_binding_audit',accepted_references=accepted)
a.output.open('x').write(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['status','candidates']},indent=2))
