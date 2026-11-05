import collections,hashlib,json,pathlib,sys,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';sys.path[:0]=[str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_pick29_core import BODY_JOINTS
from g1_fk import URDFFK
out=B/'pilot_postopt_v4/independent_state_audit.json';assert not out.exists()
u=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf');fk=URDFFK(u)
limits={j.get('name'):j.find('limit') for j in ET.parse(u).getroot().findall('joint') if j.get('type')=='revolute'}
low=np.array([float(limits[k].get('lower')) for k in BODY_JOINTS]);high=np.array([float(limits[k].get('upper')) for k in BODY_JOINTS]);rows=[json.loads(x) for x in (B/'pilot_postopt_v4/records.jsonl').read_text().splitlines()];assert len(rows)==5
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();checked=[];rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
for r in rows:
 raw=B/'pilot_postopt_v4'/('candidate_'+str(r['candidate_id']).zfill(6))/'solver_raw.npz';assert sha(raw)==r['solver_raw_sha256'] and sha(r['sample'])==r['sample_sha256'] and sha(r['output'])==r['output_sha256']
 warm=B/'pilot_postopt_v2'/('candidate_'+str(r['candidate_id']).zfill(6))/'solver_raw.npz';assert sha(warm)==r['initial_state_sha256']
 with np.load(warm,allow_pickle=False) as z:
  iq=np.clip(z['q'].astype(float),low+1e-6,high-1e-6).astype(np.float32);assert hashlib.sha256(iq.tobytes()+z['root'].tobytes()).hexdigest()==r['initial_values_sha256']
 with np.load(raw,allow_pickle=False) as z:rawq=z['q'].copy()
 with np.load(r['output'],allow_pickle=False) as z,np.load(r['sample'],allow_pickle=False) as src:
  q=z['q'];assert z['joint_names'].tolist()==BODY_JOINTS and q.shape==(180,29) and np.allclose(z['sample_times'],np.arange(180)/20)
  assert np.array_equal(q,np.clip(rawq.astype(float),low+1e-6,high-1e-6).astype(np.float32)) and np.all((q>=low)&(q<=high))
  points,rot=fk.forward(z['root_pos'],z['root_quat'][:,[1,2,3,0]],q,BODY_JOINTS,return_rotations=True)
  assert np.max(np.abs(points-z['keypoints']))<1e-5 and np.array_equal(z['hand_closed'],src['planned_hand_closed'])
  goal=src['target_points']@rotation.T;desired=src['desired_wrist_quat_wxyz'];active=[1] if r['task']=='pour' else [0,1]
  err=np.linalg.norm(points[:,20:22]-goal[:,20:22],axis=-1)[:,active]
  angles=(Rotation.from_matrix(rot[:,20:22].reshape(-1,3,3)).inv()*Rotation.from_quat(desired.reshape(-1,4)[:,[1,2,3,0]])).magnitude().reshape(180,2)[:,active]
  assert abs(err.max()-r['max_hand_error_m'])<1e-7 and abs(angles.max()-r['max_wrist_angle_rad'])<1e-7
  assert r['checks']['hand_target']==bool(err.max()<.03) and r['checks']['wrist_orientation']==bool(angles.max()<.20)
  assert r['passed_requested_kinematic_checks']==all(r['checks'].values())
 checked.append({'id':r['candidate_id'],'task':r['task'],'max_hand_error_m':float(err.max()),'max_wrist_angle_rad':float(angles.max()),'kinematic_passed':r['passed_requested_kinematic_checks'],'failed_checks':[k for k,v in r['checks'].items() if not v]})
report={'status':'passed_independent_state_target_and_limit_audit','checked_candidates':5,'kinematic_passed':sum(x['kinematic_passed'] for x in checked),'final_reference_admitted':0,'checked':checked,'scope':'FK/time/limits/target/orientation/initialization/hand provenance; full native scene admission pending'}
out.open('x').write(json.dumps(report,indent=2));print(json.dumps(report))
