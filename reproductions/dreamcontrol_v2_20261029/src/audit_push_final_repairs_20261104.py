import hashlib,json,pathlib,sys,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';sys.path[:0]=[str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_pick29_core import BODY_JOINTS
from g1_fk import URDFFK
out=B/'push_final_repair_cpu_audit.json';assert not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
u=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf');fk=URDFFK(u)
limits={j.get('name'):j.find('limit') for j in ET.parse(u).getroot().findall('joint') if j.get('type')=='revolute'};low=np.array([float(limits[k].get('lower')) for k in BODY_JOINTS]);high=np.array([float(limits[k].get('upper')) for k in BODY_JOINTS])
selection=[(102000,'push_postopt_v5','pilot_postopt_v4'),(102001,'push_postopt_v7','push_postopt_v6')];result=[];records=[];rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
for cid,version,warmversion in selection:
 d=B/version/f'candidate_{cid}';r=json.loads((d/'report.json').read_text());assert r['passed_requested_kinematic_checks'] and all(r['checks'].values())
 warm=B/warmversion/f'candidate_{cid}/solver_raw.npz';assert sha(warm)==r['initial_state_sha256'] and sha(d/'solver_raw.npz')==r['solver_raw_sha256']
 assert sha(r['sample'])==r['sample_sha256'] and sha(r['output'])==r['output_sha256']
 with np.load(warm,allow_pickle=False) as z:iq=np.clip(z['q'].astype(float),low+1e-6,high-1e-6).astype(np.float32);assert hashlib.sha256(iq.tobytes()+z['root'].tobytes()).hexdigest()==r['initial_values_sha256']
 with np.load(r['output'],allow_pickle=False) as z,np.load(r['sample'],allow_pickle=False) as src:
  assert z['q'].shape==(180,29) and z['joint_names'].tolist()==BODY_JOINTS and np.all((z['q']>=low)&(z['q']<=high)) and np.allclose(z['sample_times'],np.arange(180)/20)
  points,rot=fk.forward(z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'],BODY_JOINTS,return_rotations=True)
  goal=src['target_points']@rotation.T;err=np.linalg.norm(points[:,20:22]-goal[:,20:22],axis=-1)
  angles=(Rotation.from_matrix(rot[:,20:22].reshape(-1,3,3)).inv()*Rotation.from_quat(src['desired_wrist_quat_wxyz'].reshape(-1,4)[:,[1,2,3,0]])).magnitude()
  assert np.max(np.abs(points-z['keypoints']))<1e-5 and err.max()<.03 and angles.max()<.20
  assert abs(err.max()-r['max_hand_error_m'])<1e-7 and abs(angles.max()-r['max_wrist_angle_rad'])<1e-7
  assert np.array_equal(z['hand_closed'],src['planned_hand_closed']) and np.allclose(z['planned_object_positions'],src['desired_object_positions'])
 result.append({'id':cid,'max_wrist_error_m':float(err.max()),'max_wrist_angle_rad':float(angles.max()),'foot_height_max_m':r['maximum_foot_sole_height_m'],'state_sha256':r['output_sha256']});records.append(r)
report={'status':'passed_two_push_repairs_independent_state_audit','kinematic_passed':2,'final_admitted_references':0,'checked':result,'native_scene':'pending'};out.open('x').write(json.dumps(report,indent=2))
old=B/'staged_kinematic_pilots.json';staged=json.loads(old.read_text());assert len(staged['staged_candidates'])==4
(B/'staged4_before_push_repairs.json').open('x').write(json.dumps(staged,indent=2));staged['staged_candidates']+=records;staged['remaining_failures']=[];staged['kinematic_passed_counts']={'pour':2,'bimanual_box_lift':2,'box_push_pull':2};staged['audits'].append(str(out));staged['status']='six_kinematic_pilots_native_admission_pending';temp=old.with_suffix('.repaired.tmp');temp.write_text(json.dumps(staged,indent=2));temp.replace(old)
print(json.dumps(report))
