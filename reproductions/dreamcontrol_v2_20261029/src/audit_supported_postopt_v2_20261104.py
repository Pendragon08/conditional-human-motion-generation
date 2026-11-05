import hashlib,json,pathlib,sys,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';sys.path[:0]=[str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_pick29_core import BODY_JOINTS
from g1_fk import URDFFK
out=B/'pilot_postopt_v2/independent_state_audit.json';assert not out.exists()
u=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf');fk=URDFFK(u)
limits={j.get('name'):j.find('limit') for j in ET.parse(u).getroot().findall('joint') if j.get('type')=='revolute'}
low=np.array([float(limits[k].get('lower')) for k in BODY_JOINTS]);high=np.array([float(limits[k].get('upper')) for k in BODY_JOINTS]);rows=[json.loads(x) for x in (B/'pilot_postopt_v2/records.jsonl').read_text().splitlines()];assert len(rows)==6
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();checked=[]
for r in rows:
 raw=B/'pilot_postopt_v2'/('candidate_'+str(r['candidate_id']).zfill(6))/'solver_raw.npz';assert sha(raw)==r['solver_raw_sha256'] and sha(r['sample'])==r['sample_sha256']
 with np.load(raw,allow_pickle=False) as z:
  q=z['q'];assert q.shape==(180,29) and np.isfinite(q).all();correction=float(np.max(np.abs(np.clip(q.astype(float),low+1e-6,high-1e-6).astype(np.float32)-q)))
 if r['status']=='generated_candidate_rejected':
  assert r['rejection_reason']=='bound_correction_at_least1mrad' and correction>=.001 and abs(correction-r['rejection_metrics']['required_projection_rad'])<1e-12
 else:
  assert r['candidate_id']==101000 and r['passed_requested_kinematic_checks'] and sha(r['output'])==r['output_sha256']
  with np.load(r['output'],allow_pickle=False) as z:
   assert z['joint_names'].tolist()==BODY_JOINTS and z['q'].shape==(180,29) and np.allclose(z['sample_times'],np.arange(180)/20)
   points,rot=fk.forward(z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'],BODY_JOINTS,return_rotations=True)
   assert np.max(np.abs(points-z['keypoints']))<1e-5 and np.all((z['q']>=low)&(z['q']<=high))
   for side,index in [('left',20),('right',21)]:assert np.max(np.abs(Rotation.from_quat(z[side+'_wrist_quat'][:,[1,2,3,0]]).as_matrix()-rot[:,index]))<1e-5
   with np.load(r['sample'],allow_pickle=False) as src:assert np.array_equal(z['hand_closed'],src['planned_hand_closed'])
 checked.append({'candidate_id':r['candidate_id'],'raw_required_projection_rad':correction,'rejection_confirmed':r['status']=='generated_candidate_rejected','kinematic_pass':r['passed_requested_kinematic_checks']})
report={'status':'passed_independent_rejection_and_state_audit','checked_candidates':6,'kinematic_passed':1,'final_admitted_references':0,'checked':checked,'scope':'CPU raw-limit and state/FK/time/hand validation; native task scene pending'}
out.open('x').write(json.dumps(report,indent=2));print(json.dumps(report))
