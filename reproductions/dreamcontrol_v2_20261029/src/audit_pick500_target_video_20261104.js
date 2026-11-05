const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const script=String.raw`import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');O=R/'stage3_pick29_20261103/reference_scale500_20261104';V=O/'target_report_video_v5'
sys.path.insert(0,str(R/'training_disk_guard_20261101'))
from g1_fk import URDFFK
urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads((V/'report.json').read_text());assert report['status']=='rendered_verified' and sha(V/report['video'])==report['video_sha256']
metrics=json.loads((V/'metrics.json').read_text());assert sha(V/'metrics.json')==report['metrics_sha256'] and len(metrics)==360
fk=URDFFK(urdf);out=[]
for item,source in zip(report['selected'],report['source_proofs']):
 assert sha(item['path'])==item['sha256'] and sha(source['sample'])==source['sha256']
 with np.load(item['path'],allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
 with np.load(source['sample'],allow_pickle=False) as z:
  m=json.loads(z['metadata_utf8'].tobytes());goal=(z['target_points']@np.asarray(m['canonical_to_world_rotation']).T+np.asarray(m['canonical_to_world_translation']))[:,21]
 points=fk.forward(s['root_pos'],s['root_quat'][:,[1,2,3,0]],s['q'],s['joint_names'].tolist(),links=['right_wrist_yaw_link'])[:,0]
 rows=sorted([r for r in metrics if r['candidate_id']==item['candidate_id']],key=lambda r:r['frame']);assert [r['frame'] for r in rows]==list(range(120))
 native=np.array([r['actual_wrist_world'] for r in rows]);saved_goal=np.array([r['requested_wrist_world'] for r in rows]);distance=np.array([r['wrist_distance_m'] for r in rows])
 assert np.max(np.abs(saved_goal-goal))<1e-10 and np.max(np.linalg.norm(native-points,axis=-1))<1e-4
 assert np.max(np.abs(distance-np.linalg.norm(native-goal,axis=-1)))<1e-10
 assert all(r['object_position_drift_m']<1e-6 for r in rows)
 error=np.linalg.norm(points-goal,axis=-1)
 out.append({'candidate_id':item['candidate_id'],'independent_wrist_error_at_2s_mm':float(error[40]*1000),
  'independent_wrist_error_at_4s_mm':float(error[80]*1000),'independent_maximum_error_mm':float(error.max()*1000),
  'native_numpy_FK_max_difference_m':float(np.linalg.norm(native-points,axis=-1).max()),
  'reported_native_error_2s_mm':float(distance[40]*1000),'reported_native_error_4s_mm':float(distance[80]*1000)})
result={'status':'passed','scope':'independent CPU URDF FK, immutable requested targets, native position/error overlay arithmetic and video hash; not task success',
 'render_report_sha256':sha(V/'report.json'),'urdf_sha256':sha(urdf),'metrics_checked':360,'references':out}
(V/'independent_audit.json').open('x').write(json.dumps(result,indent=2));print(json.dumps(result))
`;
const version=process.argv[2]||'v5';if(!['v5','v6'].includes(version))throw Error('unsupported artifact version');
const result=execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','env CUDA_VISIBLE_DEVICES= /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python -'],{input:script.replaceAll('target_report_video_v5','target_report_video_'+version),encoding:'utf8',timeout:30000});
const D=path.resolve(__dirname,version==='v6'?'../reports/pick29_targets_final_20261104':'../reports/pick29_targets_20261104');fs.mkdirSync(D,{recursive:true});fs.writeFileSync(path.join(D,'independent_audit.json'),result);process.stdout.write(result);
