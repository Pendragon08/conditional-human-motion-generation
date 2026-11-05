"""Read the actual500-reference candidate conditions, without resampling or GPU use."""
import ast,hashlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage2_prior29_20261103';O=S/'reference_scale500_20261104'
sys.path[:0]=[str(S/'src'),str(R/'training_disk_guard_20261101')]
from g1_fk import PAPER_22_LINKS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=O/'final_library_v1/reference_bank.json';m=json.loads(manifest.read_text());assert len(m['references'])==500
details=[];prompts=set();errors=[];robot_errors=[]
for item in m['references']:
 post=json.loads(Path(item['kinematic_report']).read_text());sample=Path(post['sample'])
 assert sha(sample)==post['sample_sha256']
 with np.load(sample,allow_pickle=False) as z:
  meta=json.loads(z['metadata_utf8'].tobytes());mask=z['target_mask'];target=z['target_points'];points=z['canonical_points'][0]
 assert mask.shape==(1,120,22) and target.shape==(120,22,3)
 expected=np.zeros((1,120,22),dtype=bool);expected[:,:,0]=True;expected[:,:,21]=True;expected[:,::5,4:6]=True
 assert np.array_equal(mask,expected)
 rotation=np.array(meta['canonical_to_world_rotation']);translation=np.array(meta['canonical_to_world_translation'])
 world=target@rotation.T+translation;obj=np.array(meta['task_object_position'])
 assert np.allclose(world[40,21],obj+[0,0,.04],atol=2e-7) and np.allclose(world[80,21],obj+[0,0,.2],atol=2e-7)
 assert np.allclose(world[80:,21],world[80,21],atol=2e-7)
 assert np.max(np.ptp(world[:,[0,4,5]],axis=0))<1e-7
 prompts.add(meta['caption']);error=np.linalg.norm(points[:,21]-target[:,21],axis=-1)
 errors.append([float(error[40]),float(error[80]),float(error.mean())])
 with np.load(item['path'],allow_pickle=False) as z:robot_points=z['keypoints']
 robot_error=np.linalg.norm(robot_points[:,21]-world[:,21],axis=-1)
 robot_errors.append([float(robot_error[40]),float(robot_error[80]),float(robot_error.mean())])
 details.append({'candidate_id':meta['candidate_id'],'sample':str(sample),'sample_sha256':sha(sample),
  'conditioned_joint_time_points':int(mask.sum()),'object_position_world_m':obj.tolist(),
  'right_wrist_target_at_2s_world_m':world[40,21].tolist(),'right_wrist_target_at_4s_world_m':world[80,21].tolist(),
  'raw_diffusion_wrist_error_at_2s_m':float(error[40]),'raw_diffusion_wrist_error_at_4s_m':float(error[80])})
cfg=json.loads((D/'config.json').read_text());gd=Path(cfg['trajgen'])/'diffusion/gaussian_diffusion.py'
text=gd.read_text().splitlines();callsite=[]
for i,line in enumerate(text):
 if 'self.guide(' in line:callsite.extend([{'line':j+1,'text':text[j]} for j in range(max(0,i-3),min(len(text),i+4))])
source=S/'src/stage3_sample_pick29_batch.py'
report={'status':'passed','scope':'actual500 final-library candidate conditions and source callsite; no new motion generation/physics screening',
 'reference_count':500,'reference_manifest_sha256':sha(manifest),'prompts':sorted(prompts),'fps':20,'states_per_clip':120,
 'joint_targets':[{'index':i,'link':PAPER_22_LINKS[i],'constrained_frames':120 if i in [0,21] else 24} for i in [0,4,5,21]],
 'hint_shape_per_clip':[120,66],'target_mask_shape_per_clip':[120,22],
 'spatial_gradient_hook_callsite':callsite,'diffusion_source_sha256':sha(gd),'sampling_source_sha256':sha(source),
 'postoptimization_is_separate':True,'raw_diffusion_wrist_errors_m':{'median_at_2s':float(np.median(np.array(errors)[:,0])),
  'max_at_2s':float(np.max(np.array(errors)[:,0])),'median_at_4s':float(np.median(np.array(errors)[:,1]))},
 'raw_target_error_is_not_postoptimization_error_or_grasp_success':True,'details':details}
report['postoptimized_reference_wrist_errors_m']={'median_at_2s':float(np.median(np.array(robot_errors)[:,0])),
 'max_at_2s':float(np.max(np.array(robot_errors)[:,0])),'median_at_4s':float(np.median(np.array(robot_errors)[:,1]))}
out=O/'diffusion_spatial_condition_audit_v2_20261104.json'
if not out.exists():out.open('x').write(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
print(json.dumps({'examples':details[:2]},indent=2))
