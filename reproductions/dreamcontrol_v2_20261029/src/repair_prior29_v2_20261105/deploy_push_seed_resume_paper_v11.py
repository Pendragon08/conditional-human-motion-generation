"""Preserve all 23 sampled pilots and resume after the push initialization control."""
import fcntl, hashlib, importlib.util, json, os, subprocess, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
OLD=S/'src/paper_protocol_v10'; NEW=S/'src/paper_protocol_v11'
DEST=S/'paper_pipeline_v11_launch'; DIAG=S/'push_initialization_diagnostic_v1'
OLD_PLAN=S/'paper_pipeline_v10_launch/config.json'
OLD_PIDS=(1946814,1946815,1946816,1946817,1967045,1967052)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def bind(p): return {'path':str(p),'sha256':sha(p)}
def create(p,x):
 with Path(p).open('x',encoding='utf8') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False))
def change(body,a,b):
 assert body.count(a)==1,a
 return body.replace(a,b)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def absent(pid):
 p=Path(f'/proc/{pid}/cmdline');return not p.exists() or not p.read_bytes()

def audit(plan):
 assert sha(DIAG/'result.json')=='48bf1ad044b6cad55cec8007c67efa6b0fdd201c5c827ddf0a91fb16549af2c4'
 result=read(DIAG/'result.json');assert result['status']=='initialization_and_native_passed' and result['new_samples']==0
 row=result['records'][0];assert row['candidate_id']==20300007 and row['admitted']
 for key in ('source_sample','source','kinematic_report','native_proof'):assert sha(row[key])==row[key+'_sha256']
 report=read(row['kinematic_report']);proof=read(row['native_proof'])
 assert all(report['checks'].values()) and all(proof['native_checks'].values()) and proof['native_replay_frames']==180
 assert proof['kinematic_report_sha256']==row['native_compatibility_report_sha256']
 sys.path.insert(0,str(OLD));from postopt_v2 import load_sample
 raw=load_sample(row['source_sample'],row['source_sample_sha256'])
 fm=load('push_handoff_independent_fk',plan['fk']['path']);fk=fm.URDFFK(plan['urdf']['path'])
 with np.load(row['source'],allow_pickle=False) as z,np.load(DIAG/'initial_state.npz',allow_pickle=False) as seed:
  points=fk.forward(z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'],z['joint_names'].tolist(),links=fm.PAPER_22_LINKS)
  saved=float(np.max(np.abs(points-z['keypoints'])))
  fidelity=float(np.sqrt(np.mean(np.sum((points-raw['decoded_world'])**2,axis=-1))))
  wrist=float(np.linalg.norm(points[:,[20,21]]-raw['targets_world'][:,[20,21]],axis=-1)[raw['wrist_mask']].max())
  dq=float(np.max(np.abs(z['q']-seed['q'])));dp=float(np.linalg.norm(z['root_pos']-seed['root'][:,4:],axis=-1).max())
 assert saved<1e-5 and abs(fidelity-report['metrics']['decoded_point_vector_rmse_m'])<1e-7
 assert abs(wrist-report['metrics']['wrist_max_l2_m'])<1e-7 and wrist<.15 and dq>1e-6 and dp>1e-6
 create(DEST/'independent_push_readback.json',{'status':'independent_fk_and_native_passed','result':bind(DIAG/'result.json'),
  'raw_sha256':row['source_sample_sha256'],'output_sha256':row['source_sha256'],'native_proof_sha256':row['native_proof_sha256'],
  'raw_fidelity_rmse_m':fidelity,'wrist_max_m':wrist,'saved_points_max_difference_m':saved,
  'output_vs_seed_joint_max_rad':dq,'output_vs_seed_root_max_m':dp,'scope':'same new raw refit, not physical object success'})
 return row

def prepare():
 assert sha(OLD_PLAN)=='e074e9b85998355a6588a0d1d711495014e587b724adaa5ee70a307b3986798a'
 plan=read(OLD_PLAN)
 for entry in list(plan['sources'].values())+[plan['fk'],plan['urdf']]:assert sha(entry['path'])==entry['sha256']
 assert read(S/'new_references_paper_v10/status.json')['status']=='failed_preserved'
 assert all(absent(pid) for pid in OLD_PIDS)
 DEST.mkdir();NEW.mkdir();replacement=audit(plan)
 # Read the already recovered old prefix before adding the eight completed push cohorts.
 assert sha(plan['resume_manifest']['path'])==plan['resume_manifest']['sha256']
 prefix=read(plan['resume_manifest']['path']);rows=prefix['records'];cohorts=prefix['cohorts']
 assert len(rows)==15 and not any(r['task']=='box_push_pull' for r in rows)
 progress=read(S/'new_references_paper_v10/progress.json')
 pushes=[entry for entry in progress['cohorts'] if entry['task']=='box_push_pull'];assert len(pushes)==8
 for entry in pushes:
  assert sha(entry['path'])==entry['sha256'];original=read(entry['path']);assert len(original['records'])==1
  row=original['records'][0];entry=dict(entry)
  if row['candidate_id']==20300007:
   assert row['source_sample_sha256']==replacement['source_sample_sha256']
   row=dict(replacement);row['resumed_provenance']={'original_cohort':bind(entry['path']),'refit_receipt':bind(DIAG/'result.json')}
   entry.update(native_pass=1,kinematic_pass=1,refit_receipt=bind(DIAG/'result.json'))
  rows.append(row);cohorts.append(entry)
 assert dict(Counter(r['task'] for r in rows))=={'pick':8,'pour':2,'bimanual_box_lift':5,'box_push_pull':8}
 create(DEST/'resume_manifest.json',{'records':rows,'cohorts':cohorts,'pending_native':[],
  'next_cohort_numbers':{'pick':8,'pour':2,'bimanual_box_lift':5,'box_push_pull':8},
  'previous_manifest':plan['resume_manifest'],'previous_progress':bind(S/'new_references_paper_v10/progress.json')})
 # Cache old seed metadata once; original trajectories and proofs remain hash-bound.
 cpu=read(DIAG/'cpu_report.json');assert sha(DIAG/'cpu_report.json')=='2ff7b2bbb3911bd982df1a99a342533bf88bc809eed48d76353df9741824342a'
 assert sha(cpu['legacy_manifest_path'])==cpu['legacy_manifest_sha256']
 bank=[]
 for line in Path(cpu['legacy_manifest_path']).read_text().splitlines():
  r=json.loads(line)
  if r['task']!='box_push_pull':continue
  assert sha(r['path'])==r['sha256'] and sha(r['native_proof'])==r['native_proof_sha256']
  with np.load(r['path'],allow_pickle=False) as z:meta=json.loads(z['metadata_utf8'].tobytes())
  bank.append({**r,'task_object_position':meta['task_object_position'],'scene_sha256':meta['scene_sha256'],'phases':meta['task_plan']['phases']})
 assert len(bank)==500
 create(DEST/'push_initialization_bank.json',{'scope':'initialization only, never counted as new reference',
  'source':{'path':cpu['legacy_manifest_path'],'sha256':cpu['legacy_manifest_sha256']},'references':bank})
 for path in OLD.glob('*.py'):
  body=path.read_text();name=path.name
  if name=='pipeline_controller_v2.py':body=body.replace('verified_initialization_v10','verified_initialization_v11').replace('resume_prefix_paper_v10','resume_prefix_paper_v11')
  if name=='verified_initialization_v10.py':
   name='verified_initialization_v11.py'
   body=change(body,"    else:\n        return None\n    compared = []",'''    elif task == 'box_push_pull':
        bank = read(plan['push_initialization_bank'])
        center = np.asarray(sample['metadata']['task_object_position'])
        seeds = sorted(bank['references'], key=lambda r: (
            np.linalg.norm(np.asarray(r['task_object_position']) - center), r['candidate_id']))[:12]
        assert sample['frames'] == 180 and int(sample['metadata']['grasp_frame']) == 40
        for seed in seeds:
            assert seed['scene_sha256'] == sample['metadata']['scene_sha256']
            assert seed['phases'] == sample['metadata']['task_plan']['phases']
    else:
        return None
    compared = []''')
   body=change(body,"        if task == 'bimanual_box_lift':","        if task in ('bimanual_box_lift', 'box_push_pull'):")
   body=change(body,"            with np.load(seed['source'], allow_pickle=False) as z:","            with np.load(seed.get('path', seed.get('source')), allow_pickle=False) as z:")
   body=change(body,"        compared.append((error, seed.get('sha256', seed.get('source_sha256')), seed, q, root))",
    "        tie = seed['candidate_id'] if task == 'box_push_pull' else seed.get('sha256', seed.get('source_sha256'))\n        compared.append((error, tie, seed, q, root))")
   body=change(body,"    error, state_sha, seed, q, root = min(compared, key=lambda x: x[:2])",
    "    error, _, seed, q, root = min(compared, key=lambda x: x[:2])\n    state_sha = seed.get('sha256', seed.get('source_sha256'))\n    assert sha(seed.get('path', seed.get('source'))) == state_sha")
  if name=='resume_prefix_paper_v10.py':name='resume_prefix_paper_v11.py'
  create(NEW/name,body)
 for entry in plan['sources'].values():
  if Path(entry['path']).parent==OLD:
   name=Path(entry['path']).name.replace('verified_initialization_v10','verified_initialization_v11').replace('resume_prefix_paper_v10','resume_prefix_paper_v11')
   entry.update(bind(NEW/name))
 plan['resume_manifest']=bind(DEST/'resume_manifest.json');plan['push_initialization_bank']=bind(DEST/'push_initialization_bank.json')
 plan['push_initialization_contrast']=bind(DIAG/'result.json');plan['output_dir']=str(S/'new_references_paper_v11')
 plan['initialization_scope']='Pick/Pour/box unchanged. Push: nearest 12 verified same-scene/phase poses ranked by each new raw FK fit; full refit, unchanged conditions/losses/filters.'
 create(DEST/'config.json',plan)
 # Witness must reproduce the successful diagnostic start after the same limit clipping.
 fm=load('push_initialization_cpu_fk',plan['fk']['path']);fk=fm.URDFFK(plan['urdf']['path'])
 with np.load(DIAG/'initial_state.npz',allow_pickle=False) as z:names=z['joint_names'].tolist();eq=z['q'].copy();er=z['root'].copy()
 limits={j.get('name'):j.find('limit') for j in ET.parse(plan['urdf']['path']).getroot().findall('joint') if j.get('type')=='revolute'}
 links=sorted(fk.links);low=np.array([float(limits[n].get('lower')) for n in names]);high=np.array([float(limits[n].get('upper')) for n in names])
 solver=SimpleNamespace(fk=fk,names=names,all_links=links,kp=[links.index(n) for n in fm.PAPER_22_LINKS],low=low,high=high)
 from postopt_v2 import load_sample
 helper=load('verified_initialization_v11',NEW/'verified_initialization_v11.py')
 q,root,provenance=helper.prepare(plan,replacement,solver,load_sample(replacement['source_sample'],replacement['source_sample_sha256']))
 expected_q=np.clip(eq,low+1e-6,high-1e-6).astype(np.float32);expected_root=er.astype(np.float32)
 assert np.array_equal(q,expected_q) and np.array_equal(root,expected_root) and provenance['seed_sha256']==cpu['seed_source_sha256']
 create(DEST/'initializer_cpu_witness.json',{'status':'passed','production_matches_diagnostic_exactly':True,'no_gpu':True,'provenance':provenance})
 watchers=[]
 for name in ('evaluation_controller_paper_v10.py','supermimic_repaired_controller_paper_v10.py','report_video_sources_paper_v10.py'):
  path=S/'src'/name.replace('_v10','_v11');create(path,(S/'src'/name).read_text().replace('_paper_v10','_paper_v11'));watchers.append(path)
 for path in [*NEW.glob('*.py'),*watchers]:compile(path.read_text(),str(path),'exec')
 create(DEST/'source_binding.json',{str(path):sha(path) for path in [*NEW.glob('*.py'),*watchers,Path(__file__)]})
 sys.path.insert(0,str(NEW));p=load('v11_cpu_plan_validation',NEW/'pipeline_controller_v2.py');p.setup_plan(DEST/'config.json')
 from resume_prefix_paper_v11 import verify
 for row in rows:verify(p,row,replacement['checkpoint_sha256'])
 p.assert_no_duplicates(rows)
 assert p.reference_counts(rows)=={'pick':1,'pour':1,'bimanual_box_lift':2,'box_push_pull':1}
 create(DEST/'prelaunch.json',{'status':'ready','plan_sha256':sha(DEST/'config.json'),'admitted_counts':p.reference_counts(rows),'new_samples':0,'verified_raw_candidates':23,'all_four_pilots_passed':True})

def launch():
 with (DEST/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (DEST/'launch.json').exists():print((DEST/'launch.json').read_text());return
  assert not list(DEST.glob('*_launch.json')),'Inspect any partial launch'
  assert all(absent(pid) for pid in OLD_PIDS)
  for path,expected in read(DEST/'source_binding.json').items():assert sha(path)==expected,path
  plan=read(DEST/'config.json');sys.path.insert(0,str(NEW))
  p=load('v11_launch_resource_check',NEW/'pipeline_controller_v2.py');snapshot=p.resource_snapshot(S.parent);p.resource_guard(snapshot,plan['cohort_write_budget_bytes'])
  create(DEST/'actual_prelaunch_resources.json',snapshot)
  commands={'generation':[plan['sample_python'],'-B','-u',str(NEW/'pipeline_controller_v2.py'),'--plan',str(DEST/'config.json'),'--plan-sha256',sha(DEST/'config.json')]}
  for key,name in [('evaluation','evaluation_controller_paper_v11.py'),('supermimic','supermimic_repaired_controller_paper_v11.py'),('videos','report_video_sources_paper_v11.py')]:commands[key]=[plan['sample_python'],'-B','-u',str(S/'src'/name)]
  processes={}
  for key,command in commands.items():
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
   if key=='videos':env.update(plan['postopt_env']);env['JAX_PLATFORMS']='cpu';env['CUDA_VISIBLE_DEVICES']=''
   with (DEST/(key+'.log')).open('xb') as log:proc=subprocess.Popen(command,cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   processes[key]={'pid':proc.pid,'argv':command};create(DEST/(key+'_launch.json'),processes[key])
  create(DEST/'launch.json',{'status':'launched','processes':processes,'plan_sha256':sha(DEST/'config.json')});print((DEST/'launch.json').read_text())

if __name__=='__main__':
 if not DEST.exists():prepare()
 else:assert (DEST/'prelaunch.json').exists(),'Preserve incomplete preparation and inspect'
 if '--launch' in sys.argv:launch()
 else:print((DEST/'prelaunch.json').read_text())
