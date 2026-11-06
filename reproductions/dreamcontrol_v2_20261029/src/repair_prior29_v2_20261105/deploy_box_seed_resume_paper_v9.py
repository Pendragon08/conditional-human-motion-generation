"""Versioned continuation after the same-sample box initialization comparison."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace
import numpy as np

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
OLD=S/'src/paper_protocol_v8'
NEW=S/'src/paper_protocol_v9'
DEST=S/'paper_pipeline_v9_launch'
DIAG=S/'box_initialization_diagnostic_v1'
OLD_PLAN=S/'paper_pipeline_v8_launch/config.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bind(p):return {'path':str(p),'sha256':sha(p)}
def create(p,x):
 with Path(p).open('x',encoding='utf8') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False))
def change(text,a,b):
 assert text.count(a)==1,a
 return text.replace(a,b)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def absent(pid):
 p=Path(f'/proc/{pid}/cmdline');return not p.exists() or not p.read_bytes()

def audit(plan):
 result=read(DIAG/'result.json');assert result['status']=='initialization_and_native_passed'
 row=result['records'][0];assert row['candidate_id']==20200000 and row['admitted']
 for k in ('source_sample','source','kinematic_report','native_proof'):assert sha(row[k])==row[k+'_sha256']
 report=read(row['kinematic_report']);proof=read(row['native_proof'])
 assert all(report['checks'].values()) and all(proof['native_checks'].values()) and proof['native_replay_frames']==180
 assert proof['kinematic_report_sha256']==row['native_compatibility_report_sha256']
 sys.path.insert(0,str(OLD))
 from postopt_v2 import load_sample
 raw=load_sample(row['source_sample'],row['source_sample_sha256'])
 fm=load('box_handoff_independent_fk',plan['fk']['path']);fk=fm.URDFFK(plan['urdf']['path'])
 with np.load(row['source'],allow_pickle=False) as z, np.load(DIAG/'initial_state.npz',allow_pickle=False) as seed:
  names=z['joint_names'].tolist();points=fk.forward(z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'],names,links=fm.PAPER_22_LINKS)
  saved=float(np.max(np.abs(points-z['keypoints'])))
  fidelity=float(np.sqrt(np.mean(np.sum((points-raw['decoded_world'])**2,axis=-1))))
  wrist=float(np.linalg.norm(points[:,[20,21]]-raw['targets_world'][:,[20,21]],axis=-1).max())
  dq=float(np.max(np.abs(z['q']-seed['q'])));dp=float(np.linalg.norm(z['root_pos']-seed['root'][:,4:],axis=-1).max())
 assert saved<1e-5 and abs(fidelity-report['metrics']['decoded_point_vector_rmse_m'])<1e-7
 assert abs(wrist-report['metrics']['wrist_max_l2_m'])<1e-7 and wrist<.15 and dq>1e-6 and dp>1e-6
 evidence={'status':'independent_fk_and_native_passed','result_sha256':sha(DIAG/'result.json'),'raw_sha256':row['source_sample_sha256'],
           'output_sha256':row['source_sha256'],'native_proof_sha256':row['native_proof_sha256'],'raw_fidelity_rmse_m':fidelity,
           'wrist_max_m':wrist,'saved_points_max_difference_m':saved,'output_vs_seed_joint_max_rad':dq,
           'output_vs_seed_root_max_m':dp,'scope':'initialization-only same-candidate contrast; no physical object success claim'}
 create(DEST/'independent_box_readback.json',evidence)
 return row

def prepare():
 assert sha(OLD_PLAN)=='eb3c79702a08589d7eb73a40605e1f0d1515254420bdfbd20bfbf9de72d2a0c2'
 plan=read(OLD_PLAN)
 for e in plan['sources'].values():assert sha(e['path'])==e['sha256']
 assert read(S/'box_initialization_handoff_v1/status.json')['status']=='prefix_preserved_for_versioned_handoff'
 assert absent(1916624) and absent(1937140)
 DEST.mkdir();NEW.mkdir()
 replacement=audit(plan)
 # All existing raw candidates remain represented exactly once.
 rows=[];cohorts=[]
 pick_replacement=read(plan['resumed_pick_result']['path'])['records'][0]
 for entry in plan['resume_pick_cohorts']:
  assert sha(entry['path'])==entry['sha256'];original=read(entry['path']);row=original['records'][0]
  if row['candidate_id']==20000004:row=dict(pick_replacement)
  row['resumed_provenance']={'original_cohort':entry,'refit_receipt':plan['resumed_pick_result'] if row['candidate_id']==20000004 else None}
  rows.append(row);cohorts.append({**entry,'task':'pick','generated':1,'native_pass':int(row['admitted']),'reused_existing_bytes':True})
 for task in ('pour','bimanual_box_lift'):
  paths=sorted((S/'new_references_paper_v8'/task).glob('cohort_*/result.json'))
  assert len(paths)==(2 if task=='pour' else 5)
  for rp in paths:
   result=read(rp);assert len(result['records'])==1;row=result['records'][0]
   if row['candidate_id']==20200000:
    assert row['source_sample_sha256']==replacement['source_sample_sha256']
    row=dict(replacement);row['resumed_provenance']={'original_cohort':bind(rp),'refit_receipt':bind(DIAG/'result.json')}
   rows.append(row);cohorts.append({**bind(rp),'task':task,'generated':1,'native_pass':int(row['admitted']),'reused_existing_bytes':True,
                                   'refit_receipt':bind(DIAG/'result.json') if row['candidate_id']==20200000 else None})
 pending=S/'new_references_paper_v8/box_push_pull/cohort_0000'
 rp=pending/'postopt/candidate_20300000/report.json'
 assert read(rp)['status'] in ('kinematic_candidate_ready_for_native_probe','generated_candidate_rejected')
 assert not (pending/'result.json').exists(),'A completed cohort needs its completed proof, not another replay'
 prefix={'records':rows,'cohorts':cohorts,'next_cohort_numbers':{'pick':8,'pour':2,'bimanual_box_lift':5,'box_push_pull':1},
         'pending_native':[{'task':'box_push_pull','cohort_number':0,'original_cohort':str(pending),'raw_bindings':bind(pending/'raw_bindings.json'),
                            'postopt_dir':str(pending/'postopt'),'postopt_reports':{'20300000':bind(rp)}}],
         'original_handoff':bind(S/'box_initialization_handoff_v1/worker_complete.json')}
 create(DEST/'resume_manifest.json',prefix)
 for source in OLD.glob('*.py'):
  body=source.read_text()
  if source.name=='pipeline_controller_v2.py':
   body=body.replace('from verified_initialization_v8 import process','from verified_initialization_v9 import process')
   body=change(body,'def run_pilots(add, checkpoint_sha, output, prior_pick):','def run_pilots(add, checkpoint_sha, output, prior_pilots):')
   body=change(body,'    tasks = {task: {"passed": False, "attempts": []} for task in TASKS}','    tasks = prior_pilots')
   body=change(body,'    tasks["pick"] = {"passed": True, "attempts": prior_pick}\n','')
   body=change(body,'        for _ in range(PILOT_CAP):','        for _ in range(max(0, PILOT_CAP - len(details["attempts"]))):')
   start=body.index('    rows, cohorts = [], []\n',body.index('def collect('));end=body.index('    def add(task, count):',start)
   body=body[:start]+'''    from resume_prefix_paper_v9 import resume
    rows, cohorts, offsets, numbers, prior_pilots = resume(
        sys.modules[__name__], plan_path, plan_sha, plan, cfg, output, checkpoint_sha, native_spec)
    largest_bytes_per_candidate = 1396736
'''+body[end:]
   body=change(body,'    run_pilots(add, checkpoint_sha, output, prior_pick)','    run_pilots(add, checkpoint_sha, output, prior_pilots)')
  create(NEW/source.name,body)
 helper=(OLD/'verified_initialization_v8.py').read_text()
 helper=change(helper,"    else:\n        return None\n    compared = []",'''    elif task == 'bimanual_box_lift':
        receipt = read(plan['box_initialization_receipt'])
        assert receipt['native_pass'] == 1
        seeds = receipt['records']
        assert len(seeds) == 1 and seeds[0]['candidate_id'] == 20200004
        assert int(sample['metadata']['grasp_frame']) == 40 and sample['frames'] == 180
        with np.load(seeds[0]['source'], allow_pickle=False) as z:
            metadata = json.loads(z['metadata_utf8'].tobytes())
        assert metadata['task_plan']['phases'] == sample['metadata']['task_plan']['phases']
        assert metadata['scene_sha256'] == sample['metadata']['scene_sha256']
    else:
        return None
    compared = []''')
 create(NEW/'verified_initialization_v9.py',helper)
 create(NEW/'resume_prefix_paper_v9.py',(S/'src/resume_prefix_paper_v9.py').read_text())
 for entry in plan['sources'].values():
  if Path(entry['path']).parent==OLD:entry.update(bind(NEW/Path(entry['path']).name))
 plan['sources']['verified_initialization']=bind(NEW/'verified_initialization_v9.py')
 plan['sources']['resume_prefix']=bind(NEW/'resume_prefix_paper_v9.py')
 plan['resume_manifest']=bind(DEST/'resume_manifest.json')
 plan['box_initialization_receipt']=bind(S/'new_references_paper_v8/bimanual_box_lift/cohort_0004/result.json')
 plan['box_initialization_contrast']=bind(DIAG/'result.json')
 plan['output_dir']=str(S/'new_references_paper_v9')
 plan['initialization_scope']='Pick and Pour v8 unchanged; box uses new validated 20200004 as initialization only, same raw refit, unchanged conditions/losses/filters; push retains default until evidence warrants a separate fix'
 create(DEST/'config.json',plan)
 # CPU witness: production helper reproduces the exact successful diagnostic start.
 fm=load('box_init_production_fk',plan['fk']['path']);fk=fm.URDFFK(plan['urdf']['path'])
 with np.load(DIAG/'initial_state.npz',allow_pickle=False) as z:names=z['joint_names'].tolist();eq=z['q'].copy();er=z['root'].copy()
 limits={j.get('name'):j.find('limit') for j in ET.parse(plan['urdf']['path']).getroot().findall('joint') if j.get('type')=='revolute'}
 links=sorted(fk.links)
 solver=SimpleNamespace(fk=fk,names=names,all_links=links,kp=[links.index(n) for n in fm.PAPER_22_LINKS],
                        low=np.array([float(limits[n].get('lower')) for n in names]),high=np.array([float(limits[n].get('upper')) for n in names]))
 from postopt_v2 import load_sample
 helpermod=load('verified_initialization_v9',NEW/'verified_initialization_v9.py')
 q,root,provenance=helpermod.prepare(plan,replacement,solver,load_sample(replacement['source_sample'],replacement['source_sample_sha256']))
 assert np.max(np.abs(q-eq))<2e-6 and np.max(np.abs(root-er))<2e-6
 create(DEST/'initializer_cpu_witness.json',{'status':'passed','production_matches_diagnostic':True,'no_gpu':True,'provenance':provenance})
 watchers=[]
 for name in ('evaluation_controller_paper_v8.py','supermimic_repaired_controller_paper_v8.py','report_video_sources_paper_v8.py'):
  newname=name.replace('_v8','_v9');body=(S/'src'/name).read_text()
  for prefix_name in ('new_references_paper','evaluation_controller_paper','evaluation_paper','supermimic_paper','report_video_sources_paper'):
   body=body.replace(prefix_name+'_v8',prefix_name+'_v9')
  target=S/'src'/newname;create(target,body);watchers.append(target)
 for path in [*NEW.glob('*.py'),*watchers]:compile(path.read_text(),str(path),'exec')
 create(DEST/'source_binding.json',{str(p):sha(p) for p in [*NEW.glob('*.py'),*watchers,Path(__file__)]})
 controller=load('v9_cpu_plan_validation',NEW/'pipeline_controller_v2.py');controller.setup_plan(DEST/'config.json')
 from resume_prefix_paper_v9 import verify
 for row in rows:verify(controller,row,replacement['checkpoint_sha256'])
 controller.assert_no_duplicates(rows)
 create(DEST/'prelaunch.json',{'status':'ready','plan_sha256':sha(DEST/'config.json'),'counts_before_pending_native':controller.reference_counts(rows),
                             'new_samples':0,'pending_native_only':20300000})

def launch():
 with (DEST/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (DEST/'launch.json').exists():print((DEST/'launch.json').read_text());return
  assert not list(DEST.glob('*_launch.json')),'Inspect partial launch'
  for pid in (1916624,1916625,1916626,1916627,1937140,1938522,1938529):assert absent(pid),f'Prior job still active {pid}'
  for path,expected in read(DEST/'source_binding.json').items():assert sha(path)==expected,path
  plan=read(DEST/'config.json');sys.path.insert(0,str(NEW))
  p=load('v9_launch_resource_check',NEW/'pipeline_controller_v2.py');snapshot=p.resource_snapshot(S.parent);p.resource_guard(snapshot,plan['cohort_write_budget_bytes'])
  create(DEST/'actual_prelaunch_resources.json',snapshot)
  commands={'generation':[plan['sample_python'],'-B','-u',str(NEW/'pipeline_controller_v2.py'),'--plan',str(DEST/'config.json'),'--plan-sha256',sha(DEST/'config.json')]}
  for key,name in [('evaluation','evaluation_controller_paper_v9.py'),('supermimic','supermimic_repaired_controller_paper_v9.py'),('videos','report_video_sources_paper_v9.py')]:
   commands[key]=[plan['sample_python'],'-B','-u',str(S/'src'/name)]
  processes={}
  for key,command in commands.items():
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
   if key=='videos':env.update(plan['postopt_env']);env['JAX_PLATFORMS']='cpu';env['CUDA_VISIBLE_DEVICES']=''
   with (DEST/(key+'.log')).open('xb') as log:
    proc=subprocess.Popen(command,cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   processes[key]={'pid':proc.pid,'argv':command};create(DEST/(key+'_launch.json'),processes[key])
  create(DEST/'launch.json',{'status':'launched','processes':processes,'plan_sha256':sha(DEST/'config.json')})
  print((DEST/'launch.json').read_text())

if __name__=='__main__':
 if not (DEST/'config.json').exists():prepare()
 if '--launch' in sys.argv:launch()
 else:print((DEST/'prelaunch.json').read_text())
