"""Bounded initialization diagnostic on one already sampled v7 Pick; no new samples."""
import hashlib, importlib.util, json, os, pathlib, subprocess, sys, time, traceback
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'repair_prior29_v2_20261105'
HERE=S/'src/paper_protocol_v7'
OUT=S/'paper_pick_seed_diagnostic_v1'
PLAN=S/'paper_pipeline_v7_launch/config.json'
EXPECTED_PLAN='a36fcedbfccbaeb8cd065457f72e76c601d233c6bd59759db32481c7fb68db74'
BANK=R/'stage3_pick29_20261103/reference_scale500_20261104/final_library_v1/reference_bank.json'
EXPECTED_BANK='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572'
CID=20000004
COHORT=S/'new_references_paper_v7/pick/cohort_0004'
sys.path.insert(0,str(HERE))
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=pathlib.Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def bind():
 assert sha(PLAN)==EXPECTED_PLAN
 plan=read(PLAN)
 for spec in plan['sources'].values():assert sha(spec['path'])==spec['sha256']
 return plan
def prepare():
 plan=bind()
 if OUT.exists(): print((OUT/'launch.json').read_text() if (OUT/'launch.json').exists() else 'Partial existing diagnostic; inspect');return
 OUT.mkdir()
 prior=read(COHORT/f'postopt/candidate_{CID}/report.json')
 from postopt_v2 import load_sample
 raw=load_sample(prior['source_sample'],prior['source_sample_sha256'])
 assert sha(BANK)==EXPECTED_BANK
 spec=importlib.util.spec_from_file_location('fk_readonly',plan['fk']['path']);fkmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(fkmod)
 fk=fkmod.URDFFK(plan['urdf']['path']);clock=np.arange(120);contact=int(raw['metadata']['grasp_frame'])
 rows=sorted(read(BANK)['references'],key=lambda r:np.linalg.norm(np.array(r['task_object_position'])-raw['metadata']['task_object_position']))[:12]
 compared=[]
 for row in rows:
  assert sha(row['path'])==row['sha256'] and sha(row['postopt_report'])==row['postopt_report_sha256']
  old_contact=round(row['grasp_time_s']*20)
  phase=np.interp(clock,[0,contact,contact+40,119],[0,old_contact,old_contact+40,119])
  with np.load(row['path'],allow_pickle=False) as z:
   names=z['joint_names'].tolist()
   q=np.column_stack([np.interp(phase,clock,x) for x in z['q'].T])
   pos=np.column_stack([np.interp(phase,clock,x) for x in z['root_pos'].T])
   rot=Slerp(clock,Rotation.from_quat(z['root_quat'][:,[1,2,3,0]]))(phase)
   root=np.c_[rot.as_quat()[:,[3,0,1,2]],pos]
  pts=fk.forward(pos,rot.as_quat(),q,names,links=fkmod.PAPER_22_LINKS)
  err=pts-raw['decoded_world']
  item={'path':row['path'],'sha256':row['sha256'],'native_proof':row['postopt_report'],'native_proof_sha256':row['postopt_report_sha256'],
        'raw_rmse_m':float(np.sqrt(np.mean(np.sum(err**2,axis=-1)))),'wrist_target_max_m':float(np.linalg.norm(pts[:,21]-raw['targets_world'][:,21],axis=-1).max()),
        'q':q,'root':root,'joint_names':names}
  compared.append(item)
 selected=min(compared,key=lambda r:r['raw_rmse_m'])
 with (OUT/'initial_state.npz').open('xb') as f:np.savez_compressed(f,q=selected['q'],root=selected['root'],joint_names=selected['joint_names'])
 record={'candidate_id':CID,'raw_path':prior['source_sample'],'raw_sha256':prior['source_sample_sha256'],
         'raw_wrist_max_m':prior['raw_wrist_target_max_m'],'baseline_stage_a_rmse_m':prior['stage_a_decoded_vector_rmse_m'],
         'baseline_report_sha256':sha(COHORT/f'postopt/candidate_{CID}/report.json'),
         'seed_state_sha256':sha(OUT/'initial_state.npz'),'selected':{k:v for k,v in selected.items() if k not in ['q','root','joint_names']},
         'compared':[{k:v for k,v in x.items() if k not in ['q','root','joint_names']} for x in compared],
         'scope':'Only optimizer initial state; exact original raw, paper conditions, losses and final filters remain unchanged.'}
 save(OUT/'cpu_report.json',record)
 env={**os.environ,**plan['postopt_env'],'PYTHONDONTWRITEBYTECODE':'1'}
 with (OUT/'controller.log').open('xb') as log:
  child=subprocess.Popen([plan['postopt_python'],'-B','-u',str(pathlib.Path(__file__).resolve()),'--controller'],cwd=S,env=env,
              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch={'pid':child.pid,'source_sha256':sha(__file__),'plan_sha256':EXPECTED_PLAN,'cpu_report_sha256':sha(OUT/'cpu_report.json')}
 save(OUT/'launch.json',launch);print(json.dumps(launch))
def worker():
 plan=bind();record=read(OUT/'cpu_report.json')
 assert sha(record['raw_path'])==record['raw_sha256'] and sha(OUT/'initial_state.npz')==record['seed_state_sha256']
 from postopt_v2 import MotionPreservingSolverV2
 solver=MotionPreservingSolverV2(plan['urdf']['path'],120,plan['fk']['path'])
 with np.load(OUT/'initial_state.npz',allow_pickle=False) as z:
  assert z['joint_names'].tolist()==solver.names
  q=np.clip(z['q'],solver.low+1e-6,solver.high-1e-6).astype(np.float32);root=z['root'].astype(np.float32)
 solver.initial_state=lambda decoded:(q.copy(),root.copy())
 folder=OUT/f'postopt/candidate_{CID}'
 result=solver.process(record['raw_path'],folder,record['raw_sha256'])
 result['initialization_diagnostic']={'wrapper_sha256':sha(__file__),'cpu_report_sha256':sha(OUT/'cpu_report.json'),
        'seed_state_sha256':record['seed_state_sha256'],'initial_state_only':True,'raw_or_target_modified':False}
 save(folder/'report.json',result)
 if result['status']=='failed_preserved':raise RuntimeError('solver exception; preserve report')
def controller():
 import pipeline_controller_v2 as p
 plan=bind();cfg=p.load_config(p.bound(plan['training_config']))
 try:
  save(OUT/'status.json',{'status':'waiting_for_current_pilot_outcome','new_samples':0,'candidate_id':CID})
  started=time.monotonic()
  while True:
   pilot=S/'new_references_paper_v7/pilot_receipt.json'
   if pilot.exists():
    result=read(pilot)
    if result['status']!='pilot_quality_budget_exhausted':
     save(OUT/'status.json',{'status':'not_needed_pilot_passed','new_samples':0});return
    launch=read(S/'paper_pipeline_v7_launch/launch.json')
    try:os.kill(launch['pid'],0)
    except ProcessLookupError:break
   if time.monotonic()-started>3600:raise TimeoutError('original pilot did not finish within diagnostic wait')
   time.sleep(10)
  command=[plan['postopt_python'],'-B','-u',str(pathlib.Path(__file__).resolve()),'--worker']
  p.phase(PLAN,EXPECTED_PLAN,plan,cfg,OUT,OUT,'postopt',command,os.environ.copy(),str(OUT))
  report=read(OUT/f'postopt/candidate_{CID}/report.json')
  if not report['passed_requested_kinematic_checks']:
   save(OUT/'status.json',{'status':'seed_diagnostic_rejected','report_sha256':sha(OUT/f'postopt/candidate_{CID}/report.json'),'new_samples':0});return
  rows=p.producer_records(COHORT/'raw','pick',[CID],report['checkpoint_sha256'])
  staged=p.stage_native(plan,'pick',OUT/'postopt',OUT/'native_input',rows)
  native=read(p.bound(plan['native_runtime']));script=p.bound(plan['sources']['native_pick'])
  env=p.runtime_env(native['env'],remove_visibility=True)
  env['PYTHONPATH']=plan['native_import_dir']+os.pathsep+str(script.parent)+os.pathsep+env.get('PYTHONPATH','')
  native_out=OUT/'native'
  argv=[native['python'],'-B','-u',str(script),'--headless','--device',plan['native_device'],
        '--output-dir',str(native_out),'--candidate-dir',str(staged[0][1])]
  p.phase(PLAN,EXPECTED_PLAN,plan,cfg,OUT,OUT,'native',argv,env,native['cwd'],(native_out/'report.json','native_replay_completed'))
  p.native_proofs('pick',staged,native_out)
  result={'status':'seed_and_native_passed' if rows[0]['admitted'] else 'native_rejected','new_samples':0,'records':rows,
          'generation_resumed':False,'report_sha256':sha(OUT/f'postopt/candidate_{CID}/report.json')}
  save(OUT/'result.json',result);save(OUT/'status.json',result)
 except BaseException:
  (OUT/'traceback.txt').write_text(traceback.format_exc());save(OUT/'status.json',{'status':'failed_preserved','new_samples':0});raise
if __name__=='__main__':
 {'--controller':controller,'--worker':worker}.get(sys.argv[1] if len(sys.argv)>1 else '',prepare)()

