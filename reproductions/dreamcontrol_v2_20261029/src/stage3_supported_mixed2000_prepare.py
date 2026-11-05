"""Build a versioned native input library; never modify producers or start training.

Final mode requires all four hash-bound 500-reference selections. Preview mode
uses two existing pilots per class to exercise the different source contracts.
The original arrays retain hand, object, text and task metadata losslessly.
"""
import argparse,ast,collections,hashlib,json,os,pathlib,shutil,time,traceback
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
B=R/'stage3_supported_references_20261104';W=B/'bulk_v1'
N=R/'stage3_supermimic29_20261103/native_universal_20261104'
TASKS=['pick','pour','bimanual_box_lift','box_push_pull']
PRIOR='109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9'
PICK_MANIFEST='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572'
ASSET='13b283a83a2b250a3846d9950e0b641c959e3ce80fcaf706ea060ce0d8fb7888'
FLOOR=8*1024**3+456*1024**2
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines()]
def content(z):return hashlib.sha256(b''.join(np.ascontiguousarray(z[k]).tobytes() for k in ['q','root_pos','root_quat','hand_closed'])).hexdigest()
def atomic(p,j):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=pathlib.Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
 assert not a.output_dir.exists() and a.output_dir.resolve().is_relative_to(B.resolve())
 count=2 if a.preview else 500;started=time.monotonic()
 assert shutil.disk_usage('/data').free>FLOOR+150*1024**2
 pick_path=O/'final_library_v1/reference_bank.json';pick=read(pick_path)
 assert sha(pick_path)==PICK_MANIFEST and len(pick['references'])==500
 pick_audit_path=O/'final_library_v1/cpu_audit.json';pick_audit=read(pick_audit_path)
 assert pick_audit['manifest_sha256']==PICK_MANIFEST and pick_audit['reference_count']==500
 assert read(O/'final_library_audit.json')['status']=='passed'
 pick_details={x['candidate_id']:x for x in pick_audit['details']}
 prior_native_path=N/'native_motion_manifest.json';prior_native=read(prior_native_path)
 assert prior_native['source_manifest_sha256']==PICK_MANIFEST and len(prior_native['records'])==500
 old_csv={r['source']:r for r in prior_native['records']}
 bindings={str(pick_path):sha(pick_path),str(pick_audit_path):sha(pick_audit_path),str(prior_native_path):sha(prior_native_path)}
 selected={'pick':pick['references'][:count]};admitted=rows(W/'library_records.jsonl')
 admitted_map={(r['task'],r['candidate_id']):r for r in admitted}
 assert len(admitted_map)==len(admitted)
 if not a.preview:
  completion=read(W/'completion.json')
  assert completion['status']=='all1500_scene_references_admitted_final_mixed_loader_pending'
  bindings[str(W/'completion.json')]=sha(W/'completion.json')
 for task in TASKS[1:]:
  if a.preview:
   refs=sorted([r for r in admitted if r['task']==task],key=lambda r:r['candidate_id'])[:count]
  else:
   path=W/(task+'_500_scene_references.json');m=read(path)
   assert m['task']==task and m['reference_count']==500 and len(m['references'])==500
   bindings[str(path)]=sha(path);refs=m['references']
  assert len(refs)==count
  assert all(r==admitted_map[(task,r['candidate_id'])] for r in refs)
  selected[task]=refs
 native_robot=N/'source/supermimic/robot.py';tree=ast.parse(native_robot.read_text())
 order=ast.literal_eval(next(x.value for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_JOINT_NAMES' for t in x.targets)))
 assert len(order)==29 and len(set(order))==29
 bindings[str(native_robot)]=sha(native_robot)
 a.output_dir.mkdir(exist_ok=False);csv_dir=a.output_dir/'motion_csv20';csv_dir.mkdir()
 state={'status':'building_preview' if a.preview else 'building_final_inputs','pid':os.getpid(),'preview':a.preview,'formal_training_started':False,'references_completed':0}
 atomic(a.output_dir/'status.json',state);records=[];seen={k:set() for k in ['id','noise','features','state']};cached={}
 def proof_json(path):
  path=pathlib.Path(path)
  if str(path) not in cached:cached[str(path)]=read(path);bindings[str(path)]=sha(path)
  return cached[str(path)]
 try:
  for task_index,task in enumerate(TASKS):
   for index,ref in enumerate(selected[task]):
    assert shutil.disk_usage('/data').free>FLOOR+10*1024**2
    src=pathlib.Path(ref['path']);assert src.resolve().is_relative_to(R.resolve()) and sha(src)==ref['sha256']
    kinpath=pathlib.Path(ref['kinematic_report']);kin=proof_json(kinpath)
    nativepath=pathlib.Path(ref['postopt_report'] if task=='pick' else ref['native_proof']);native=proof_json(nativepath)
    assert sha(nativepath)==ref['postopt_report_sha256' if task=='pick' else 'native_proof_sha256']
    assert native['eligible_for_calibrated_scene_reference'] and all(native['native_checks'].values()) and native['asset_sha256']==ASSET
    assert sha(kinpath)==native['kinematic_report_sha256'] and kin['output_sha256']==ref['sha256']
    assert kin['passed_requested_kinematic_checks'] and all(kin['checks'].values()) and kin['max_joint_projection_rad']<.001
    assert sha(kin['sample'])==kin['sample_sha256']
    with np.load(src,allow_pickle=False) as z,np.load(kin['sample'],allow_pickle=False) as raw:
     meta=json.loads(z['metadata_utf8'].tobytes());assert meta==json.loads(raw['metadata_utf8'].tobytes())
     cid=int(meta['candidate_id']);assert cid==kin['candidate_id']
     frames=120 if task=='pick' else 180
     assert meta['checkpoint_sha256']==PRIOR and meta['fps']==20 and meta['frames']==frames
     assert z['q'].shape==(frames,29) and z['root_pos'].shape==(frames,3) and z['root_quat'].shape==(frames,4)
     assert z['hand_closed'].shape==(frames,2) and np.isin(z['hand_closed'],[0.,1.]).all()
     assert float(z['fps'])==20 and np.array_equal(z['sample_times'],np.arange(frames)/20)
     assert native['native_replay_frames']==frames and frames/20<=10
     names=z['joint_names'].tolist();assert len(names)==29 and set(names)==set(order)
     for key in ['q','root_pos','root_quat','keypoints','hand_closed']:assert np.isfinite(z[key]).all()
     assert np.max(np.abs(np.linalg.norm(z['root_quat'],axis=1)-1))<1e-4
     assert np.allclose(z['task_object_position'],ref.get('task_object_position',meta['task_object_position']),atol=1e-12,rtol=0)
     assert meta['scene_sha256']==native['scene_sha256']==kin['scene_sha256']
     feature=hashlib.sha256(raw['features'][0].tobytes()).hexdigest()
     assert feature==meta['feature_content_sha256']==kin['source_feature_content_sha256']
     assert meta['initial_noise_sha256']==kin['source_initial_noise_sha256']
     fingerprint=content(z)
     if task=='pick':
      detail=pick_details[cid];assert detail['state_sha256']==ref['sha256'] and detail['robot_state_content_sha256']==fingerprint
      assert detail['sample_sha256']==kin['sample_sha256'] and detail['native_report_sha256']==sha(nativepath)
      noise_evidence={'path':str(pick_audit_path),'sha256':sha(pick_audit_path),'candidate_id':cid,'noise_audit_sha256':detail['noise_recomputation_audit_sha256']}
     else:
      assert meta['task']==task and ref['initial_noise_sha256']==meta['initial_noise_sha256'] and ref['feature_content_sha256']==feature and ref['state_fingerprint']==fingerprint
      assert np.array_equal(z['hand_closed'],raw['planned_hand_closed']) and np.array_equal(z['planned_object_positions'],raw['desired_object_positions'])
      assert np.isfinite(z['planned_object_positions']).all() and z['planned_object_positions'].shape==(frames,3)
      if cid<1000000:
       auditpath=B/'pilot_v2/candidate_audit.json';audit=proof_json(auditpath)
       assert audit['status']=='passed_candidate_file_noise_and_task_binding_audit'
       row=next(x for x in audit['records'] if x['candidate_id']==cid and x['task']==task)
      else:
       cohort=next(x for x in pathlib.Path(kin['sample']).parents if x.parent==W/'cohorts')
       summary=proof_json(cohort/'summary.json');auditpath=cohort/'raw_audit.json';audit=proof_json(auditpath)
       assert summary['status']=='cohort_completed' and summary['raw_audit_sha256']==sha(auditpath)
       assert audit['status']=='passed_raw_noise_and_condition_audit' and audit['task']==task
       native_auditpath=cohort/'native_audit.json';na=proof_json(native_auditpath)
       assert summary['native_audit_sha256']==sha(native_auditpath) and na['status']=='passed_native_admission_binding_audit'
       assert ref in na['accepted_references']
       producerpath=cohort/'candidates/records.jsonl'
       if str(producerpath) not in cached:cached[str(producerpath)]=rows(producerpath);bindings[str(producerpath)]=sha(producerpath)
       row=next(x for x in cached[str(producerpath)] if x['candidate_id']==cid)
      assert row['sha256']==kin['sample_sha256'] and row['initial_noise_sha256']==meta['initial_noise_sha256'] and row['feature_content_sha256']==feature
      noise_evidence={'path':str(auditpath),'sha256':sha(auditpath),'candidate_id':cid,'source_row':row}
     for key,value in [('id',cid),('noise',meta['initial_noise_sha256']),('features',feature),('state',fingerprint)]:
      assert value not in seen[key],'duplicate '+key;seen[key].add(value)
     permutation=[names.index(n) for n in order]
     matrix=np.c_[z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'][:,permutation]]
     target=csv_dir/(str(task_index)+'_'+task+'_'+str(index).zfill(4)+'.csv')
     if task=='pick':
      old=old_csv[str(src)];assert old['source_sha256']==ref['sha256'] and old['body_joint_order']==order and old['source_fps']==20
      assert sha(old['csv'])==old['csv_sha256'];target.symlink_to(old['csv'])
     else:np.savetxt(target,matrix,delimiter=',',fmt='%.9g')
     reread=np.loadtxt(target,delimiter=',');error=float(np.max(np.abs(reread-matrix)))
     assert reread.shape==(frames,36) and error<1e-7
     record={'task':task,'candidate_id':cid,'csv':str(target),'csv_sha256':sha(target),'source':str(src),'source_sha256':ref['sha256'],
      'source_fps':20,'source_frames':frames,'last_source_time_s':float(z['sample_times'][-1]),'nominal_clip_duration_s':frames/20,
      'body_joint_order':order,'csv_root_quaternion_format':'xyzw','source_root_quaternion_format':'wxyz','positions':'world xyz metres, z up',
      'kinematic_report':str(kinpath),'kinematic_report_sha256':sha(kinpath),'native_proof':str(nativepath),'native_proof_sha256':sha(nativepath),
      'source_sample':kin['sample'],'source_sample_sha256':kin['sample_sha256'],'initial_noise_sha256':meta['initial_noise_sha256'],
      'feature_content_sha256':feature,'state_fingerprint':fingerprint,'noise_recomputation_evidence':noise_evidence,
      'csv_conversion_max_abs_error':error,'csv_reused_existing_pick_input':task=='pick','caption':meta['caption'],'metadata':meta,
      'hand_commands':{'source_array':'hand_closed','order':['left','right'],'fps':20,'values':'0 open, 1 closed','resampling':'zero-order hold on source timeline'},
      'object_metadata':{'initial_position':z['task_object_position'].tolist(),'initial_quaternion_wxyz':z['task_object_quaternion_wxyz'].tolist(),
       'planned_positions_array':'planned_object_positions' if task!='pick' else None,'planned_motion_is_not_simulated_dynamics':True},
      'scope':'body29 native tracker input plus original hand/object/text references; no dynamic task-success claim'}
     records.append(record)
    state.update(references_completed=len(records),task=task,elapsed_seconds=time.monotonic()-started);atomic(a.output_dir/'status.json',state)
  counts=dict(collections.Counter(r['task'] for r in records));assert counts=={t:count for t in TASKS}
  manifest={'status':'preview_inputs_native_loader_pending' if a.preview else 'final2000_inputs_native_loader_pending','preview':a.preview,
   'reference_count':len(records),'per_task':counts,'body_dof':29,'source_fps':20,'native_control_fps':50,'body_joint_order':order,
   'records':records,'input_bindings':bindings,'preparer_sha256':sha(__file__),'formal_training_started':False,
   'native_output_scope':'29 body actions, reference-conditioned; hands and objects retained in original files and metadata, not native learned actions',
   'required_next':'independent CSV reread and actual full native MotionLib loading / resampling / FK / future-goal checks'}
  (a.output_dir/'native_motion_manifest.json').open('x').write(json.dumps(manifest,indent=2,allow_nan=False))
  state.update(status=manifest['status'],manifest_sha256=sha(a.output_dir/'native_motion_manifest.json'),per_task=counts,
   independent_source_proof_clock_and_global_dedup_passed=True,elapsed_seconds=time.monotonic()-started)
  atomic(a.output_dir/'status.json',state);print(json.dumps(state),flush=True)
 except BaseException as e:
  state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());atomic(a.output_dir/'status.json',state);raise

if __name__=='__main__':main()
