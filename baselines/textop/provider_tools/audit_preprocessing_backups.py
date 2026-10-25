"""Read persistent private archives without extracting them or accepting data.

Checks every NPZ hash and shape, selected69D fields, annotations, mirror mapping,
and exact original-index source/split coverage. Only aggregate results are
public. Missing/cohort-in-progress backups produce an incomplete result, never
a preprocessing-complete or physically-accepted result.
"""
import argparse,collections,datetime,hashlib,io,json,os,pathlib,sys,tarfile,time,traceback
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
from session import REPO
sys.path.insert(0,str(REPO/'baselines/textop/scripts'))
from reconcile_raw_sources import source_key

BASE=REPO/'baselines/textop'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def require(value,message):
 if not value:raise AssertionError(message)

def arrays(blob):
 with np.load(io.BytesIO(blob),allow_pickle=False) as z:
  names={'qpos_29','contact','features_69','mirrored_qpos_29','mirrored_contact','mirrored_features_69','times_s'}
  require(set(z.files)==names,'Unexpected NPZ field set')
  d={k:z[k] for k in names}
 q=d['qpos_29'];n=len(q);require(n>=2 and q.shape==(n,36),'Invalid pose dimensions')
 shapes={'contact':(n,2),'features_69':(n-1,69),'mirrored_qpos_29':(n,36),'mirrored_contact':(n,2),'mirrored_features_69':(n-1,69),'times_s':(n,)}
 require(all(d[k].shape==shape for k,shape in shapes.items()),'Invalid feature/contact/timestamp dimensions')
 require(all(np.isfinite(v).all() for v in d.values()),'Nonfinite archived arrays')
 t=d['times_s'];require(abs(float(t[0]))<1e-10 and np.allclose(np.diff(t),.02,rtol=0,atol=1e-9),'Invalid50Hz timeline')
 errors=[]
 for prefix in ['','mirrored_']:
  pose=d[prefix+'qpos_29'];feature=d[prefix+'features_69'];contact=d[prefix+'contact']
  require(np.isin(contact,[0,1]).all(),'Nonbinary contact data')
  require(np.allclose(np.linalg.norm(pose[:,3:7],axis=1),1,rtol=0,atol=1e-5),'Nonunit pose quaternions')
  checks=[(feature[:,5:7],contact[:-1]),(feature[:,10],pose[:-1,2]),(feature[:,11:40],pose[:-1,7:]),(feature[:,40:69],np.diff(pose[:,7:],axis=0))]
  for actual,expected in checks:
   error=float(np.max(np.abs(actual-expected)));require(error<1e-5,'69D contact/height/joint/difference mismatch');errors.append(error)
 mirror=d['mirrored_qpos_29'];expected=q[:,:7].copy();expected[:,1]*=-1;expected[:,[4,6]]*=-1
 require(np.allclose(expected,mirror[:,:7],rtol=0,atol=1e-8),'Mirror root mismatch')
 return {'frames':n,'selected69_fields_max_error':max(errors),'sha256':hashlib.sha256(blob).hexdigest()}

def audit_cohort(config,archive,expected_digest,index,seen):
 require(sha(archive)==expected_digest,'Private archive SHA mismatch')
 rows=None;progress=None;found={};total_frames=0;max_error=0.
 with tarfile.open(archive,mode='r|gz') as t:
  for member in t:
   require(not member.issym() and not member.islnk(),'Archive links unsupported')
   if not member.isfile():continue
   leaf=pathlib.PurePosixPath(member.name).name
   if leaf=='prepared_manifest.private.jsonl':
    require(rows is None,'Multiple prepared manifests');rows=[json.loads(line) for line in t.extractfile(member).read().decode().splitlines()]
   elif leaf=='progress.json':
    require(progress is None,'Multiple progress receipts');progress=json.loads(t.extractfile(member).read().decode())
   elif leaf.endswith('.private.npz'):
    require(leaf not in found,'Duplicate NPZ basename');value=arrays(t.extractfile(member).read());found[leaf]=value
    total_frames+=value['frames'];max_error=max(max_error,value['selected69_fields_max_error'])
 require(rows is not None and progress is not None,'Missing archived manifest or progress')
 target=config['target'];require(progress['status']=='PREPROCESSING_FINISHED_FILTER_PENDING','Archived preprocessing was incomplete')
 require(progress['failed_sequences']==0 and progress['completed_sequences']==target and progress['total_requested']==target,'Archived preprocessing count/failure mismatch')
 require(progress['physical_filter_passed'] is False and progress['ready_for_generator_training'] is False,'Unexpected acceptance claim in preprocessing')
 require(len(rows)==len(found)==target,'Manifest/payload/target counts disagree')
 payloads=set();splits=collections.Counter();paired=0;rejected=0
 for row in rows:
  key=row['canonical_source'];require(key in index and key not in seen,'Unexpected or duplicate source key')
  require(row['split']==index[key] and row['dof']==29 and row['features']==69 and row['physical_filter_passed'] is False,'Source split/dimensions/acceptance mismatch')
  leaf=pathlib.PurePosixPath(row['output_file']).name;require(leaf not in payloads and leaf in found,'Missing/duplicate payload')
  value=found[leaf];require(value['sha256']==row['output_sha256'] and value['frames']==row['frames50'],'Per-motion payload SHA/frame count mismatch')
  require(max(row['roundtrip'].values())<1e-3 and max(row['mirror_roundtrip'].values())<1e-3 and row['mirror_fk_error_m']<1e-3,'Recorded codec or mirror check failed')
  names=row['joint_names'];require(len(names)==len(set(names))==29,'Invalid joint name mapping')
  # Both original and mirrored labels must use the same bounded50Hz timeline.
  for group in ['annotations','mirrored_annotations']:
   for ann in row[group]:
    lo=ann['start_frame'];hi=ann['end_frame_exclusive']
    require(type(lo) is int and type(hi) is int and 0<=lo<hi<=value['frames'],'Annotation bounds invalid')
    require(abs(ann['start_s']-lo/50)<1e-8 and abs(ann['end_s']-hi/50)<1e-8,'Annotation frame/time mismatch')
   paired+=len(row[group])
  require(len(row['annotations'])==len(row['mirrored_annotations']),'Mirror annotation counts differ')
  rejected+=len(row['rejected_annotations']);seen.add(key);payloads.add(leaf);splits[row['split']]+=1
 require(payloads==found.keys(),'Orphaned archived payloads')
 return {'id':config['id'],'status':'PERSISTENT_PREPROCESSING_BACKUP_AUDIT_PASSED','source_sequences':target,'mirrored_sequences':target,
  'archive_bytes':archive.stat().st_size,'archive_sha256':expected_digest,'per_motion_sha_verified':len(found),
  'pose_frames_including_mirrors':total_frames*2,'paired_annotations_including_mirrors':paired,'rejected_annotation_count':rejected,
  'selected69_fields_max_error':max_error,'splits':dict(splits)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 require(a.run.replace('_','').isalnum(),'Invalid run name')
 out=BASE/'results/preprocessing_backup_audit_20261025'/a.run;out.mkdir(parents=True,exist_ok=False)
 private=LOCAL/'datasets/preprocessing_audit_private_20261025'/a.run;private.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();receipt={'status':'RUNNING_READ_ONLY_BACKUP_AUDIT','read_only':True,'dataset_accepted':False,
 'physical_filter_executed':False,'full_generator_training_started':False,'cohorts':[],'pending_cohorts':[],
 'source_sha256':sha(pathlib.Path(__file__)),'output_writers_closed':False}
 def save():
  receipt['elapsed_seconds']=time.monotonic()-started
  receipt['time_beijing']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
  (out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 try:
  cfg_path=BASE/'configs/preprocessing-completion-audit.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
  plan=json.loads((BASE/'configs/paper-step2.json').read_text(encoding='utf-8'));profile=plan['data_scope']
  receipt['spec_sha256']=sha(cfg_path);receipt['numpy_version']=np.__version__
  require(cfg['available_target_sources']==sum(x['target'] for x in cfg['cohorts'])==profile['available_target_sources'],'Target profile mismatch')
  root=LOCAL/'datasets/BABEL';original=root/'textop_original_verified_pairs.private.json';missing=root/'cmu_missing_selected_export.private.json'
  require(sha(original)==profile['original_index_sha256'] and sha(missing)==profile['unavailable_index_sha256'],'Source-index SHA mismatch')
  originals=json.loads(original.read_text(encoding='utf-8'));unavailable=json.loads(missing.read_text(encoding='utf-8'))
  mapping={source_key(x['source']):x['split'] for x in originals};absent={source_key(x['source']) for x in unavailable}
  require(len(mapping)==len(originals)==profile['requested_index_sources'] and len(absent)==len(unavailable)==cfg['unavailable_index_sources'] and absent<=mapping.keys(),'Source-index count/duplicate mismatch')
  expected={k:v for k,v in mapping.items() if k not in absent};seen=set()
  for cohort in cfg['cohorts']:
   archive=LOCAL/'datasets'/cohort['archive'];digest=cohort.get('sha256')
   if 'state' in cohort:
    state_file=BASE/cohort['state'];state=json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
    if state.get('status')!='PREPROCESSING_BACKED_UP_FILTER_PENDING':receipt['pending_cohorts'].append(cohort['id']);save();continue
    digest=state['archive_sha256']
   require(archive.is_file() and digest,'Claimed persistent backup not found')
   result=audit_cohort(cohort,archive,digest,expected,seen);receipt['cohorts'].append(result)
   receipt['verified_sources']=len(seen);save();print(json.dumps({'cohort':cohort['id'],'verified_sources':len(seen)}),flush=True)
  full=not receipt['pending_cohorts'] and seen==expected.keys()
  require(len(seen)<=len(expected),'Prepared source overflow')
  if not receipt['pending_cohorts']:require(full,'All backups exist but source coverage is incomplete')
  receipt.update(status='PREPROCESSING_ALL_AVAILABLE_SOURCES_BACKED_UP_VERIFIED_FILTER_PENDING' if full else 'PREPROCESSING_INCOMPLETE_AVAILABLE_SOURCES_PENDING',
   preprocessing_complete=full,available_target_sources=len(expected),verified_sources=len(seen),remaining_available_sources=len(expected)-len(seen),
   unavailable_index_sources=len(absent),author_private_clips_unavailable=profile['author_private_clips_unavailable'],paper_data_scope_exact=False,
   output_writers_closed=True,limitation='Private persistent preprocessing backups only; no dynamic tracking filter, accepted corpus or training')
  # Per-source identifiers remain private even on successful verification.
  (private/'verified_sources.private.json').write_text(json.dumps(sorted(seen)),encoding='utf-8');save()
  print(json.dumps({k:receipt[k] for k in ['status','verified_sources','remaining_available_sources','preprocessing_complete']}),flush=True)
 except BaseException:
  (private/'failure.private.txt').write_text(traceback.format_exc(),encoding='utf-8')
  receipt.update(status='FAILED_PREPROCESSING_BACKUP_AUDIT',preprocessing_complete=False,output_writers_closed=True,
   error='Read-only backup verification failed; full details retained privately, no inputs or other workers changed');save();raise

if __name__=='__main__':main()
