"""Read-only exact-source/FK binding for the available public filter library.

Private per-motion rows are returned in process only. The public receipt contains
counts and aggregate digests. Input completeness is not physics acceptance.
"""
import argparse,hashlib,json,os,pathlib,time
from reconcile_raw_sources import source_key

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
REFERENCES=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(pathlib.Path(path).read_text())

def inspect(cap,profile,require_complete=False):
 assert cap['profile_id']==profile['profile_id'] and profile['paper_data_scope_exact'] is False
 scope=pathlib.Path(cap['scope_input_directory'])
 original=scope/'textop_original_verified_pairs.private.json';unavailable=scope/'cmu_missing_selected_export.private.json'
 assert sha(original)==profile['original_index_sha256'] and sha(unavailable)==profile['unavailable_index_sha256']
 rows=read(original);missing_rows=read(unavailable)
 declared={source_key(x['source']):x['split'] for x in rows};unavailable_keys={source_key(x['source']) for x in missing_rows}
 assert len(declared)==len(rows)==profile['requested_index_sources']
 assert len(unavailable_keys)==len(missing_rows)==profile['unavailable_index_sources'] and unavailable_keys<=declared.keys()
 expected={k:v for k,v in declared.items() if k not in unavailable_keys}
 assert len(expected)==cap['expected_original_sources']==profile['available_target_sources']
 originals=[];seen=set();cohorts=[];pending=[];body_names=None;joint_names=None
 for cohort in cap['cohorts']:
  prepared_paths=[pathlib.Path(x) for x in cohort['prepared']];folder=REFERENCES/cohort['run']
  if any(not (p/'progress.json').exists() or read(p/'progress.json').get('status')!='PREPROCESSING_FINISHED_FILTER_PENDING' for p in prepared_paths):
   pending.append({'run':cohort['run'],'reason':'preprocessing_pending'});continue
  if not (folder/'receipt.json').exists() or read(folder/'receipt.json').get('status')!='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED':
   pending.append({'run':cohort['run'],'reason':'reference_export_pending'});continue
  prepared={};prepared_hashes=[]
  for p in prepared_paths:
   progress=read(p/'progress.json');assert progress['failed_sequences']==0
   manifest=p/'prepared_manifest.private.jsonl';prepared_hashes.append(sha(manifest));count=0
   for line in manifest.read_text().splitlines():
    row=json.loads(line);key=row['canonical_source'];count+=1
    assert key in expected and key not in prepared and key not in seen
    assert row['split']==expected[key] and row['dof']==29 and row['features']==69 and row['frames50']>=2
    output=pathlib.Path(row['output_file']);assert output.parent==p and output.is_file() and not output.is_symlink()
    assert sha(output)==row['output_sha256'];prepared[key]=row
   assert count==progress['completed_sequences']
  assert len(prepared)==cohort['sources']
  meta=read(folder/'receipt.json');manifest=folder/'manifest.private.jsonl'
  assert meta['output_writers_closed'] and meta['physics_steps']==0 and meta['fps']==50
  assert meta['exported_source_sequences']==len(prepared) and meta['exported_trajectories']==2*len(prepared)
  assert sorted(meta['input_manifest_hashes'])==sorted(prepared_hashes) and sha(manifest)==meta['private_manifest_sha256']
  assert len(meta['joint_names'])==29 and len(meta['body_names'])==30
  if body_names is None:body_names=meta['body_names'];joint_names=meta['joint_names']
  assert meta['body_names']==body_names and meta['joint_names']==joint_names
  variants={};frames=0;original_frames=0;reference_rows=0;cohort_originals=[]
  for line in manifest.read_text().splitlines():
   row=json.loads(line);key=row['canonical_source'];assert key in prepared and type(row['mirrored']) is bool
   pair=(key,row['mirrored']);assert pair not in variants;variants[pair]=True
   prior=prepared[key]
   assert row['split']==prior['split'] and row['frames']==prior['frames50']
   assert row['prepared_file']==prior['output_file'] and row['prepared_sha256']==prior['output_sha256']
   path=pathlib.Path(row['file']);assert path.parent==folder and not path.is_symlink() and path.stat().st_uid==1013
   assert sha(path)==row['sha256'];frames+=row['frames'];reference_rows+=1
   if not row['mirrored']:
    original_frames+=row['frames'];cohort_originals.append(row)
  assert len(variants)==2*len(prepared) and all((k,False) in variants and (k,True) in variants for k in prepared)
  assert frames==meta['exported_frames'] and len(cohort_originals)==len(prepared)
  originals+=cohort_originals;seen.update(prepared)
  cohorts.append({'run':cohort['run'],'original_sources':len(prepared),'stored_reference_trajectories':reference_rows,
                  'original_frames':original_frames,'stored_frames_with_mirrors':frames,
                  'prepared_manifest_sha256':prepared_hashes,'reference_manifest_sha256':meta['private_manifest_sha256']})
 complete=not pending and seen==expected.keys()
 report={'status':'FULL_AVAILABLE_REFERENCE_INPUTS_VERIFIED' if complete else 'AVAILABLE_REFERENCE_INPUTS_PENDING',
         'profile_id':profile['profile_id'],'expected_original_sources':len(expected),'verified_original_sources':len(seen),
         'remaining_original_sources':len(expected)-len(seen),'cohorts':cohorts,'pending_cohorts':pending,
         'original_frames':sum(x['original_frames'] for x in cohorts),'all_available_references_verified':complete,
         'prepared_and_reference_file_sha_verified':True,'source_split_and_original_mirror_binding_verified':True,
         'training_selection':'original references only, declared export/manifest order preserved',
         'unavailable_selected_export_sources':len(unavailable_keys),'author_private_clips_unavailable':profile['author_private_clips_unavailable'],
         'paper_data_scope_exact':False,'physical_filter_executed':False,'dataset_accepted':False,
         'formal_filter_training_started':False,'full_generator_training_started':False}
 report['reference_order_fingerprint']=hashlib.sha256(json.dumps([{'sha256':x['sha256'],'frames':x['frames']} for x in originals],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if require_complete:assert complete,'All original sources and their FK exports must exist before capacity/training'
 return originals,report

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
 assert args.run.replace('_','').isalnum() and os.getuid()==1013
 os.sched_setaffinity(0,{0,1,2,3});os.nice(19)
 out=ROOT/'receipts'/('full_library_inputs_'+args.run+'.json');assert not out.exists()
 start=time.monotonic();cap=read(ROOT/'configs/full-library-capacity-spec.json');profile=read(ROOT/'configs/paper-step2.json')['data_scope']
 _,report=inspect(cap,profile)
 report.update(elapsed_seconds=time.monotonic()-start,source_sha256=sha(__file__),read_only=True,gpu_used=False)
 out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
