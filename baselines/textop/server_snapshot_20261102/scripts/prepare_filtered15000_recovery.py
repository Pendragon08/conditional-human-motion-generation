"""Bounded CPU preparation of small private rebuild material after real loaders."""
import argparse,json,os,pathlib,shutil,tarfile,time,traceback,subprocess
from filtered15000_corpus_acceptance import validate_corpus,sha,read,owned
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);a=p.parse_args()
spec=read(a.spec);assert spec['schema']=='textop.filtered15000_data_recovery.v1' and spec['prepare_authorized']
assert os.getuid()==1013 and set(os.sched_getaffinity(0))=={0,1,2,3}
assert sha(__file__)==spec['executor_sha256']
out=pathlib.Path(a.output);assert out.parent==ROOT/'filtered15000_20261027/data_runs';owned(out.parent,ROOT);out.mkdir(mode=0o700,exist_ok=False)
started=time.monotonic();result=dict(status='PREPARING_FILTERED_CORPUS_RECOVERY',optimizer_updates=0,new_model_loads=0,new_gpu_contexts=0,new_sdk_runs=0,
 dataset_accepted=False,generator_started=False,output_writers_closed=False)
def save():
 result['elapsed_seconds']=time.monotonic()-started;t=out/'receipt.partial';t.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');t.replace(out/'receipt.json')
save()
try:
 for name,h in spec['bound_sources'].items():assert sha(ROOT/name)==h,name
 checked=validate_corpus(spec,verify_artifacts=True);view=checked['view'];meta=checked['meta'];cs=checked['corpus_spec']
 result.update(status='COPYING_PRIVATE_SMALL_RECOVERY',corpus_receipt_sha256=checked['view_receipt_sha256'],strict_audit_receipt_sha256=cs['strict_audit_receipt_sha256'],
  corpus_spec_sha256=spec['corpus_spec_sha256'],source_sequences=meta['source_sequences'],source_sequences_in_view=meta['source_sequences_in_view'],
  motion_variants=meta['motion_variants'],input_library_sources=8278,witnesses=checked['witnesses'],source_sha256=sha(__file__),spec_sha256=sha(a.spec));save()
 folder=out/'corpus_recovery';folder.mkdir(mode=0o700)
 small=[name for name in meta['artifacts'] if name not in ['train.pkl','val.pkl']]+['view_receipt.json']
 assert sum((view/name).stat().st_size for name in small)<spec['maximum_recovery_bytes']
 for name in small:shutil.copyfile(view/name,folder/name)
 audit=checked['strict_audit']
 for name in ['all_decisions.private.json','accepted_sources.private.json']:
  item=audit['private_artifacts'][name];path=owned(pathlib.Path(item['path']),ROOT)
  assert sha(path)==item['sha256'];shutil.copyfile(path,folder/name)
 rebuild=dict(schema='textop.filtered15000_private_rebuild.v1',status='DERIVED_RAM_CORPUS_REBUILD_RECIPE_NOT_RAW_DATA_COPY',
  original_reference_and_prepared_arrays_preserved_separately=True,
  corpus_spec_sha256=spec['corpus_spec_sha256'],strict_audit_receipt_sha256=cs['strict_audit_receipt_sha256'],
  source_selected=meta['source_sequences'],source_in_view=meta['source_sequences_in_view'],
  omitted_derived_RAM_artifacts={n:meta['artifacts'][n] for n in ['train.pkl','val.pkl']},
  prepared_manifest_hashes=meta['prepared_manifest_hashes'],
  rebuild='Restore verified original private inputs, actual strict selected source index and pinned warm source; rebuild the derived joblib cache. Recheck generated artifacts before any resume.',
  scientific_acceptance_unchanged=True,paper_two_stage_route_reproduced=False)
 (folder/'rebuild.private.json').write_text(json.dumps(rebuild,indent=2),encoding='utf-8')
 files={q.name:dict(bytes=q.stat().st_size,sha256=sha(q)) for q in folder.iterdir() if q.is_file()}
 assert sum(x['bytes'] for x in files.values())<spec['maximum_recovery_bytes']
 archive=out/'corpus_recovery.private.tar.gz';assert not archive.exists()
 with tarfile.open(str(archive),'w:gz',compresslevel=1) as tar:
  for name in sorted(files):tar.add(str(folder/name),arcname='corpus_recovery/'+name,recursive=False)
 os.chmod(str(archive),0o600);assert archive.stat().st_size<spec['maximum_recovery_bytes']
 # This is a candidate only. No HOME directory or checkpoint is created here.
 home=pathlib.Path.home();owned(home,home);assert home==pathlib.Path('/home/ouyanghao')
 fs=subprocess.check_output(['findmnt','-T',str(home),'-n','-o','FSTYPE'],text=True,timeout=10).strip()
 assert fs in ['ext4','xfs','btrfs','zfs']
 policy=read(ROOT/'configs/filtered15000-vae-spec.json')['checkpoint_policy']
 required=policy['reserve_bytes']+policy['log_allowance_bytes']+5*policy['checkpoint_upper_bound_bytes']
 result.update(status='FILTERED15000_CORPUS_RECOVERY_PREPARED',private_archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),
  private_recovery_file_count=len(files),small_private_recovery_files=files,
  checkpoint_candidate=dict(home=str(home),proposed_allowed_root=str(home/'textop-filtered15000-checkpoints-20261027'),
   proposed_checkpoint_dir=str(home/'textop-filtered15000-checkpoints-20261027/vae/checkpoints'),filesystem=fs,
   free_bytes=shutil.disk_usage(home).free,required_bytes=required,accepted=False),
  corpus_loader_and_small_recovery_verified=True,private_D_backup_verified=False,
  formal_gradient_witnesses_accepted=False,formal_training_ready=False,output_writers_closed=True)
 save();print(json.dumps(result),flush=True)
except BaseException as error:
 result.update(status='FAILED_FILTERED15000_CORPUS_RECOVERY',error=repr(error),traceback=traceback.format_exc());save();raise
