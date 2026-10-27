"""CPU provenance of two real actual-pass corpus loader witnesses.

No Torch/pickle deserialization, labels or arrays are exposed by this module.
Loader/CLIP/finite results must come from the source-bound real GPU jobs.
"""
import hashlib,json,os,pathlib
from filtered15000_inputs import load_selection

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def owned(path,root):
 p=pathlib.Path(path);root=pathlib.Path(root)
 assert p.resolve()==p and root.resolve()==root and (p==root or root in p.parents)
 here=p
 while True:
  assert here.stat().st_uid==os.getuid()==1013
  if here==root:break
  here=here.parent
 return p

def validate_corpus(spec,verify_artifacts=True):
 root=pathlib.Path(spec['root']);cp=root/'configs/filtered15000-corpus-spec.json'
 owned(cp,root);assert sha(cp)==spec['corpus_spec_sha256']
 cs=read(cp);assert cs['schema']=='textop.filtered15000_corpus.v1' and cs['build_authorized']
 for n,h in cs['bound_sources'].items():assert sha(root/n)==h,n
 assert sha(root/'scripts/build_filtered15000_corpus.py')==cs['builder_sha256']
 assert sha(root/'scripts/supervise_filtered15000_corpus.py')==cs['supervisor_sha256']
 strict,selected=load_selection(cs);witnesses=spec['corpus_witnesses']
 assert len(witnesses)==len({x['run'] for x in witnesses})==2
 assert [x['mode'] for x in witnesses]==['build','verify']
 assert spec['fresh_document_summary_sha256'] and len(spec['fresh_document_summary_sha256'])==64
 view=pathlib.Path(cs['candidate_view']);view_parent=pathlib.Path('/dev/shm/ouyanghao_textop_20261025');owned(view,view_parent)
 vp=owned(view/'view_receipt.json',view_parent);meta=read(vp);view_sha=sha(vp)
 assert meta['status']=='DIRECT_STRICT15000_CORPUS_BUILT_LOADER_PENDING'
 assert meta['physical_filter_passed'] and not meta['dataset_accepted'] and not meta['paper_two_stage_route_reproduced']
 assert meta['input_library_sources']==8278 and meta['source_sequences']==len(selected)==strict['complete_passes']
 assert meta['strict_audit_receipt_sha256']==cs['strict_audit_receipt_sha256']
 assert meta['source_sequences_in_view']==len(selected)-meta['quarantined_train_source_families']>0
 assert meta['fit_split']=='train' and meta['normalization_scope']=='ACTUAL_STRICT15000_PASSES_TRAIN_ONLY_REFIT'
 assert meta['normalization_boundary_handling_verified'] and meta['exact_cross_split_pose_overlap_after_quarantine']==0
 assert meta['train_stat_feature_frames']>0 and meta['unique_clip_texts']>0 and len(meta['clip_weight_sha256'])==64
 assert meta['validation_variants_removed']==0
 for name,item in meta['artifacts'].items():
  assert name==pathlib.Path(name).name and name not in ['.','..']
  p=owned(view/name,view_parent);assert p.is_file() and p.stat().st_size==item['bytes']
  if verify_artifacts:assert sha(p)==item['sha256'],name
 required={'train.pkl','val.pkl','meanstd.pkl','train_text_embed.pkl','val_text_embed.pkl','statistics.yaml','action_statistics.json','cross_split_quarantine.private.json'}
 assert required<=set(meta['artifacts'])
 checks=[]
 for w in witnesses:
  out=root/'filtered15000_20261027/corpus_runs'/w['run'];rp=owned(out/'receipt.json',root)
  prefix='filtered15000_corpus_'+w['run'];ep=owned(root/'receipts'/(prefix+'_exit.json'),root)
  assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256']
  rec,e=read(rp),read(ep);pp=owned(root/'receipts'/(prefix+'_process_exit.json'),root);pr=read(pp)
  outer=read(owned(root/'receipts'/(prefix+'_outer_exit.json'),root));assert outer['actual_returncode']==0
  assert rec['status']=='FILTERED15000_CORPUS_LOADER_WITNESS_PASSED' and rec['output_writers_closed']
  assert rec['source_sha256']==cs['builder_sha256'] and rec['spec_sha256']==spec['corpus_spec_sha256']
  assert rec['view_receipt_sha256']==view_sha and rec['sampler_source_sha256']==cs['bound_sources']['scripts/paper69_data_runtime.py']
  assert rec['strict_audit_receipt_sha256']==cs['strict_audit_receipt_sha256'] and rec['input_library_sources']==8278
  assert rec['actual_strict_input_corpus_loader_verified'] and not rec['paper_filtered_dataset_accepted']
  assert rec['normalization_cross_source_boundaries_excluded'] and rec['train_only_statistics']
  # The build job computes source/pose disjointness. Verify mode reloads the
  # same SHA-bound artifacts and does not repeat that original preprocessing.
  if w['mode']=='build':assert rec['source_splits_disjoint'] and rec['exact_pose_splits_disjoint']
  assert rec['source_sequences']==len(selected) and rec['clip_unique_texts']==meta['unique_clip_texts']
  assert rec['source_sequences']*2-rec['quarantined_train_variants']==rec['motion_variants']==meta['motion_variants']
  assert rec['optimizer_updates']==0 and rec['cpu_affinity']==[0,1,2,3]
  assert len(rec['loader_checks'])==3
  assert [(x['split'],x['weighted']) for x in rec['loader_checks']]==[('train',False),('val',False),('train',True)]
  for c in rec['loader_checks']:
   assert c['valid_sequences']>0 and c['raw_sequences']>=c['valid_sequences']
   assert c['batch_shape']==[4,512,10,69] and c['text_shape']==[512,512]
   assert c['missing_text_embeddings']==0 and c['normalization_finite'] and c['history_overlap_error']<1e-5
  for x in [e,pr]:
   assert x['child_reaped'] and x['child_exit_code']==0 and x['supervision_passed'] and not x['timed_out']
   assert x['supervisor_error'] is None and not x.get('cleanup_error') and x['sample_errors']==[]
  assert e['mode']==w['mode'] and e['run']==w['run'] and e['spec_sha256']==spec['corpus_spec_sha256']
  assert e['receipt_sha256']==sha(rp) and e['source_sha256']==cs['supervisor_sha256'] and e['process_exit_sha256']==sha(pp)
  resources=owned(root/'logs'/(prefix+'_resources.jsonl'),root);assert sha(resources)==e['resource_sha256']
  samples=[json.loads(x) for x in resources.read_text().splitlines()]
  assert len(samples)==e['samples_written']==pr['samples_written'] and samples[-1]['phase']=='after_exit'
  assert all(x['data_free_bytes']>=2*1024**3 and x['available_host_ram_bytes']>=16*1024**3 for x in samples)
  checks.append(dict(run=w['run'],mode=w['mode'],receipt_sha256=sha(rp),exit_sha256=sha(ep),actual_exit_code=0,
   source_sequences=rec['source_sequences'],loader_checks=rec['loader_checks']))
 assert meta['released_action_statistics_sha256']==cs['released_action_statistics_sha256']
 return dict(corpus_spec=cs,strict_audit=strict,view=view,meta=meta,view_receipt_sha256=view_sha,witnesses=checks)
