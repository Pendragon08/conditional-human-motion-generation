"""Private REFERENCE+BABEL corpus from the real quick pilot, never execution poses.

Build and verify modes use the warm generator environment. No model optimizer,
training checkpoint, whole-library acceptance or shared dependency is created.
"""
import argparse, hashlib, importlib, importlib.metadata, json, os, pathlib, random, shutil, sys, time, traceback

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--mode',choices=['build','verify'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
spec=json.loads(pathlib.Path(a.spec).read_text());base=json.loads((ROOT/'configs/paper69-env-spec.json').read_text())
assert spec['schema']=='textop.quick69_corpus.v1' and spec['build_authorized'] is True
assert os.getuid()==1013 and os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and set(os.sched_getaffinity(0))=={0,1,2,3}
assert sys.version.split()[0]==base['python_version']
assert {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
assert {k:importlib.metadata.version(k) for k in spec['additional_warm_package_versions']}==spec['additional_warm_package_versions']
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(__file__)==spec['builder_sha256']
assert hashlib.sha256(json.dumps(base,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==spec['warm_environment_hash']
for n,h in spec['bound_sources'].items():assert sha(ROOT/n)==h
assert a.run.isidentifier() and a.run.isascii()
private=ROOT/'quick_source_views/quick_sample64_nominal_v1/source_decisions.private.json'
assert sha(private)==spec['private_source_view_sha256']
assert sha(private.parent/'summary.json')==spec['source_view_summary_sha256']
decisions=json.loads(private.read_text());kept=[x for x in decisions if x['include_in_candidate_view']]
assert len(kept)==42 and {s:sum(x['reference']['split']==s for x in kept) for s in ['train','val']}=={'train':31,'val':11}
assert all(x['reference']['mirrored'] is False and x['nominal_tracking_completed'] is True and not x['training_ready'] for x in kept)
assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>spec['reserve_bytes']+spec['corpus_budget_bytes']
from select_quick_corpus_sources import select_actual
verified,_=select_actual(ROOT,'quick_pilot_sample64_v1',json.loads((ROOT/'configs/quick-corpus-selection-spec.json').read_text()))
assert verified==decisions
parent=ROOT/'quick_generator_20261026';parent.mkdir(mode=0o700,exist_ok=True)
assert parent.resolve()==parent and parent.stat().st_uid==1013 and parent.stat().st_dev==ROOT.stat().st_dev
out=pathlib.Path(spec['corpus_dir']);assert out.parent==parent and out.resolve()==out
logs=parent/'corpus_validation';logs.mkdir(mode=0o700,exist_ok=True)
run=logs/a.run;run.mkdir(mode=0o700,exist_ok=False);started=time.monotonic()
r={'status':'STARTING_QUICK69_CORPUS','mode':a.mode,'quality_label':'QUICK_REPRODUCTION_NOT_PAPER_COMPLETE','optimizer_updates':0,
   'full_generator_training_started':False,'whole_library_filtered':False,'quick_training_ready':False,
   'source_view_sha256':spec['private_source_view_sha256'],'spec_sha256':sha(a.spec),'source_sha256':sha(__file__)}
def save():
 r['elapsed_seconds']=time.monotonic()-started
 tmp=run/'receipt.json.partial';tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(run/'receipt.json')
save()
try:
 import numpy as np, torch, joblib, yaml
 from omegaconf import OmegaConf
 from paper69_runtime import install_runtime,encode69
 from paper69_data_runtime import install_data_sampling
 random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(spec['torch_memory_fraction'])
 assert torch.cuda.mem_get_info()[0]>20*1024**3
 install_runtime(base['robot_xml']);install_data_sampling()
 from robotmdar.dataloader.data import SkeletonPrimitiveDataset
 from robotmdar.dtype.motion import MotionKeys
 if a.mode=='build':
  out.mkdir(mode=0o700,exist_ok=False)
  index_path=pathlib.Path(spec['babel_source_index']);assert sha(index_path)==spec['babel_source_index_sha256']
  from reconcile_raw_sources import source_key
  index={source_key(x['source']):x for x in json.loads(index_path.read_text())};assert len(index)==8291
  data={'train':[],'val':[]};source_sets={'train':set(),'val':set()};pose_sets={'train':set(),'val':set()}
  for item in kept:
   ref=item['reference'];source=ref['canonical_source'];split=ref['split'];assert index[source]['split']==split
   assert sha(ref['prepared_file'])==ref['prepared_sha256'] and sha(ref['file'])==ref['sha256']
   assert pathlib.Path(ref['prepared_file']).stat().st_uid==1013
   with np.load(ref['prepared_file'],allow_pickle=False) as z:
    q=np.array(z['qpos_29'],dtype=np.float32);contact=np.array(z['contact'],dtype=np.float32)
   assert q.shape==(ref['frames'],36) and contact.shape==(len(q),2) and len(q)>=35
   assert np.isfinite(q).all() and np.isfinite(contact).all()
   assert np.allclose(np.linalg.norm(q[:,3:7],axis=-1),1,atol=1e-3)
   annotations=[[x['start_s'],x['end_s'],x['text'],x['action_categories']] for x in ref['annotations']]
   assert all(0<=x[0]<x[1]<=len(q)/50+1e-8 and isinstance(x[2],str) for x in annotations)
   pose=hashlib.sha256(q.tobytes()).hexdigest();source_sets[split].add(source);pose_sets[split].add(pose)
   motion={'root_trans_offset':q[:,:3].copy(),'root_rot':q[:,[4,5,6,3]].copy(),'dof':q[:,7:].copy(),'contact_mask':contact.copy(),'fps':50,'motion_len':len(q)}
   data[split].append({'feat_p':source,'babel_sid':index[source]['babel_sid'],'length':len(q),'mirrored':False,
    'pose_fingerprint':pose,'physical_filter_passed':False,'quick_nominal_tracking_completed':True,'motion':motion,'frame_ann':annotations})
  assert not source_sets['train']&source_sets['val']
  overlap=pose_sets['train']&pose_sets['val'];quarantined={x['feat_p'] for x in data['train'] if x['pose_fingerprint'] in overlap}
  excluded=[{'source':x['feat_p'],'reason':'exact_float32_pose_overlap_with_validation'} for x in data['train'] if x['feat_p'] in quarantined]
  (out/'pose_quarantine.private.json').write_text(json.dumps(excluded,indent=2))
  data['train']=[x for x in data['train'] if x['feat_p'] not in quarantined]
  assert not {x['pose_fingerprint'] for x in data['train']}&pose_sets['val'] and len(data['train'])>=2 and data['val']
  sums=np.zeros(69,np.float64);squares=sums.copy();count=0
  for item in data['train']:
   values={k:torch.from_numpy(item['motion'][k]) for k in MotionKeys};f,_=encode69(values)
   f=f.numpy().astype(np.float64);assert f.shape==(item['length']-1,69) and np.isfinite(f).all()
   sums+=f.sum(0);squares+=np.square(f).sum(0);count+=len(f)
  assert count==sum(x['length']-1 for x in data['train'])
  mean=sums/count;std=np.sqrt(np.maximum(0,squares/count-mean**2));floor=np.flatnonzero(std<1e-6).tolist();std=np.maximum(std,1e-6)
  torch.save((torch.from_numpy(mean.astype(np.float32)),torch.from_numpy(std.astype(np.float32))),out/'meanstd.pkl')
  (out/'statistics.yaml').write_text(yaml.safe_dump({'fps':50,'nfeats':69,'dof':29,'quick_subset_only':True}))
  table=pathlib.Path(spec['released_action_statistics']);assert sha(table)==spec['released_action_statistics_sha256'];shutil.copy2(table,out/'action_statistics.json')
  for split in data:joblib.dump(data[split],out/(split+'.pkl'),compress=0)
  import clip
  impl=importlib.import_module('clip.clip');expected=impl._MODELS['ViT-B/32'].split('/')[-2]
  weight=next(pathlib.Path(x) for x in base['clip_weight_candidates'] if pathlib.Path(x).exists());assert sha(weight)==expected
  model,_=clip.load(str(weight),device='cuda',jit=False);clip.model.convert_weights(model);model.eval()
  for p in model.parameters():p.requires_grad=False
  from robotmdar.model.clip import encode_text
  texts=sorted({x[2] for items in data.values() for item in items for x in item['frame_ann'] if x[2]});cache={'':torch.zeros(512)}
  assert texts
  with torch.no_grad():
   for start in range(0,len(texts),64):
    names=texts[start:start+64];values=encode_text(model,names).cpu();assert values.shape==(len(names),512) and torch.isfinite(values).all();cache.update(zip(names,list(values)))
  for split,items in data.items():
   needed={x[2] for item in items for x in item['frame_ann']}|{''};torch.save({k:cache[k] for k in sorted(needed)},out/(split+'_text_embed.pkl'))
  meta={'status':'QUICK69_REFERENCE_CORPUS_BUILT_LOADER_PENDING','quality_label':r['quality_label'],'source_view_sha256':spec['private_source_view_sha256'],
   'input_nominal_completed_sources':42,'retained_sources':sum(len(x) for x in data.values()),'split_sources':{s:len(v) for s,v in data.items()},
   'annotation_pairs':sum(len(x['frame_ann']) for v in data.values() for x in v),'unique_nonempty_clip_texts':len(texts),'clip_weight_sha256':expected,
   'train_stat_feature_frames':count,'fit_split':'train','statistic_source_boundaries':'each source encoded independently; no cross-source transitions',
   'std_floor':1e-6,'std_floored_feature_indices':floor,'exact_pose_cross_split_overlap_before':len(overlap),'quarantined_train_sources':len(quarantined),
   'source_splits_disjoint':True,'exact_float32_pose_splits_disjoint':True,'mirrors_included':False,'unassessed_mirrors_excluded':True,
   'action_statistics_scope':'Pinned released author table; NOT frequencies fitted to this quick TRAIN subset',
   'action_statistics_sha256':spec['released_action_statistics_sha256'],'whole_library_filtered':False,'quick_training_ready':False,
   'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir() if p.is_file()}}
  (out/'corpus_receipt.json').write_text(json.dumps(meta,indent=2));del data,model,cache
  torch.cuda.empty_cache()
 meta=json.loads((out/'corpus_receipt.json').read_text());assert meta['source_view_sha256']==spec['private_source_view_sha256'] and meta['whole_library_filtered'] is False
 for name,item in meta['artifacts'].items():assert sha(out/name)==item['sha256']
 robot=pathlib.Path(base['robot_xml']);robot_cfg=OmegaConf.create({'asset':{'assetRoot':str(robot.parent),'assetFileName':robot.name}})
 checks=[]
 for split,weighted in [('train',False),('val',False),('train',True)]:
  ds=SkeletonPrimitiveDataset(robot_cfg,512,69,2,8,4,str(out),str(out/'action_statistics.json'),weighted_sample=weighted,split=split)
  batch=ds._generate_batch_optimized(generator=torch.Generator().manual_seed(spec['seed']))
  assert len(batch)==4 and all(x.shape==(512,10,69) and y.shape==(512,512) and torch.isfinite(x).all() and torch.isfinite(y).all() for x,y in batch)
  continuity=max(float((batch[i][0][:,-2:]-batch[i+1][0][:,:2]).abs().max()) for i in range(3));assert continuity<1e-5
  missing=sum(x[2] not in ds.text_embeddings_dict for item in ds.raw_data for x in item['frame_ann'])
  assert missing==0
  checks.append({'split':split,'weighted':weighted,'source_count':len(ds.raw_data),'valid_sequences':len(ds),'primitives':4,
    'feature_batch_shape':[512,10,69],'text_batch_shape':[512,512],'history_overlap_error':continuity,'missing_text_embeddings':missing,
    'max_abs_normalized_batch_feature':max(float(x.abs().max()) for x,y in batch)})
  del ds,batch
 assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<spec['corpus_budget_bytes']
 r.update(status='QUICK69_CORPUS_LOADER_WITNESS_PASSED',corpus_receipt_sha256=sha(out/'corpus_receipt.json'),
    corpus_summary={k:v for k,v in meta.items() if k!='artifacts'},loader_checks=checks,output_writers_closed=True,
    peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),remaining_data_bytes=shutil.disk_usage(ROOT).free,
    limitation='Quick42-reference subset and real cache/loader only; gradients/resume/trainer acceptance still pending; no model optimized')
 save();print(json.dumps(r))
except BaseException as e:
 r.update(status='FAILED_QUICK69_CORPUS',error=repr(e),traceback=traceback.format_exc());save();raise
