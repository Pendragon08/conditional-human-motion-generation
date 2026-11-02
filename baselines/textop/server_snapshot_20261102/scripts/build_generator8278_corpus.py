"""Build/read the user-authorized PRIVATE full pre-filter generator corpus.

No optimizer or model training here. This corpus is valid only for the explicitly
authorized pre-filter experiment; it is never paper-filtered acceptance.
"""
import argparse,hashlib,importlib,importlib.metadata,json,os,pathlib,random,shutil,sys,time,traceback
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--base-spec',required=True)
p.add_argument('--output',required=True);p.add_argument('--build',action='store_true');a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'dataset_accepted':False,'physical_filter_executed':False,
 'full_generator_training_started':False,'candidate_view_only':True};started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def save():
 receipt['elapsed_seconds']=time.monotonic()-started;t=out/'receipt.json.partial';t.write_text(json.dumps(receipt,indent=2));t.replace(out/'receipt.json')
try:
 spec=json.loads(pathlib.Path(a.spec).read_text());assert spec['schema']=='textop.generator8278_corpus.v1' and spec['build_authorized'];assert sha(__file__)==spec['builder_sha256'];base=json.loads(pathlib.Path(a.base_spec).read_text())
 assert os.getuid()==1013 and len(os.sched_getaffinity(0))<=4 and os.environ['CUDA_VISIBLE_DEVICES']=='1'
 assert sys.version.split()[0]==base['python_version']
 assert {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
 assert hashlib.sha256(json.dumps(base,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==spec['warm_base_canonical_hash']
 assert sha(spec['preprocessing_receipt'])==spec['preprocessing_receipt_sha256']
 proof=json.loads(pathlib.Path(spec['preprocessing_receipt']).read_text());assert proof['verified_sources']==8278 and proof['preprocessing_complete'] is True
 for n,h in spec['bound_sources'].items():assert sha(pathlib.Path(spec['root'])/n)==h
 import numpy as np,torch,joblib,yaml
 from omegaconf import OmegaConf
 from paper69_runtime import install_runtime,encode69
 from paper69_data_runtime import install_data_sampling
 torch.set_num_threads(4);torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
 torch.cuda.set_per_process_memory_fraction(.65);assert torch.cuda.mem_get_info()[0]>20*1024**3
 install_runtime(base['robot_xml']);adapter=install_data_sampling()
 from robotmdar.dataloader.data import SkeletonPrimitiveDataset
 view=pathlib.Path(spec['candidate_view']);assert view.parent.stat().st_uid==1013 and view.parent.resolve()==view.parent;receipt.update(source_sha256=sha(__file__),spec_sha256=sha(a.spec),adapter=adapter,
   sampler_source_sha256=sha(pathlib.Path(__file__).with_name('paper69_data_runtime.py')),cpu_affinity=sorted(os.sched_getaffinity(0)))
 save()
 if a.build:
  assert not view.exists() and shutil.disk_usage(view.parent).free>spec['minimum_shm_free_gib']*1024**3
  view.mkdir(mode=0o700);index_path=pathlib.Path(spec['verified_source_index']);assert sha(index_path)==spec['verified_source_index_sha256']
  from reconcile_raw_sources import source_key
  index=json.loads(index_path.read_text());index={source_key(x['source']):x for x in index};assert len(index)==8291
  rows=[];manifest_hashes={}
  for name in spec['prepared_manifests']:
   path=pathlib.Path(name);assert sha(path)==spec['prepared_manifest_sha256'][name];proof=json.loads((path.parent/'progress.json').read_text())
   assert proof['status']=='PREPROCESSING_FINISHED_FILTER_PENDING' and proof['failed_sequences']==0
   batch=[json.loads(x) for x in path.read_text().splitlines()];assert len(batch)==proof['completed_sequences']
   rows.extend(batch);manifest_hashes[name]=sha(path)
  assert len(rows)==spec['expected_sources']==8278 and len({x['canonical_source'] for x in rows})==len(rows)
  assert {s:sum(x['split']==s for x in rows) for s in ['train','val']}=={'train':6203,'val':2075}
  data={'train':[],'val':[]};sources={'train':set(),'val':set()};pose_hashes={'train':set(),'val':set()}
  sums=np.zeros(69,np.float64);squares=np.zeros(69,np.float64);count=0;annotation_count=0
  for position,row in enumerate(rows):
   split=row['split'];source=row['canonical_source'];assert split in data and index[source]['split']==split
   assert row['dof']==29 and row['features']==69 and sha(row['output_file'])==row['output_sha256']
   sources[split].add(source)
   with np.load(row['output_file'],allow_pickle=False) as motion:
    for mirrored,key,ckey,akey in [(False,'qpos_29','contact','annotations'),(True,'mirrored_qpos_29','mirrored_contact','mirrored_annotations')]:
     q=np.array(motion[key],dtype=np.float32);contact=np.array(motion[ckey],dtype=np.float32)
     assert q.shape==(row['frames50'],36) and np.isfinite(q).all()
     assert contact.shape==(len(q),2) and np.isfinite(contact).all()
     assert np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-3)
     values={'root_trans_offset':q[:,:3].copy(),'root_rot':q[:,[4,5,6,3]].copy(),'dof':q[:,7:].copy(),'contact_mask':contact.copy()}
     pose_fingerprint=hashlib.sha256(q.tobytes()).hexdigest();pose_hashes[split].add(pose_fingerprint)
     ann=[[x['start_s'],x['end_s'],x['text'],x['action_categories']] for x in row[akey]]
     assert all(0<=x[0]<x[1]<=len(q)/50+1e-8 for x in ann)
     item={'feat_p':source,'babel_sid':index[source]['babel_sid'],'length':len(q),'mirrored':mirrored,'pose_fingerprint':pose_fingerprint,
           'physical_filter_passed':False,'motion':{**values,'fps':50,'motion_len':len(q)},'frame_ann':ann}
     data[split].append(item);annotation_count+=len(ann)
   if (position+1)%100==0:receipt.update(status='BUILDING_PRIVATE_CANDIDATE_VIEW',processed_sources=position+1);save()
  assert not sources['train']&sources['val'],'Cross-split source identity leakage'
  overlap=pose_hashes['train']&pose_hashes['val']
  quarantined={x['feat_p'] for x in data['train'] if x['pose_fingerprint'] in overlap}
  # Keep validation unchanged and quarantine the entire corresponding TRAIN
  # source family, including its mirrored variant. Original assets are kept.
  excluded=[{'canonical_source':x['feat_p'],'mirrored':x['mirrored'],'reason':'exact_float32_pose_overlap_with_validation',
             'pose_fingerprint':x['pose_fingerprint']} for x in data['train'] if x['feat_p'] in quarantined]
  (view/'cross_split_quarantine.private.json').write_text(json.dumps(excluded,indent=2))
  data['train']=[x for x in data['train'] if x['feat_p'] not in quarantined]
  annotation_count=sum(len(x['frame_ann']) for items in data.values() for x in items)
  assert not {x['pose_fingerprint'] for x in data['train']}&pose_hashes['val']
  receipt.update(status='FITTING_TRAIN_ONLY_CANDIDATE_STATISTICS',processed_sources=len(rows),
   exact_cross_split_pose_groups=len(overlap),quarantined_train_source_families=len(quarantined),
   quarantined_train_variants=len(excluded),validation_variants_removed=0);save()
  eligible=[x for x in data['train'] if x['length']>=35]
  assert len(eligible)>=2
  from robotmdar.dtype.motion import MotionKeys
  joined={k:torch.cat([torch.from_numpy(x['motion'][k]) for x in eligible],dim=0) for k in MotionKeys}
  features,_=encode69(joined);valid=torch.ones(len(features),dtype=torch.bool)
  boundaries=np.cumsum([x['length'] for x in eligible])[:-1]-1;valid[torch.as_tensor(boundaries)]=False
  # Pairwise transforms are independent of earlier frames. Exclude every
  # cross-source transition and accumulate only valid training rows in float64.
  small=[{k:torch.from_numpy(x['motion'][k][:35]) for k in MotionKeys} for x in eligible[:2]]
  combined,_=encode69({k:torch.cat([x[k] for x in small],0) for k in MotionKeys})
  separate=torch.cat([encode69(x)[0] for x in small],0);mask=torch.ones(len(combined),dtype=torch.bool);mask[34]=False
  assert torch.allclose(combined[mask],separate,atol=1e-6,rtol=1e-6)
  for start in range(0,len(features),65536):
   f=features[start:start+65536][valid[start:start+65536]].numpy().astype(np.float64);assert np.isfinite(f).all()
   sums+=f.sum(0);squares+=np.square(f).sum(0);count+=len(f)
  assert count==sum(x['length']-1 for x in eligible)
  del features,valid,joined
  assert count>0
  mean=sums/count;variance=np.maximum(0,squares/count-mean*mean);std=np.sqrt(variance);floored=np.flatnonzero(std<1e-6).tolist();std=np.maximum(std,1e-6)
  torch.save((torch.from_numpy(mean.astype(np.float32)),torch.from_numpy(std.astype(np.float32))),view/'meanstd.pkl')
  (view/'statistics.yaml').write_text(yaml.safe_dump({'fps':50,'nfeats':69,'dof':29,'candidate_only':True}))
  assert pathlib.Path(spec['released_action_statistics']).is_file()
  shutil.copyfile(spec['released_action_statistics'],view/'action_statistics.json')
  for split in ['train','val']:joblib.dump(data[split],view/(split+'.pkl'),compress=0)
  receipt.update(status='ENCODING_PRIVATE_CLIP_CACHE',source_sequences=len(rows),motion_variants=sum(len(x) for x in data.values()),
   annotation_pairs=annotation_count,split_motion_variants={k:len(v) for k,v in data.items()},train_stat_feature_frames=count,
   train_only_normalization=True,std_floor=1e-6,std_floored_feature_indices=floored,source_splits_disjoint=True,
   exact_pose_splits_disjoint=True,prepared_manifest_hashes=manifest_hashes);save()
  import clip
  clip_impl=importlib.import_module('clip.clip');expected=clip_impl._MODELS['ViT-B/32'].split('/')[-2]
  weight=next(pathlib.Path(x) for x in base['clip_weight_candidates'] if pathlib.Path(x).exists());assert sha(weight)==expected
  model,_=clip.load(str(weight),device='cuda',jit=False);clip.model.convert_weights(model);model.eval()
  for parameter in model.parameters():parameter.requires_grad=False
  unique=sorted({ann[2] for items in data.values() for x in items for ann in x['frame_ann']});cache={'':torch.zeros(512)}
  from robotmdar.model.clip import encode_text
  with torch.no_grad():
   for start in range(0,len(unique),64):
    texts=unique[start:start+64];embeddings=encode_text(model,texts).cpu()
    assert torch.isfinite(embeddings).all() and embeddings.shape==(len(texts),512)
    cache.update(zip(texts,list(embeddings)))
  for split,items in data.items():
   texts={ann[2] for x in items for ann in x['frame_ann']}|{''}
   torch.save({text:cache[text] for text in sorted(texts)},view/(split+'_text_embed.pkl'))
  assert sha(spec['released_action_statistics'])==spec['released_action_statistics_sha256']
  meta={'status':'PREFILTER_8278_CORPUS_BUILT_NOT_PHYSICALLY_FILTERED','physical_filter_passed':False,'dataset_accepted':False,
   'source_sequences':len(rows),'motion_variants':sum(len(v) for v in data.values()),'prepared_manifest_hashes':manifest_hashes,
   'quarantined_train_source_families':len(quarantined),'quarantined_train_variants':len(excluded),'validation_variants_removed':0,
   'source_sequences_in_view':len(rows)-len(quarantined),'normalization_boundary_handling_verified':True,
   'cross_split_pose_groups_before_quarantine':len(overlap),'exact_cross_split_pose_overlap_after_quarantine':0,
   'fit_split':'train','train_stat_feature_frames':count,'std_floor':1e-6,'std_floored_feature_indices':floored,
   'unique_clip_texts':len(unique),'clip_weight_sha256':expected,'released_action_statistics_sha256':sha(spec['released_action_statistics']),
   'normalization_scope':'USER_AUTHORIZED_PREFILTER_TRAINING_ONLY_REFIT_AFTER_ACTUAL_FILTERING',
   'artifacts':{x.name:{'bytes':x.stat().st_size,'sha256':sha(x)} for x in view.iterdir() if x.is_file()}}
  (view/'view_receipt.json').write_text(json.dumps(meta,indent=2));del data,cache,model
 else:meta=json.loads((view/'view_receipt.json').read_text())
 assert meta['status']=='PREFILTER_8278_CORPUS_BUILT_NOT_PHYSICALLY_FILTERED' and not meta['dataset_accepted'] and meta['source_sequences']==spec['expected_sources']
 for name,entry in meta['artifacts'].items():assert sha(view/name)==entry['sha256']
 robot=pathlib.Path(base['robot_xml']);robot_cfg=OmegaConf.create({'asset':{'assetRoot':str(robot.parent),'assetFileName':robot.name}})
 results=[]
 for split,weighted in [('train',False),('val',False),('train',True)]:
  dataset=SkeletonPrimitiveDataset(robot_cfg,spec['batch_size'],69,2,8,4,str(view),str(view/'action_statistics.json'),weighted_sample=weighted,split=split)
  batch=dataset._generate_batch_optimized(generator=torch.Generator().manual_seed(spec['seed']))
  assert len(batch)==4
  for features,text in batch:assert features.shape==(spec['batch_size'],10,69) and text.shape==(spec['batch_size'],512) and torch.isfinite(features).all() and torch.isfinite(text).all()
  continuity=max(float((batch[i][0][:,-2:]-batch[i+1][0][:,:2]).abs().max()) for i in range(3));assert continuity<1e-5
  missing=sum(ann[2] not in dataset.text_embeddings_dict for x in dataset.raw_data for ann in x['frame_ann']);assert missing==0
  results.append({'split':split,'weighted':weighted,'raw_sequences':len(dataset.raw_data),'valid_sequences':len(dataset),
                  'batch_shape':[4,spec['batch_size'],10,69],'text_shape':[spec['batch_size'],512],
                  'missing_text_embeddings':missing,'history_overlap_error':continuity,'normalization_finite':bool(torch.isfinite(dataset.std).all() and (dataset.std>0).all())})
  del dataset,batch
 # Exact35frame and invalid-source weighted mapping regressions, no synthetic corpus publication.
 mock=object.__new__(SkeletonPrimitiveDataset);mock.valid_indices=[1];mock.batch_size=2;mock.weighted_sample=True;mock.frame_weight=False
 mock.seq_weights=np.array([1.,1.]);mock.raw_data=[{'length':1},{'length':35}];mock.segment_len=35
 mock._generate_motion_primitives=lambda source,start:(source['length'],start)
 assert mock._sample_motion_batch(torch.Generator().manual_seed(0))==[(35,0),(35,0)]
 receipt.update(status='GENERATOR8278_CORPUS_LOADER_WITNESS_PASSED',source_sequences=meta['source_sequences'],motion_variants=meta['motion_variants'],
   view_receipt_sha256=sha(view/'view_receipt.json'),loader_checks=results,exact35_frame_regression_passed=True,
   quarantined_train_source_families=meta['quarantined_train_source_families'],quarantined_train_variants=meta['quarantined_train_variants'],
   validation_variants_removed=0,normalization_cross_source_boundaries_excluded=True,
   invalid_weighted_index_regression_passed=True,clip_unique_texts=meta['unique_clip_texts'],train_only_statistics=True,
   peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),output_writers_closed=True,
   limitation='User-authorized8278-source pre-filter corpus; no optimizer/physical filtering/semantic acceptance. Paper-filtered dataset acceptance remains false')
 receipt['prefilter_corpus_loader_verified']=True;receipt['paper_filtered_dataset_accepted']=False
 receipt['spec_scope']='8278-source pre-filter training explicitly selected by user2026-10-27'
 save();print(receipt['status'],flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
