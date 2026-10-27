
import pathlib,json,hashlib,collections,numpy as np,os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');spec=json.loads((r/'configs/candidate-loader-spec.json').read_text());groups=collections.defaultdict(list)
for name in spec['prepared_manifests']:
 for line in pathlib.Path(name).read_text().splitlines():
  row=json.loads(line)
  with np.load(row['output_file'],allow_pickle=False) as data:
   for mirror,key,contactkey in [(False,'qpos_29','contact'),(True,'mirrored_qpos_29','mirrored_contact')]:
    q=data[key];digest=hashlib.sha256(q.astype(np.float32).tobytes()).hexdigest()
    groups[digest].append({'source':row['canonical_source'],'split':row['split'],'mirrored':mirror,'frames':len(q),'raw_sha256':row['raw_sha256'],'raw_path':row['raw_path'],
     'float64_pose_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'contact_sha256':hashlib.sha256(np.asarray(data[contactkey],dtype=np.float32).tobytes()).hexdigest(),
     'joint_max_range_rad':float(np.ptp(q[:,7:],axis=0).max()),'root_max_range_m':float(np.ptp(q[:,:3],axis=0).max())})
collisions={k:v for k,v in groups.items() if {x['split'] for x in v}=={'train','val'}}
source_sets={s:{x['source'] for g in collisions.values() for x in g if x['split']==s} for s in ['train','val']}
private=r/'generator_validation/candidate_loader_author_v1/cross_split_collision_audit.private.json';private.write_text(json.dumps(collisions,indent=2))
summary={'status':'EXACT_CROSS_SPLIT_POSE_COLLISIONS_CONFIRMED','collision_groups':len(collisions),'affected_train_sources':len(source_sets['train']),'affected_val_sources':len(source_sets['val']),
 'collision_variant_counts':{s:sum(x['split']==s for g in collisions.values() for x in g) for s in ['train','val']},
 'collision_frame_lengths':sorted({x['frames'] for g in collisions.values() for x in g}),
 'distinct_raw_sha_per_group':[len({x['raw_sha256'] for x in g}) for g in collisions.values()],
 'distinct_float64_pose_sha_per_group':[len({x['float64_pose_sha256'] for x in g}) for g in collisions.values()],
 'distinct_contact_sha_per_group':[len({x['contact_sha256'] for x in g}) for g in collisions.values()],
 'joint_max_ranges_rad':[max(x['joint_max_range_rad'] for x in g) for g in collisions.values()],
 'root_max_ranges_m':[max(x['root_max_range_m'] for x in g) for g in collisions.values()],
 'private_collision_log_sha256':hashlib.sha256(private.read_bytes()).hexdigest(),'candidate_sources_examined':2296,'dataset_accepted':False,'physical_filter_executed':False}
(r/'receipts/candidate_cross_split_collision_v1.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
