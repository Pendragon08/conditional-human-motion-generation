"""Read actual saved relaxed-training statistics for every original source.

No new physics, fitting, policy update, data deletion or strict pass is implied.
The checkpoint has 10000 completed teacher rounds, a disclosed paper difference.
"""
import argparse,hashlib,json,math,os,pathlib
import torch
from teacher10000_pilot_protocol import load_teacher,authorize_teacher,sha
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert a.run.isascii() and a.run.isidentifier() and os.getuid()==1013
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
assert set(os.sched_getaffinity(0))<={0,1,2,3};torch.set_num_threads(4)
out=ROOT/'teacher10000_full_filter_20261027'/a.run;assert not out.exists()
spec=json.loads((ROOT/'configs/teacher10000-pilot-spec.json').read_text())
teacher=load_teacher(ROOT,spec);ck=authorize_teacher(spec,teacher)
checkpoint=pathlib.Path(ck['path']);assert sha(checkpoint)==ck['sha256']
saved=torch.load(checkpoint,map_location='cpu');info=saved['infos']
assert info['completed_learning_iterations']==10000 and info['actual_optimizer_updates']==200000
assert info['diagnostic_only'] is False
plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text());rows=[]
for name in plan['input_reference_exports']:
 folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
 meta=json.loads((folder/'receipt.json').read_text())
 assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED' and sha(folder/'manifest.private.jsonl')==meta['private_manifest_sha256']
 rows.extend(r for r in map(json.loads,(folder/'manifest.private.jsonl').read_text().splitlines()) if not r['mirrored'])
assert len(rows)==len({r['canonical_source'] for r in rows})==8278
fp=hashlib.sha256(json.dumps([{'sha256':r['sha256'],'frames':r['frames']} for r in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert fp==info['reference_fingerprint']==spec['expected_reference_fingerprint']
stats=info['recording'];keys=['raw_failed','raw_success','failed_motion_count','success_motion_count']
assert all(len(stats[k])==8278 and bool(torch.isfinite(stats[k]).all()) and bool((stats[k]>=0).all()) for k in keys)
decisions=[];counts={'eligible_for_strict':0,'rejected_relaxed_rate':0,'unattempted':0,'invalid_denominator':0};by_split={};by_dataset={}
for i,row in enumerate(rows):
 failed=int(stats['raw_failed'][i]);success=int(stats['raw_success'][i]);ef=float(stats['failed_motion_count'][i]);es=float(stats['success_motion_count'][i])
 rate=ef/(ef+es) if ef+es>0 else None
 decision='unattempted' if failed+success==0 else 'invalid_denominator' if rate is None else 'rejected_relaxed_rate' if rate>.05 else 'eligible_for_strict'
 counts[decision]+=1
 split=by_split.setdefault(row['split'],{k:0 for k in counts});split[decision]+=1
 dataset=row.get('dataset',row['canonical_source'].split('/')[0]);group=by_dataset.setdefault(dataset,{k:0 for k in counts});group[decision]+=1
 decisions.append(dict(global_index=i,canonical_source=row['canonical_source'],reference_sha256=row['sha256'],reference_frames=row['frames'],split=row['split'],
  motion_id=hashlib.sha256(row['canonical_source'].encode()).hexdigest(),raw_failed=failed,raw_success=success,ema_failed=ef,ema_success=es,relaxed_failure_probability=rate,decision=decision))
out.mkdir(parents=True,exist_ok=False)
private=out/'relaxed_source_decisions.private.json';private.write_text(json.dumps({'schema':'textop.teacher10000_relaxed_scope.v1','checkpoint_sha256':ck['sha256'],'reference_fingerprint':fp,'motions':decisions},indent=2,allow_nan=False),encoding='utf-8')
summary=dict(status='TEACHER10000_ALL_SOURCE_RELAXED_STATISTICS_EXPORTED_STRICT_PENDING',schema='textop.teacher10000_relaxed_scope.v1',
 checkpoint_sha256=ck['sha256'],teacher_completed_rounds=10000,teacher_optimizer_updates=200000,paper_reference_rounds=40000,source_count=8278,
 reference_fingerprint=fp,counts=counts,by_split=by_split,by_dataset=by_dataset,relaxed_threshold=.05,
 eligible_original_frames=sum(x['reference_frames'] for x in decisions if x['decision']=='eligible_for_strict'),
 raw_attempts_total=sum(x['raw_failed']+x['raw_success'] for x in decisions),
 private_decisions_sha256=sha(private),private_decisions_bytes=private.stat().st_size,source_sha256=sha(__file__),
 new_physics_steps=0,new_sdk_runs=0,new_optimizer_updates=0,strict_evaluation_completed=False,dataset_accepted=False,
 original_assets_deleted=0,limitation='Actual relaxed training EMA at10000, not paper40000; passing first gate is not strict physics or semantic acceptance')
assert not torch.cuda.is_initialized();(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
