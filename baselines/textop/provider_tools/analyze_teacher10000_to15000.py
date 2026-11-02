"""Compare existing SHA-bound saved counters; no checkpoint or GPU loads.

New-interval raw counts are diagnostic only and never replace the EMA gate.
"""
import hashlib
import json
import pathlib
import subprocess
from session import SSH

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
OUT = B / 'results/filter_recovery_checkpoints_20261027/round_15000/saved_counter_comparison.json'
assert not OUT.exists(), 'Keep the existing analysis; do not repeat unchanged inputs'
code = r'''
import hashlib,json,math,os,pathlib
os.sched_setaffinity(0,{0,1,2,3})
assert os.getuid()==1013
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
paths=[root/'teacher10000_full_filter_20261027/relaxed_scope_v1/relaxed_source_decisions.private.json',root/'checkpoint_backups/round_15000/filter_recovery_author_v1/relaxed_source_decisions.private.json']
digests=['0dbe24ed5055e56ce0b5284bad797cdd5440c855780c4a0f391839c047f401b8','aa992b9c490dfc5894de66651f7a1710e5a9c3a95277f726a7daf9981ad3d0c5']
checkpoints=['6f949e4edc50d9a763a3634d716162cb5ddecb079bd1142d3720f907f5efab45','66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88']
docs=[]
for p,h,ck in zip(paths,digests,checkpoints):
 assert p.resolve()==p and p.stat().st_uid==1013
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==h
 doc=json.loads(raw);assert doc['checkpoint_sha256']==ck
 assert doc['reference_fingerprint']=='0b786ba08bf8f33aa92cb4e91ed72cd5b56159c4e251cedf88f429269141c1f7'
 assert len(doc['motions'])==len({x['canonical_source'] for x in doc['motions']})==8278
 docs.append(doc)
def distribution(values):
 s=sorted(values)
 def q(p):
  k=(len(s)-1)*p;i=int(k);j=min(i+1,len(s)-1);return s[i]+(s[j]-s[i])*(k-i)
 return dict(count=len(s),minimum=s[0],median=q(.5),p95=q(.95),maximum=s[-1]) if s else None
ema_before=[];ema_after=[];raw_interval=[];failed=success=improved=worsened=unchanged=0
identity_keys=['global_index','canonical_source','reference_sha256','reference_frames','split']
for before,after in zip(docs[0]['motions'],docs[1]['motions']):
 assert all(before[k]==after[k] for k in identity_keys)
 for item in (before,after):
  assert all(type(item[k]) is int and item[k]>=0 for k in ['raw_failed','raw_success'])
  rate=item['ema_failed']/(item['ema_failed']+item['ema_success'])
  assert math.isfinite(rate) and rate==item['relaxed_failure_probability']
 f=after['raw_failed']-before['raw_failed'];s=after['raw_success']-before['raw_success']
 assert f>=0 and s>=0
 failed+=f;success+=s
 if f+s:raw_interval.append(f/(f+s))
 x=before['relaxed_failure_probability'];y=after['relaxed_failure_probability']
 ema_before.append(x);ema_after.append(y)
 improved+=y<x;worsened+=y>x;unchanged+=y==x
assert failed+success==21785568-15887307
assert improved+worsened+unchanged==8278
print(json.dumps(dict(status='EXISTING_SAVED_COUNTER_INTERVAL_COMPARISON_COMPLETE',source_count=8278,
 input_decision_sha256=digests,checkpoint_sha256=checkpoints,source_identity_pairs_verified=8278,
 ema_failure_before=distribution(ema_before),ema_failure_after=distribution(ema_after),
 ema_lower_sources=improved,ema_higher_sources=worsened,ema_unchanged_sources=unchanged,
 interval_attempted_sources=len(raw_interval),interval_attempts=failed+success,interval_failed=failed,
 interval_success=success,interval_pooled_raw_failure_rate=failed/(failed+success),
 interval_source_raw_failure_distribution=distribution(raw_interval),
 interval_raw_rate_at_most_5_percent_DIAGNOSTIC_ONLY=sum(x<=.05 for x in raw_interval),
 actual_ema_gate_candidates_at_15000=sum(x<=.05 for x in ema_after),
 strict_physics_accepted=False,dataset_accepted=False,new_model_loads=0,new_sdk_runs=0,
 new_physics_steps=0,new_optimizer_updates=0,
 limitation='Interval raw rates are accumulated during learning, not a frozen-policy evaluation; they do not replace the EMA criterion or prove a unique cause.')))
'''
result = subprocess.run(SSH + ['python3 -'], input=code, text=True, encoding='utf-8', capture_output=True, timeout=90)
assert result.returncode == 0, result.stderr
summary = json.loads(result.stdout)
summary.update(actual_client_exit_code=result.returncode, source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), remote_script_sha256=hashlib.sha256(code.encode('utf-8')).hexdigest())
OUT.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
mirror = D / 'results/filter_recovery_checkpoints_20261027/round_15000/saved_counter_comparison.json'
mirror.parent.mkdir(parents=True, exist_ok=True)
assert not mirror.exists()
mirror.write_bytes(OUT.read_bytes())
assert hashlib.sha256(mirror.read_bytes()).digest() == hashlib.sha256(OUT.read_bytes()).digest()
print(json.dumps(summary))
