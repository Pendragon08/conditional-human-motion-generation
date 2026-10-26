"""Collect closed real pilot evidence; keep all per-motion traces and SDK logs private."""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
from session import remote

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
a = ap.parse_args()
assert re.fullmatch('[a-z0-9_]+', a.run)
script = r'''
import pathlib,json,hashlib,tarfile,io,os,time,math
r=pathlib.Path(ROOT_LITERAL);run=RUN_LITERAL;os.sched_setaffinity(0,{0,1,2,3})
assert os.getuid()==1013
out=r/'tracker_validation'/run;prefix='quick_pilot_'+run
paths={'receipt':out/'receipt.json','exit':r/'receipts'/(prefix+'_exit.json'),
       'process_exit':r/'receipts'/(prefix+'_process_exit.json'),
       'resources':r/'logs'/(prefix+'_resources.jsonl'),'sdk_log':r/'logs'/(prefix+'.log')}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
data={k:json.loads(paths[k].read_text()) for k in ['receipt','exit','process_exit']}
rec,ex,pr=[data[k] for k in ['receipt','exit','process_exit']]
assert rec['mode']=='quick_pilot' and rec['quality_label']=='QUICK_REPRODUCTION_NOT_PAPER_COMPLETE'
assert ex['mode']=='quick_pilot' and ex['run']==run and pr['child_reaped'] is True
assert ex['evaluation_receipt_sha256']==sha(paths['receipt'])
assert ex['resources_sha256']==sha(paths['resources']) and ex['process_exit_sha256']==sha(paths['process_exit'])
for key in ['child_pid','child_exit_code','child_reaped','timed_out','supervision_passed','supervisor_error','samples_written']:
 assert ex[key]==pr[key]
resources=[json.loads(s) for s in paths['resources'].read_text().splitlines()]
assert len(resources)==pr['samples_written'] and resources[0]['phase']=='before_launch' and resources[-1]['phase']=='after_exit'
spec=(r/'configs/quick-pilot-spec.json').read_bytes()
dest=r/'private_backups/quick_pilot_actual'/run
assert not dest.is_symlink();dest.mkdir(parents=True,exist_ok=True);os.chmod(dest,0o700)
archive=dest/'closed_records.private.tar.gz';proof=dest/'backup.json'
if not archive.exists():
 assert not proof.exists()
 files=list(paths.values())
 for name in ['strict.private.json','strict_trials.private.jsonl','executed_traces.private.json']:
  p=out/name
  if p.exists():files.append(p)
 folder=out/'execution_private'
 if folder.exists():files.extend(p for p in folder.rglob('*') if p.is_file())
 assert all(p.is_file() and not p.is_symlink() for p in files)
 total=sum(p.stat().st_size for p in files)
 assert total<300*1024**2
 assert os.statvfs(dest).f_bavail*os.statvfs(dest).f_frsize>total+2*1024**3
 assert hashlib.sha256(spec).hexdigest()==rec['pilot_spec_sha256'], 'Archive runtime spec before changing flags/witnesses'
 members={str(p.relative_to(r)):sha(p) for p in files}
 with archive.open('xb') as target:
  with tarfile.open(fileobj=target,mode='w:gz') as tar:
   for p in files:tar.add(str(p),arcname=str(p.relative_to(r)),recursive=False)
   info=tarfile.TarInfo('runtime-spec.json');info.size=len(spec);tar.addfile(info,io.BytesIO(spec))
 os.chmod(archive,0o600)
 with tarfile.open(archive,'r:gz') as tar:
  assert len(tar.getmembers())==len(files)+1
  for name,h in members.items():assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==h
  assert hashlib.sha256(tar.extractfile('runtime-spec.json').read()).hexdigest()==rec['pilot_spec_sha256']
 saved={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'path':str(archive),
        'member_count':len(files)+1,'source_receipt_sha256':sha(paths['receipt']),
        'runtime_spec_sha256':rec['pilot_spec_sha256'],'members_verified':True,'created_unix':time.time()}
 proof.write_text(json.dumps(saved,indent=2));os.chmod(proof,0o600)
else:
 saved=json.loads(proof.read_text());assert sha(archive)==saved['archive_sha256']
 assert saved['source_receipt_sha256']==sha(paths['receipt'])
metrics={}
index=out/'executed_traces.private.json'
if index.exists():
 records=json.loads(index.read_text())
 assert hashlib.sha256(index.read_bytes()).hexdigest()==rec['private_executed_trace_index_sha256']
 for group in ['all','nominal_duration_completed','failed']:
  rows=[x for x in records if group=='all' or x['physical_failure']==(group=='failed')]
  steps=sum(x['samples']-1 for x in rows)
  if not steps:continue
  metrics[group]={'trials':len(rows),'trial_control_steps':steps,
    'aligned_14_body_position_error_mean_m':sum(x['metrics']['aligned_14_body_position_error_mean_m']*(x['samples']-1) for x in rows)/steps,
    'aligned_14_body_position_error_max_m':max(x['metrics']['aligned_14_body_position_error_max_m'] for x in rows),
    'joint_rmse_rad':math.sqrt(sum(x['metrics']['joint_rmse_rad']**2*(x['samples']-1) for x in rows)/steps)}
print(json.dumps({'data':data,'private_backup':saved,'resource_samples':len(resources),'own_tracking_metrics':metrics,
                  'receipt_sha256':sha(paths['receipt']),'exit_sha256':sha(paths['exit']),
                  'process_exit_sha256':sha(paths['process_exit']),'resources_sha256':sha(paths['resources'])}))
'''.replace('ROOT_LITERAL', repr(ROOT)).replace('RUN_LITERAL', repr(a.run))
j = json.loads(remote(script, timeout=100))
private = D/'datasets/quick_pilot_execution_private_20261026'/a.run
private.mkdir(parents=True, exist_ok=True)
meta = j['private_backup']
target = private/'closed_records.private.tar.gz'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()

if not target.exists():
    assert shutil.disk_usage(private).free > meta['archive_bytes'] + 1024**3
    partial = target.with_suffix(target.suffix+'.partial')
    assert not partial.exists(), 'Preserve interrupted download; inspect before resume'
    result = subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',
        'C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+meta['path'],str(partial)],
        capture_output=True,text=True,encoding='utf-8',timeout=180)
    (private/'transfer.json').write_text(json.dumps({'exit_code':result.returncode,'stderr':result.stderr}),encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert digest(partial) == meta['archive_sha256'] and partial.stat().st_size == meta['archive_bytes']
    partial.replace(target)
assert digest(target) == meta['archive_sha256'] and target.stat().st_size == meta['archive_bytes']
(private/'backup.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
public = B/'results/quick_pilot_actual_20261026'/a.run
public.mkdir(parents=True, exist_ok=True)
for key in ['receipt','exit','process_exit']:
    (public/(key+'.json')).write_text(json.dumps(j['data'][key],indent=2),encoding='utf-8')
rec,ex=j['data']['receipt'],j['data']['exit']
summary = {k:rec.get(k) for k in ['status','stage','quality_label','finished_trials','complete_trials',
    'failed_trials','actual_vectorized_control_calls','actual_scene_physics_steps','num_envs',
    'output_writers_closed','native_close_returned','teardown_fallback_used','final_reference_endpoint_comparison_performed',
    'peak_torch_allocated_bytes','policy_checkpoint_sha256','policy_checkpoint_completed_iterations',
    'policy_checkpoint_optimizer_updates','unattempted_original_sources']}
summary.update(run=a.run,child_exit_code=ex['child_exit_code'],supervision_passed=ex['supervision_passed'],
    receipt_sha256=j['receipt_sha256'],exit_sha256=j['exit_sha256'],resource_samples=j['resource_samples'],
    private_archive_sha256=meta['archive_sha256'],private_archive_bytes=meta['archive_bytes'],
    private_two_location_sha_verified=True,full_library_filtered=False,generator_trained=False,
    actual_execution_video_created=False,private_arrays_publicly_published=False)
summary['own_tracking_metrics'] = j['own_tracking_metrics']
summary['metrics_definition'] = ('Control-step-weighted errors against consumed current targets before autoreset; '
    '29-joint RMSE and 14-body positions; not independently checked final endpoint or paper benchmark scores')
(public/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
local = D/'results/quick_pilot_actual_20261026'/a.run
local.mkdir(parents=True, exist_ok=True)
for p in public.iterdir():
    if p.is_file():
        shutil.copy2(p,local/p.name)
        assert digest(p) == digest(local/p.name)
print(json.dumps(summary))
