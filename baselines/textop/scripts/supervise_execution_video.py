"""Bound only a new CPU/Mesa video worker; never acquire/interrupt the live SDK."""
import argparse, hashlib, json, os, pathlib, re
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['renderer_test','actual'],required=True);p.add_argument('--run',required=True)
p.add_argument('--pilot-run');p.add_argument('--trial',type=int,default=0);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
specpath=ROOT/'configs/execution-video-spec.json';spec=json.loads(specpath.read_text())
if a.mode=='actual':assert spec['actual_rendering_enabled'] is True,'Actual rendering disabled before worker dispatch'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(ROOT/'scripts/render_quick_execution_video.py')==spec['renderer_sha256']
assert sha(__file__)==spec['supervisor_sha256']
qp=json.loads((ROOT/'configs/quick-reproduction-plan-20261026.json').read_text())
for name,value in qp['frozen_runtime_files'].items():assert sha(ROOT/name)==value
for name,value in qp['control_sources'].items():assert sha(ROOT/name)==value
prefix='execution_video_'+a.run
receipt=ROOT/'receipts'/(prefix+'_exit.json');process=ROOT/'receipts'/(prefix+'_process_exit.json')
samples=ROOT/'logs'/(prefix+'_resources.jsonl')
out=ROOT/('video_validation' if a.mode=='renderer_test' else 'recorded_videos_private')/a.run
assert not out.exists() and not receipt.exists()
command=['/usr/bin/env']+[key+'='+value for key,value in spec['environment'].items()]+[
 spec['python'],str(ROOT/'scripts/render_quick_execution_video.py'),'--spec',str(specpath),'--mode',a.mode,'--run',a.run]
if a.mode=='actual':command+=['--pilot-run',a.pilot_run,'--trial',str(a.trial)]
proof=run_supervised(command,ROOT/'logs'/(prefix+'.log'),samples,process,
 lambda pid:sample_resources(ROOT,pid,2*1024**3,spec['minimum_disk_free_bytes']),
 spec['test_timeout_seconds'] if a.mode=='renderer_test' else spec['actual_timeout_seconds'],15)
path=out/'receipt.json';r=json.loads(path.read_text()) if path.exists() else {}
unchanged=all(sha(ROOT/name)==value for name,value in qp['frozen_runtime_files'].items())
controls=all(sha(ROOT/name)==value for name,value in qp['control_sources'].items())
proof.update(mode=a.mode,run=a.run,numerical_status=r.get('status'),evaluation_receipt_sha256=sha(path) if path.exists() else None,
             process_exit_sha256=sha(process),resources_sha256=sha(samples),original16_unchanged=unchanged,quick_controls_unchanged=controls,
             training_signals_sent=0,sdk_runs=0,new_physics_steps=0)
with receipt.open('x') as stream:json.dump(proof,stream,indent=2)
print(json.dumps(proof))
expected='SOFTWARE_RENDERER_SYNTHETIC_WITNESS_PASSED' if a.mode=='renderer_test' else 'RECORDED_ISAACLAB_POSE_VIDEO_CREATED'
assert proof['supervision_passed'] and r.get('status')==expected and r.get('output_writers_closed') is True and unchanged and controls
