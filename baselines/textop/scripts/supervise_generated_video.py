"""CPU-only bounded input-contract witness with the unchanged Mesa visual core."""
import argparse,json,os,pathlib,re
from generated_pipeline_protocol import sha
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run)
sp=ROOT/'configs/generated-execution-video-spec.json';s=json.loads(sp.read_text());assert s['generated_video_enabled'] and sha(__file__)==s['supervisor_sha256'] and sha(ROOT/'scripts/render_generated_execution.py')==s['wrapper_sha256']
assert sha(ROOT/'scripts/render_quick_execution_video.py')==s['renderer_sha256']
qp=json.loads((ROOT/'configs/quick-reproduction-plan-20261026.json').read_text());assert all(sha(ROOT/n)==h for n,h in dict(qp['frozen_runtime_files'],**qp['control_sources']).items())
prefix='generated_video_'+a.run;ep=ROOT/'receipts'/(prefix+'_exit.json');assert not ep.exists() and not(ROOT/'recorded_videos_private'/a.run).exists()
samples=ROOT/'logs'/(prefix+'_resources.jsonl');process=ROOT/'receipts'/(prefix+'_process_exit.json')
cmd=['/usr/bin/env']+[k+'='+v for k,v in s['environment'].items()]+[s['python'],str(ROOT/'scripts/render_generated_execution.py'),'--run',a.run]
proof=run_supervised(cmd,ROOT/'logs'/(prefix+'.log'),samples,process,lambda pid:sample_resources(ROOT,pid,2*1024**3,s['minimum_disk_free_bytes']),900,15)
rp=ROOT/'recorded_videos_private'/a.run/'receipt.json';r=json.loads(rp.read_text()) if rp.exists() else {}
proof.update(evaluation_receipt_sha256=sha(rp) if rp.exists() else None,process_exit_sha256=sha(process),resources_sha256=sha(samples),new_sdk_runs=0,new_physics_steps=0,new_optimizer_updates=0)
with ep.open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof));assert proof['supervision_passed'] and r.get('status')=='RECORDED_ISAACLAB_POSE_VIDEO_CREATED' and r.get('output_writers_closed')
