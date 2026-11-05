"""Read-only native body-tracking evaluation with the entire upstream evaluator."""
import hashlib,json,os,runpy,sys,time,traceback,math
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104';W=N/'evaluation_entry500_v1'
W.mkdir(exist_ok=False);started=time.monotonic();report={'status':'initializing','pid':os.getpid(),'scope':'native29 body tracking evaluation entry; not Pick object interaction'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():
 t=W/'status.tmp';t.write_text(json.dumps(report,indent=2,allow_nan=False));t.replace(W/'status.json')
save()
try:
 spec=json.loads((N/'environment_spec.json').read_text())['spec'];sys.path[:0]=spec['module_paths']
 assert json.loads((N/'benchmark_8912_v1_guard.json').read_text())['status']=='passed'
 source=N/'source/supermimic/evaluate.py';assert sha(source)=='5b8923a8c46dcbf2b19e048729ab0039b9ffe1a45777f83eb355df14b245bb36'
 checkpoint=N/'benchmark_8912_v1/logs/ckpt_000010.pt';before=sha(checkpoint)
 logs=W/'logs';logs.mkdir();(logs/'ckpt_000010.pt').symlink_to(checkpoint)
 import torch
 torch.set_num_threads(2);torch.manual_seed(20261104);torch.cuda.set_per_process_memory_fraction(.20,0)
 command=[str(source),'--run_dir',str(logs),'--motion_dir',spec['input_csv_directory'],'--visualizer','none','--device','cuda:0']
 report.update(status='native_evaluation_running',command=command,checkpoint_sha256=before);save()
 sys.argv=command;g=runpy.run_path(str(source),run_name='__main__')
 result=json.loads((logs/'ckpt_000010_eval.json').read_text())
 assert len(result['per_clip'])==500 and 0<=result['n_failed']<=500
 assert len(result['failed'])==result['n_failed'] and all(math.isfinite(v) for v in result['per_clip'].values())
 assert sha(checkpoint)==before
 report.update(status='native_evaluation_entry_passed',reference_count=500,num_envs=500,checkpoint_unchanged=True,
  optimizer_updates=0,n_failed=result['n_failed'],body_tracking_survival=(500-result['n_failed'])/500,
  normalized_tracking_return=result['norm_return'],per_clip_report=str(logs/'ckpt_000010_eval.json'),
  elapsed_seconds=time.monotonic()-started,formal_policy=False,object_grasp_success_not_measured=True)
 save();print(json.dumps(report),flush=True)
except BaseException as e:
 report.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save();raise
