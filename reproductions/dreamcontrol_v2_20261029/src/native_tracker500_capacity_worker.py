"""Invoke the entire pinned upstream train.py; no replacement optimization loop."""
import argparse,ctypes,hashlib,json,os,runpy,sys,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';W=D/'native_universal_20261104'
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['capacity','benchmark'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(exist_ok=False);began=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(v):
 t=a.output/'status.tmp';t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(a.output/'status.json')
report={'status':'initializing','pid':os.getpid(),'mode':a.mode,'formal_training':False};save(report)
try:
 spec_doc=json.loads((W/'environment_spec.json').read_text());spec=spec_doc['spec'];sys.path[:0]=spec['module_paths']
 assert sha(spec['full_native_train'])==spec['full_native_train_sha256']=='b94569605bd3ff96a49d4903bb791bb54bc4695a03f1914c8447c0ec7dd1e6bd'
 import torch
 torch.set_num_threads(2);torch.manual_seed(20261104);torch.cuda.set_per_process_memory_fraction(.20,0)
 x=torch.randn((8,8),device='cuda:0');v=x@x;error=float((v.cpu()-x.cpu()@x.cpu()).abs().max());assert torch.isfinite(v).all() and error<1e-4
 report.update(seed=20261104,shape=[8,8],device=torch.cuda.get_device_name(0),torch=torch.__version__,spec_hash=spec_doc['spec_hash'],seeded_kernel_error=error)
 if False:
  report.update(status='kernel_passed',elapsed_seconds=time.monotonic()-began);save(report);print('NATIVE_WITNESS '+json.dumps(report),flush=True)
 else:
  command=[spec['full_native_train'],'--visualizer','none','--device','cuda:0','--seed','20261104',
   '--motion_dir',spec['input_csv_directory'],'--log_dir',str(a.output/'logs'),'--amp','bf16',
   '--num_envs','8912','--iterations','0' if a.mode=='capacity' else '10','--buffer_steps','128','--warmup_steps','64','--wm_only_iters','1',
   '--wm_batches','10','--wm_batch','2048','--pi_batch','2048','--horizon','32','--log_every','1','--ckpt_every','10']
  (a.output/'command.json').open('x').write(json.dumps(command,indent=2));report.update(status='full_native_trainer_running',command=command);save(report)
  sys.argv=command;g=runpy.run_path(spec['full_native_train'],run_name='__main__');torch.cuda.synchronize()
  if a.mode=='capacity':
   assert g['sim_steps']==570368 and g['motion'].clip_len.numel()==500 and g['buffer'].size==64
   assert all(bool(torch.isfinite(v[:64]).all()) for v in g['buffer'].data.values())
   report.update(status='native_capacity_passed',num_envs=8912,reference_clips=500,real_environment_transitions=570368,
    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    elapsed_seconds=time.monotonic()-began,scope='actual8912 native TrackingSim+64real steps+500 library; zero optimizer updates')
   save(report);print(json.dumps(report),flush=True);sys.exit(0)
  def steps(opt):
   values=[int(s['step']) for s in opt.state.values() if 'step' in s];assert values and min(values)==max(values);return max(values)
  updates={k:steps(g[k]) for k in ['opt_wm','opt_pi','opt_v']}
  assert updates=={'opt_wm':100,'opt_pi':9,'opt_v':180},'native iteration count is not an optimization-step proof'
  assert g['sim_steps']==793168 and g['motion'].clip_len.numel()==500
  checkpoint=a.output/'logs/last.pt';stored=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert stored['iteration']==10 and stored['sim_steps']==793168
  for key in ['wm','pi','critic']:
   actual=g[key].state_dict();assert all(torch.equal(v.cpu(),stored[key][k]) for k,v in actual.items())
  import torch._inductor.metrics as im
  generated=im.generated_kernel_count;assert generated>0,'no compiled-kernel execution evidence'
  report.update(status='native_benchmark_passed',num_envs=8912,reference_clips=500,native_motion_frames=g['motion'].num_frames,
   native_learned_body_actions=29,reference_conditioned_actor=True,optimizer_updates=updates,real_environment_transitions=793168,
   full_native_train_sha256=sha(spec['full_native_train']),generated_inductor_kernels=generated,
   checkpoint={'path':str(checkpoint),'sha256':sha(checkpoint),'bytes':checkpoint.stat().st_size,'model_roundtrip':True,
    'scope':'upstream native model/optimizer checkpoint; upstream format does not save replay/RNG'},
   torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   elapsed_seconds=time.monotonic()-began,scope='real full upstream compiled/batched optimizer test; not formal Pick or grasp success')
  save(report);print('NATIVE_TRAINER_WITNESS '+json.dumps(report),flush=True)
except SystemExit:
 raise
except BaseException as error:
 report.update(status='failed_preserved',error=repr(error),traceback=traceback.format_exc());save(report);raise
