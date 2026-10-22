"""Seeded dispatch witness: actual MuJoCo-Warp step, not just import success."""
import argparse,datetime,importlib.metadata as md,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled')
import torch,numpy as np,mujoco,mujoco_warp as mw,warp as wp
torch.set_num_threads(4);torch.manual_seed(42)
x=torch.randn(8,8,requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
if not torch.isfinite(x.grad).all():raise RuntimeError('PyTorch kernel failed')
wp.init();model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
data=mujoco.MjData(model)
start=time.monotonic()
with wp.ScopedDevice('cpu'):
 wm=mw.put_model(model);wd=mw.put_data(model,data,nworld=1,nconmax=10,njmax=10)
 before=wd.qpos.numpy().copy()
 for _ in range(2):mw.step(wm,wd)
 after=wd.qpos.numpy().copy()
 if not np.isfinite(after).all() or not after[0,2]<before[0,2]:raise RuntimeError('Actual CPU MJWarp integration did not fall under gravity')
result={'status':'MJLAB_CPU_KERNEL_WITNESS_PASSED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'versions':{n:md.version(n) for n in ['torch','torchvision','tensordict','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib']},'device':'cpu','torch_seed':42,'torch_loss':loss.item(),'physics_steps':2,'initial_z':float(before[0,2]),'final_z':float(after[0,2]),'elapsed_physics_seconds':time.monotonic()-start,'scope':'Seeded CPU tensor backward and actual MuJoCo-Warp dispatch; no policy training or tracking quality claim.'}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
