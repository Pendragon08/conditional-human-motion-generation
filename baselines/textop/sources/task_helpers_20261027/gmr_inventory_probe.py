import pathlib,subprocess,json,os
root=pathlib.Path('/data/code/ouyanghao/oyh')
g=root/'humanoid-combat-motion/third_party/GMR'
envpy=root/'phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
out={'gmr_git':{},'raw_examples':[],'gmr_env_pyvenv':(envpy.parent.parent/'pyvenv.cfg').read_text()}
for args in [['rev-parse','HEAD'],['status','--short'],['config','--get','remote.origin.url']]:out['gmr_git'][' '.join(args)]=subprocess.run(['git','-C',str(g)]+args,capture_output=True,text=True).stdout.strip()
p=root/'phygile-reproduction-20261017/data/amass'
for subgroup in ['ssm_20261021','remaining_20261021','remaining_20261021_v4','remaining_20261021_v5']:
    files=[x for x in (p/subgroup).rglob('*.npz') if not 'SMPLX_lockedhead' in str(x)]
    out['raw_examples'].extend([str(x) for x in files[:5]])
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
code="import sys,torch,numpy,smplx,importlib.util,json; print(json.dumps({'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'gmr':str(importlib.util.find_spec('general_motion_retargeting'))}))"
p=subprocess.run([str(envpy),'-c',code],env=env,capture_output=True,text=True)
out['env_probe']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-2500:]}
(root/'textop-reproduction-20261024/receipts/gmr_inventory.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
