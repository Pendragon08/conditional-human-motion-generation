import sys,pathlib,json,subprocess,tarfile,hashlib
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,SSH,REPO
BASE=REPO/'baselines/textop'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
for p in [BASE/'scripts',BASE/'configs',BASE/'reports',BASE/'results',BASE/'sources',LOCAL]:p.mkdir(parents=True,exist_ok=True)
manifest=json.loads((LOCAL/'hf_asset_manifest.json').read_text(encoding='utf-8'))
spec={'base':{'python':'3.8.20','torch':'1.13.1+cu117','numpy':'1.24.4','read_only_base':'/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training','isolation':'venv --system-site-packages'},'pip_phases':[['omegaconf==2.3.0','hydra-core==1.3.2','loguru==0.7.3','loralib==0.1.2','toolz==1.0.0','einops==0.8.1','tqdm==4.66.4','easydict==1.13','wandb==0.16.6','tensorboard<2.15','ftfy','regex'],['clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'],['-e upstream/TextOpRobotMDAR --no-deps','-e upstream/TextOpRobotMDAR/deps/isaac_utils --no-deps']], 'env':{'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','WANDB_MODE':'disabled','PYTHONUNBUFFERED':'1'},'gpu_fraction':0.65,'smoke':{'seed':0,'gpu_tests':[{'cmd':'env/bin/python scripts/kernel_witness.py','expect':'TEXTOP_KERNEL_WITNESS'}]},'deferred_dependencies':{'torch_tensorrt':'Only imported by TensorRT frequency evaluation, not MVAE/DAR training. Will build separate deployment environment.'}}
(BASE/'configs/env-spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
(BASE/'sources/hf_asset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
sh=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
(BASE/'SERVER_NOTES.md').write_text(f'# TextOp environment ledger\n\n### env: textop@{sh}\nhow: isolated venv on nlp03, shared base read only.\ntier: CPUs 4, GPU1 A6000, CUDA allocation cap 65%.\nstatus: spec written BEFORE environment build; validation pending.\nsource: TeleHuman/TextOp ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22\nassets: Yochish/TextOp-Data bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9\n',encoding='utf-8')
script=f'''
import pathlib,os,json
r=pathlib.Path('{ROOT}');r.mkdir(parents=True,exist_ok=True)
for x in ['scripts','configs','receipts','logs','assets','runs']: (r/x).mkdir(exist_ok=True)
pathlib.Path('/home/ouyanghao/textop-assets-20261024').mkdir(exist_ok=True)
'''
print(remote(script))
tunnel=subprocess.Popen(SSH[:-1]+['-o','ExitOnForwardFailure=yes','-N','-R','17897:127.0.0.1:7897',SSH[-1]],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=open(TASK/'tunnel.log','ab'),creationflags=subprocess.CREATE_NO_WINDOW)
(TASK/'tunnel.json').write_text(json.dumps({'pid':tunnel.pid,'remote_port':17897,'local_proxy':'127.0.0.1:7897'}))
archive=TASK/'upstream.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in (TASK/'upstream').rglob('*'):
        if p.is_file() and '.git' not in p.relative_to(TASK/'upstream').parts:tar.add(p,arcname='upstream/'+p.relative_to(TASK/'upstream').as_posix())
scp=['scp','-i','C:/Users/Pende/.ssh/id_ed25519']
subprocess.run(scp+[str(archive),'ouyanghao@10.147.17.32:'+ROOT+'/upstream.tar.gz'],check=True)
print(remote(f"import tarfile; t=tarfile.open('{ROOT}/upstream.tar.gz'); t.extractall('{ROOT}'); print('SOURCE_STAGED')"))
for src,dst in [(BASE/'configs/env-spec.json','configs/env-spec.json'),(BASE/'sources/hf_asset_manifest.json','configs/asset_manifest.json')]:subprocess.run(scp+[str(src),'ouyanghao@10.147.17.32:'+ROOT+'/'+dst],check=True)
print('SPEC_HASH',sh,'TUNNEL_PID',tunnel.pid)
