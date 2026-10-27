import pathlib,json,subprocess,shutil,os
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');ram=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/local1653_v1')
p=ram/'prepared1653_v1/progress.json'
out={'progress':json.loads(p.read_text()) if p.exists() else {},'space':{x:shutil.disk_usage(x).free for x in [str(r),'/home/ouyanghao','/dev/shm']},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader'],text=True),'runtime_files':{}}
for name in ['/data/code/ouyanghao/oyh/phygile-unitree-20261022/scripts/run_isaac.sh','/data/code/ouyanghao/oyh/phygile-unitree-20261022/.aris/compute/ssh.md','/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-engineai-official/pyvenv.cfg','/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/pyvenv.cfg']:
 p=pathlib.Path(name)
 if p.exists():out['runtime_files'][name]=p.read_text()[:8000]
manifest=ram/'prepared1653_v1/prepared_manifest.private.jsonl'
if manifest.exists():
 x=json.loads(manifest.open().readline());out['prepared_record_fields']=list(x);out['prepared_record_nontext']={k:v for k,v in x.items() if k not in ['annotations','mirrored_annotations','rejected_annotations','canonical_source','source','source_file']}
print(json.dumps(out))
