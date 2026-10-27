import pathlib,sys,subprocess,json,time,hashlib,os
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO,GH
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');DEST=LOCAL/'checkpoints';DEST.mkdir(exist_ok=True)
env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
print('Waiting for real MVAE step1000 checkpoint',flush=True)
deadline=time.time()+1800;data=None
while time.time()<deadline:
    data=json.loads(remote(r'''
import pathlib,json,time,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
files=list((r/'runs').rglob('ckpt_1000.pth'))
if files and time.time()-files[0].stat().st_mtime>15:
    p=files[0];h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'step':1000}))
else:print('null')
''',timeout=60))
    if data:break
    time.sleep(20)
if not data:raise TimeoutError('No step1000 checkpoint within deadline')
local=DEST/'TextOp_MVAE_Public_step1000.pth'
if not local.exists() or sha(local)!=data['sha256']:
    subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+data['path'],str(local)],check=True)
assert local.stat().st_size==data['bytes'] and sha(local)==data['sha256']
data['local_path']=str(local);print('LOCAL_CHECKPOINT_VERIFIED',data['bytes'],flush=True)
meta=DEST/'TextOp_MVAE_Public_step1000.sha256.json';meta.write_text(json.dumps(data,indent=2))
repo='Pendragon08/conditional-human-motion-generation';tag='textop-mvae-initial-20261024'
def gh(*args,check=True):return subprocess.run([GH,*args],env=env,check=check,capture_output=True,text=True,encoding='utf-8')
existing=gh('release','view',tag,'--repo',repo,'--json','url',check=False)
if existing.returncode:
    notes=DEST/'release-notes.md';notes.write_text('TextOp公开数据MVAE第1000步初始训练检查点，包含模型与Adam优化器状态。用于续训和启动验证，尚未完成20万步训练，不代表论文最终质量或闭环结果。\n\n官方代码固定ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22；数据固定bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9。原始AMASS/SMPL及配对数据不包含在本发布中。\n',encoding='utf-8')
    print(gh('release','create',tag,'--repo',repo,'--target','codex/textop-reproduction-20261024','--title','TextOp public MVAE: initial step1000 checkpoint','--notes-file',str(notes),'--prerelease').stdout,flush=True)
release=json.loads(gh('api',f'repos/{repo}/releases/tags/{tag}').stdout)
names={x['name'] for x in release['assets']}
upload=[str(p) for p in [local,meta] if p.name not in names]
if upload:print(gh('release','upload',tag,*upload,'--repo',repo).stdout,flush=True)
release=json.loads(gh('api',f'repos/{repo}/releases/tags/{tag}').stdout)
asset=next(x for x in release['assets'] if x['name']==local.name)
assert asset['size']==data['bytes']
if asset.get('digest'):
    assert asset['digest']=='sha256:'+data['sha256'];verification='GitHub asset SHA256 digest and size match local/server'
else:
    vd=DEST/'verification';vd.mkdir(exist_ok=True)
    gh('release','download',tag,'--repo',repo,'--pattern',local.name,'--dir',str(vd),'--clobber')
    assert sha(vd/local.name)==data['sha256'];verification='GitHub downloaded bytes SHA256 match local/server'
    candidate=(vd/local.name).resolve();assert candidate.parent==vd.resolve();candidate.unlink()
data.update(github_release=release['html_url'],github_asset=asset['browser_download_url'],github_digest=asset.get('digest'),verification=verification)
(BASE/'results/initial_checkpoint_backup.json').write_text(json.dumps(data,indent=2))
(LOCAL/'initial_checkpoint_backup.json').write_text(json.dumps(data,indent=2))
print('CHECKPOINT_GITHUB_VERIFIED',json.dumps(data),flush=True)
