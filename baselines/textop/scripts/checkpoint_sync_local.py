"""Sync this one authorized TextOp run; archive only verified old server copies.

Datasets, pretrained assets and all other runs/users are outside this scope.
All checkpoint versions remain in the local report folder and GitHub Releases.
"""
import argparse,hashlib,json,os,pathlib,subprocess,time,traceback

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--once',action='store_true');args=ap.parse_args()
    c=json.loads(pathlib.Path(args.config).read_text(encoding='utf-8'))
    base=pathlib.Path(c['baseline']);local=pathlib.Path(c['local']);dest=local/'checkpoints'
    statefile=base/'results/checkpoint_sync_state.json'
    state=json.loads(statefile.read_text(encoding='utf-8')) if statefile.exists() else {'backups':{},'server_archives':[]}
    initial=json.loads((base/'results/initial_checkpoint_backup.json').read_text(encoding='utf-8'))
    initial.update(loop_step=initial['step'],local_path=str(dest/'TextOp_MVAE_Public_step1000.pth'))
    state['backups'].setdefault(initial['path'],initial)
    env=os.environ.copy();env.update(HTTPS_PROXY=c['proxy'],HTTP_PROXY=c['proxy'])
    def remote(script,py='python3'):
        p=subprocess.run(c['ssh']+[py+' -'],input=script,capture_output=True,text=True,encoding='utf-8',timeout=180)
        if p.returncode:raise RuntimeError(p.stderr[-2000:]+p.stdout[-2000:])
        return json.loads(p.stdout)
    def gh(*a,check=True):
        return subprocess.run([c['gh'],*a],env=env,capture_output=True,text=True,encoding='utf-8',check=check,timeout=3600)
    def persist():
        state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        temp=statefile.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(statefile)
        (local/'checkpoint_sync_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    def inventory():
        return remote(f'''
import pathlib,json,os,time,shutil
r=pathlib.Path({c['run_dir']!r});root=pathlib.Path({c['server_root']!r})
assert root.resolve() in r.resolve().parents and r.name.startswith('train-mvae-')
files=[]
for p in r.glob('ckpt_*.pth'):
    if p.is_symlink() or p.stat().st_uid!=1013 or time.time()-p.stat().st_mtime<30:continue
    files.append({{'path':str(p),'loop_step':int(p.stem.split('_')[1]),'bytes':p.stat().st_size}})
proc=pathlib.Path('/proc/{c['training_pid']}/cmdline')
alive=proc.exists() and str(root/'scripts/train_official.py') in proc.read_bytes().decode(errors='replace')
print(json.dumps({{'files':sorted(files,key=lambda x:x['loop_step']),'alive':alive,'free':shutil.disk_usage(root).free}}))
''')
    def verify_asset(receipt):
        release=json.loads(gh('api',f"repos/{c['github_repo']}/releases/tags/{receipt['tag']}").stdout)
        asset=next(x for x in release['assets'] if x['name']==pathlib.Path(receipt['local_path']).name)
        assert asset['size']==receipt['bytes'] and asset['digest']=='sha256:'+receipt['sha256']
    # Normalize the pre-existing initial backup, verified by the startup task.
    state['backups'][initial['path']]['tag']='textop-mvae-initial-20261024'
    while True:
        try:
            inv=inventory();changed=False
            for item in inv['files']:
                if item['path'] in state['backups']:continue
                proof=remote(f'''
import torch,pathlib,json,hashlib
torch.set_num_threads(1);p=pathlib.Path({item['path']!r});s=torch.load(p,map_location='cpu')
assert s['step']=={item['loop_step']}
assert all(torch.isfinite(t).all() for t in s['vae'].values())
assert all(torch.isfinite(t).all() for st in s['optimizer']['state'].values() for t in st.values() if torch.is_tensor(t))
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
steps=[int(st['step']) for st in s['optimizer']['state'].values()]
print(json.dumps({{'sha256':h.hexdigest(),'optimizer_step_min':min(steps),'optimizer_step_max':max(steps)}}))
''',c['server_python'])
                item.update(proof);target=dest/f"TextOp_MVAE_Public_step{item['loop_step']}.pth"
                if not target.exists() or digest(target)!=item['sha256']:
                    if __import__('shutil').disk_usage(dest).free<item['bytes']+512*1024**2:raise RuntimeError('Local disk reserve insufficient; retaining all server files')
                    subprocess.run(c['scp']+[c['host']+':'+item['path'],str(target)],check=True,timeout=1800)
                assert target.stat().st_size==item['bytes'] and digest(target)==item['sha256']
                tag='textop-mvae-public-20261024'
                if gh('release','view',tag,'--repo',c['github_repo'],check=False).returncode:
                    notes=dest/'full-mvae-release-notes.md';notes.write_text('TextOp public-data MVAE training checkpoints. Training is in progress; these are not final paper or closed-loop results. Raw AMASS/SMPL data are excluded. Loop steps and actual optimizer steps are recorded separately in JSON receipts.\n')
                    gh('release','create',tag,'--repo',c['github_repo'],'--target','codex/textop-reproduction-20261024','--title','TextOp public MVAE checkpoints (training in progress)','--notes-file',str(notes),'--prerelease')
                release=json.loads(gh('api',f"repos/{c['github_repo']}/releases/tags/{tag}").stdout)
                if target.name not in {x['name'] for x in release['assets']}:gh('release','upload',tag,str(target),'--repo',c['github_repo'])
                item.update(local_path=str(target),tag=tag,github_release=f"https://github.com/{c['github_repo']}/releases/tag/{tag}")
                verify_asset(item)
                state['backups'][item['path']]=item;persist();changed=True
                print('VERIFIED_CHECKPOINT_BACKUP',json.dumps(item),flush=True)
            # Keep the newest complete server checkpoint. All versions remain local and on GitHub.
            inv=inventory();files=inv['files']
            if len(files)>1 and all(x['path'] in state['backups'] for x in files):
                newest=state['backups'][files[-1]['path']]
                assert digest(pathlib.Path(newest['local_path']))==newest['sha256'];verify_asset(newest)
                for item in files[:-1]:
                    receipt=state['backups'][item['path']]
                    assert digest(pathlib.Path(receipt['local_path']))==receipt['sha256'];verify_asset(receipt)
                    event=remote(f'''
import pathlib,hashlib,json,os
r=pathlib.Path({c['run_dir']!r}).resolve();p=pathlib.Path({item['path']!r})
assert p.resolve().parent==r and not p.is_symlink() and p.stat().st_uid==1013
assert p.name=="ckpt_{item['loop_step']}.pth" and p.stat().st_size=={receipt['bytes']}
assert (r/"ckpt_{files[-1]['loop_step']}.pth").exists()
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert h.hexdigest()=={receipt['sha256']!r}
p.unlink()
print(json.dumps({{'archived_server_path':str(p),'sha256':h.hexdigest(),'reason':'newer server checkpoint retained; this exact version is verified local and GitHub'}}))
''')
                    state['server_archives'].append(event);persist();changed=True
            state['last_inventory']=inv;state.pop('last_error',None);persist()
            if changed or not inv['alive']:
                for name in ['collect.py','publish.py']:
                    subprocess.run([c['local_python'],str(pathlib.Path(c['orchestrator'])/name)],check=True,timeout=300)
            if args.once or not inv['alive']:return
        except Exception:
            state['last_error']=traceback.format_exc();persist();print(state['last_error'],flush=True)
            if args.once:raise
        time.sleep(60)

if __name__=='__main__':main()
