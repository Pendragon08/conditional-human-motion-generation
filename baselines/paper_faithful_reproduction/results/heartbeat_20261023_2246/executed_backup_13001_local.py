from session import *
import shutil
import time

root=LOCAL/'heartbeat_20261023_2246'
target=LOCAL/'recovery_v4_checkpoint_13001_direct_v1'
target.mkdir(exist_ok=False)
manifest=json.loads((root/'checkpoint_13001_manifest.json').read_text(encoding='utf-8'))
publisher=json.loads(remote(f"from pathlib import Path;print(Path({(REMOTE+'/server_publication_13001_v1/checkpoint_receipt.json')!r}).read_text())"))
assert publisher['status']=='SERVER_AND_GITHUB_SHA256_VERIFIED'
dump(root/'server_checkpoint_publication_receipt.json',publisher)
repo='Pendragon08/conditional-human-motion-generation';tag='paper-reproduction-progress-20261023'
info=json.loads(subprocess.check_output([GH,'api',f'repos/{repo}/releases/tags/{tag}'],encoding='utf-8'))
assets={r['name']:r for r in info['assets']}
rows=[]
for item in manifest['files']:
    name='reactive_v4_'+Path(item['file']).name
    a=assets[name]
    assert a['size']==item['bytes'] and a['digest']=='sha256:'+item['sha256']
    assert shutil.disk_usage(target).free>item['bytes']+2*2**30
    cmd=[GH,'release','download',tag,'--repo',repo,'--pattern',name,'--dir',str(target)]
    start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',timeout=900)
    dump(root/(name+'_download_execution.json'),{'command':cmd,'elapsed_s':time.monotonic()-start,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    assert p.returncode==0,p.stderr
    local=target/name
    with local.open('rb') as f:
        h=hashlib.file_digest(f,'sha256').hexdigest()
    assert local.stat().st_size==item['bytes'] and h==item['sha256']
    rows.append({'name':name,'local_path':str(local),'bytes':local.stat().st_size,'sha256':h,'url':a['browser_download_url']})
receipt={'utc':utc(),'status':'GITHUB_AND_LOCAL_VERIFIED','method':'server_direct_github_then_local_download','optimizer_updates':13001,'copied':rows,'server_receipt':publisher,'credential_saved':False,'deletion':False}
dump(root/'checkpoint_13001_backup_receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
