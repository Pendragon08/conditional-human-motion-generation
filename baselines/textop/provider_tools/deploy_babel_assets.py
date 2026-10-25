"""Install verified private annotations in the user's server directory only."""
import json,pathlib,subprocess,sys
from session import REPO,remote
from restore_indexed_cache import sha
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
SOURCE=LOCAL/'datasets/BABEL';TARGET='/data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025/babel_v1_0'
sys.stdout.reconfigure(encoding='utf-8')

def main():
    alignment=json.loads((BASE/'results/data_preparation_20261025/babel_alignment.json').read_text(encoding='utf-8'))
    assert alignment['mismatch_sequences']==0
    names=['babel_v1-0_release.zip','train.json','val.json','test.json','extra_train.json','extra_val.json','textop_original_verified_pairs.private.json']
    files=[{'file':n,'bytes':(SOURCE/n).stat().st_size,'sha256':sha(SOURCE/n)} for n in names]
    plan=repr(files)
    preflight=f'''
import pathlib,os,shutil,json,hashlib
assert os.getuid()==1013
r=pathlib.Path({TARGET!r});r.mkdir(mode=0o700,exist_ok=True)
assert r.resolve()==r and r.stat().st_uid==1013
files={plan};need=[]
for x in files:
 p=r/x['file'];assert p.parent==r and not p.is_symlink()
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 else:need.append(x['file'])
assert shutil.disk_usage(r).free>sum(x['bytes'] for x in files)+1024**3
print(json.dumps({{'need':need}}))
'''
    need=json.loads(remote(preflight))['need']
    if need:
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519',
                        *[str(SOURCE/n) for n in need],'ouyanghao@10.147.17.32:'+TARGET+'/'],check=True,capture_output=True,timeout=240)
    check=f'''
import pathlib,hashlib,json,os,shutil
r=pathlib.Path({TARGET!r});files={plan}
for x in files:
 p=r/x['file'];assert p.stat().st_uid==1013 and p.stat().st_size==x['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];p.chmod(0o600)
print(json.dumps({{'status':'SERVER_PRIVATE_BABEL_INSTALLED_AND_SHA256_VERIFIED','files_verified':len(files),'bytes':sum(x['bytes'] for x in files),'server_directory':str(r),'data_free_bytes':shutil.disk_usage(r).free,'raw_annotations_published':False}}))
'''
    proof=json.loads(remote(check));proof['files']=files
    (BASE/'results/data_preparation_20261025/babel_server_install.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    (LOCAL/'BABEL服务器安装核验_20261025.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in proof.items() if k!='files'}),flush=True)

if __name__=='__main__':main()
