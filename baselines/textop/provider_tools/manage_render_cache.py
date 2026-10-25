"""Publish the inventory before applying the narrowly scoped server cleaner."""
import argparse,json,pathlib,subprocess,sys
from session import REPO,remote,GH
BASE=REPO/'baselines/textop'
DEST=BASE/'results/data_preparation_20261025'
SERVER='/data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/'
sys.stdout.reconfigure(encoding='utf-8')

def collect(name):
    obj=json.loads(remote('import pathlib;print(pathlib.Path('+repr(SERVER+name+'.json')+').read_text())'))
    (DEST/(name+'.json')).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in obj.items() if k!='files'}),flush=True)
    return obj

def apply(commit):
    subprocess.run([GH,'api','repos/Pendragon08/conditional-human-motion-generation/git/commits/'+commit],check=True,stdout=subprocess.DEVNULL)
    for rel in ['scripts/clean_inactive_render_cache.py','results/data_preparation_20261025/render_cache_inventory.json']:
        old=subprocess.check_output(['git','-C',str(REPO),'show',commit+':baselines/textop/'+rel])
        current=(BASE/rel).read_bytes()
        assert (json.loads(old)==json.loads(current)) if rel.endswith('.json') else old==current
    source=(BASE/'scripts/clean_inactive_render_cache.py').read_text(encoding='utf-8')
    print(remote("import sys\nsys.argv=['-', '--apply']\nexec(compile("+repr(source)+",'<backed-up-cache-cleaner>','exec'))",timeout=180),flush=True)
    result=collect('render_cache_cleanup')
    result['source_and_inventory_backed_up_commit']=commit
    (DEST/'render_cache_cleanup.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply-backed-up-commit');a=p.parse_args()
    if a.apply_backed_up_commit:apply(a.apply_backed_up_commit)
    else:collect('render_cache_inventory')
