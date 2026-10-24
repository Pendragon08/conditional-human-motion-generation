"""Pinned official HF assets, direct final paths, resumable and hash verified.

Private dataset/weights storage is excluded from Git. No existing dataset is removed.
"""
import concurrent.futures,hashlib,json,os,pathlib,shutil,time,urllib.parse,urllib.request,threading
ROOT=pathlib.Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'configs/asset_manifest.json').read_text())
HOME=pathlib.Path('/home/ouyanghao/textop-assets-20261024')
PROXY='http://127.0.0.1:17897'
def chunked_transfer(item,partial):
    """Four ranges write directly into one file; journal allows safe sparse resume."""
    journal=partial.with_name(partial.name+'.ranges.json')
    if journal.exists():state=json.loads(journal.read_text())
    else:
        state={'prefix':partial.stat().st_size if partial.exists() else 0,'completed':[]}
        journal.write_text(json.dumps(state))
    prefix=state['prefix'];chunk=16*1024*1024
    ranges=[(a,min(a+chunk,item['size'])-1) for a in range(prefix,item['size'],chunk)]
    missing=[r for r in ranges if r[0] not in state['completed']]
    remaining=sum(b-a+1 for a,b in missing)
    if shutil.disk_usage(partial.parent).free<remaining+512*1024**2:raise RuntimeError('Disk reserve would be exhausted')
    fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o600);lock=threading.Lock()
    def get_range(pair):
        a,b=pair
        for attempt in range(6):
            try:
                url='https://huggingface.co/datasets/Yochish/TextOp-Data/resolve/'+MANIFEST['revision']+'/'+urllib.parse.quote(item['path'],safe='/')+'?download=true&t='+str(time.time_ns())
                req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':PROXY}))
                with opener.open(req,timeout=120) as response:
                    if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'):raise ValueError('Range not honored')
                    data=response.read()
                if len(data)!=b-a+1:raise ValueError('Incomplete range')
                if os.pwrite(fd,data,a)!=len(data):raise IOError('Incomplete write')
                with lock:
                    state['completed'].append(a)
                    temp=journal.with_suffix('.tmp');temp.write_text(json.dumps(state));temp.replace(journal)
                return
            except Exception:
                if attempt==5:raise
                time.sleep(2*(attempt+1))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('TEXTOP_RANGE_WORKERS','4'))) as pool:list(pool.map(get_range,missing))
        os.fsync(fd)
    finally:os.close(fd)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def download(item):
    rel=item['path']; expected=item.get('lfs',{}).get('oid')
    # Validation split and pretrained models live on the user's system-volume folder.
    external=rel.endswith('/val.pkl') or '/logs/' in rel
    dest=(HOME if external else ROOT/'assets')/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==item['size'] and (not expected or digest(dest)==expected):pass
    else:
        partial=dest.with_name(dest.name+'.partial')
        for attempt in range(6):
            try:
                if item['size']>64*1024**2:
                    chunked_transfer(item,partial)
                    if partial.stat().st_size!=item['size']:raise ValueError('Size mismatch '+rel)
                    if expected and digest(partial)!=expected:raise ValueError('SHA256 mismatch '+rel)
                    partial.replace(dest);break
                offset=partial.stat().st_size if partial.exists() else 0
                reserve=item['size']-offset+512*1024**2
                if shutil.disk_usage(dest.parent).free<reserve:raise RuntimeError('Insufficient space with 512MiB reserve: '+str(dest))
                url='https://huggingface.co/datasets/Yochish/TextOp-Data/resolve/'+MANIFEST['revision']+'/'+urllib.parse.quote(rel,safe='/')+'?download=true&t='+str(time.time_ns())
                req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':PROXY}))
                with opener.open(req,timeout=120) as response:
                    append=offset and response.status==206
                    with partial.open('ab' if append else 'wb') as out:
                        shutil.copyfileobj(response,out,8*1024*1024)
                if partial.stat().st_size!=item['size']:raise ValueError('Size mismatch '+rel)
                actual=digest(partial)
                if expected and actual!=expected:raise ValueError('SHA256 mismatch '+rel)
                partial.replace(dest);break
            except Exception as e:
                print(json.dumps({'event':'retry','path':rel,'attempt':attempt,'error':repr(e)}),flush=True)
                if attempt==5:raise
                time.sleep(3*(attempt+1))
    target=ROOT/'upstream'/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(dest)
    receipt={'path':rel,'local_path':str(dest),'bytes':dest.stat().st_size,'sha256':digest(dest),'lfs_sha256':expected,'verified':True}
    print(json.dumps({'event':'verified',**receipt}),flush=True)
    return receipt
def main():
    selected=MANIFEST['files']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(download,selected))
    output={'repo':MANIFEST['repo'],'revision':MANIFEST['revision'],'files':receipts,'total_bytes':sum(x['bytes'] for x in receipts)}
    (ROOT/'receipts/download_complete.json').write_text(json.dumps(output,indent=2))
    print('ALL_ASSETS_VERIFIED',output['total_bytes'],flush=True)
if __name__=='__main__':main()
