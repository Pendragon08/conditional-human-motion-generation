"""Separate bounded online schema. Old offline receipts and source files stay immutable."""
import hashlib,json,pathlib,os,time
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(path,value):
    path=pathlib.Path(path);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def load_spec():
    sp=ROOT/'configs/quick-online-spec.json';s=json.loads(sp.read_text())
    assert os.getuid()==1013 and s['schema']=='textop.quick_online.v1' and s['validation_enabled']
    assert s['generator_torch_fraction']+s['sdk_torch_fraction']<=.65
    for n,h in s['bindings'].items():assert sha(ROOT/n)==h,n
    for x in s['fk_witnesses']:
        p=ROOT/x['receipt'];r=json.loads(p.read_text());assert sha(p)==x['sha256'] and r['passed']
    return s,sha(sp)
def request(out,index,phase,chunks=1):
    q={'index':index,'phase':phase,'text':['walk forward','wave your right hand'][phase],
       'chunks':chunks,'submitted_monotonic':time.monotonic()}
    p=out/('request_%03d.json'%index);assert not p.exists();write(p,q);return q
def response(out,index):
    p=out/('response_%03d.json'%index)
    if not p.exists():return None
    r=json.loads(p.read_text());assert r['index']==index
    a=out/r['npz'];assert a.parent==out and sha(a)==r['sha256']
    assert r['received_monotonic']>=r['submitted_monotonic'] and r['published_monotonic']>=r['received_monotonic']
    return r
