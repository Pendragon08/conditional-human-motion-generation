"""Create isolated 120Hz temporal-weight experiment from linear contact pilot."""
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def replace_once(data,a,b):
    if data.count(a)!=1:raise ValueError('expected one occurrence: '+a)
    return data.replace(a,b)

def save(p,s):
    with p.open('x',encoding='utf8',newline='\n') as f:f.write(s)

def main():
    base=ROOT/'pyroki_grab_linear_contact_20261030'
    receipt=json.loads((base/'report.json').read_text())
    helper=base/'retarget_helpers/_utils.py';pilot=base/'pyroki_grab_legacy_pilot.py'
    if sha(helper)!=receipt['helper_sha256'] or sha(pilot)!=receipt['script_sha256']:
        raise ValueError('linear baseline code hash mismatch')
    d=HERE/'retarget_helpers';d.mkdir(exist_ok=False);save(d/'__init__.py','')
    content=replace_once(helper.read_text(),'jnp.array([0.2]),','jnp.array([1.2]),')
    save(d/'_utils.py',content)
    content=replace_once(pilot.read_text(),"'root_smoothness':2.,'foot_skating':2.,'world_collision':1.",
                         "'root_smoothness':12.,'foot_skating':12.,'world_collision':1.")
    content=replace_once(content,'CPU legacy v1 PyRoki with linear foot residual variant and assumed z=0',
                         'CPU legacy v1 PyRoki with linear foot residual and sixfold temporal weights at 120Hz, with assumed z=0')
    save(HERE/'pyroki_grab_legacy_pilot.py',content)
    for name in ('watch_pyroki_grab_audit.py','audit_pyroki_grab_pilot.py'):
        save(HERE/name,(base/name).read_text())
    doc={'baseline_report_sha256':sha(base/'report.json'),'baseline_helper_sha256':sha(helper),
         'baseline_pilot_sha256':sha(pilot),'helper_sha256':sha(d/'_utils.py'),
         'pilot_sha256':sha(HERE/'pyroki_grab_legacy_pilot.py'),
         'description':'Same seven frames and linear-contact variant. Multiply per-frame root, foot and joint smoothness residual weights by 120/20=6 as a velocity-normalization experiment. This is not a paper-specified weight and may alter optimization tradeoffs.'}
    save(HERE/'variant_manifest.json',json.dumps(doc,indent=2));print(json.dumps(doc),flush=True)

if __name__=='__main__':main()
