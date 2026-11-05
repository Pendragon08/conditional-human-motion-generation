"""Isolated weight-3 variant of the audited S1 sole-contact experiment."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
OLD_VARIANT=ROOT/'pyroki_sole_variant_20261030';NEW_VARIANT=ROOT/'pyroki_sole_w3_variant_20261030'
OLD_RUN=ROOT/'pyroki_s1_sole_pilot_20261030'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('unexpected occurrence')
    return text.replace(old,new)
def save(path,text):
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)

def main():
    manifest=json.loads((OLD_RUN/'manifest.json').read_text());previous=json.loads((OLD_VARIANT/'report.json').read_text())
    driver=OLD_RUN/'pyroki_grab_windowed_pilot.py';audit=OLD_RUN/'audit_pyroki_windowed_pilot.py';helper=OLD_VARIANT/'retarget_helpers/_utils.py'
    if (sha(driver)!=manifest['driver_sha256'] or sha(audit)!=manifest['audit_sha256'] or sha(helper)!=manifest['variant_helper_sha256'] or
        sha(OLD_VARIANT/'report.json')!=manifest['variant_report_sha256'] or previous['weights']['sole_contact']!=10.):raise ValueError('old variant changed')
    target=NEW_VARIANT/'retarget_helpers';target.mkdir(parents=True,exist_ok=False)
    save(target/'__init__.py',(OLD_VARIANT/'retarget_helpers/__init__.py').read_text());save(target/'_utils.py',helper.read_text())
    previous['weights']['sole_contact']=3.0
    previous['warning']='Experimental sole-clearance residual weight 3; same URDF geometry, assumed z=0 and source low-speed mask. Not an author-v2 loss or physical acceptance.'
    save(NEW_VARIANT/'report.json',json.dumps(previous,indent=2,allow_nan=False))
    code=replace_once(driver.read_text(),"VARIANT=ROOT/'pyroki_sole_variant_20261030'","VARIANT=ROOT/'pyroki_sole_w3_variant_20261030'")
    save(HERE/driver.name,code)
    code=replace_once(audit.read_text(),"helper=ROOT/'pyroki_sole_variant_20261030/retarget_helpers/_utils.py'","helper=ROOT/'pyroki_sole_w3_variant_20261030/retarget_helpers/_utils.py'")
    save(HERE/audit.name,code)
    out={'weight10_report_sha256':sha(OLD_VARIANT/'report.json'),'weight10_driver_sha256':sha(driver),'weight10_audit_sha256':sha(audit),
         'helper_sha256':sha(target/'_utils.py'),'variant_report_sha256':sha(NEW_VARIANT/'report.json'),
         'driver_sha256':sha(HERE/driver.name),'audit_sha256':sha(HERE/audit.name),'builder_sha256':sha(Path(__file__)),
         'sole_contact_weight':3.,'windows':5,'limitations':'S1 tuning sample, assumed plane/contact, heuristic objective; not independent or paper-author implementation.'}
    save(HERE/'manifest.json',json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
