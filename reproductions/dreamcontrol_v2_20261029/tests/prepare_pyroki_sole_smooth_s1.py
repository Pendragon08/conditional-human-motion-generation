"""Create one isolated smooth-min sole-loss comparison at fixed weight 3."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
OLD_VARIANT=ROOT/'pyroki_sole_w3_variant_20261030';NEW_VARIANT=ROOT/'pyroki_sole_smooth_variant_20261030';OLD_RUN=ROOT/'pyroki_s1_sole_w3_pilot_20261030'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def change(text,old,new):
    if text.count(old)!=1:raise ValueError('unexpected source match '+old[:60])
    return text.replace(old,new)
def write(path,text):
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)

def main():
    manifest=json.loads((OLD_RUN/'manifest.json').read_text());old_report=json.loads((OLD_VARIANT/'report.json').read_text())
    helper=OLD_VARIANT/'retarget_helpers/_utils.py';driver=OLD_RUN/'pyroki_grab_windowed_pilot.py';audit=OLD_RUN/'audit_pyroki_windowed_pilot.py'
    if sha(helper)!=manifest['helper_sha256'] or sha(OLD_VARIANT/'report.json')!=manifest['variant_report_sha256'] or sha(driver)!=manifest['driver_sha256'] or sha(audit)!=manifest['audit_sha256'] or old_report['weights']['sole_contact']!=3.:raise ValueError('weight-3 inputs changed')
    code=helper.read_text();code=change(code,'import jax\n','import jax\nimport jax.scipy.special as jsp\n')
    for side,idx in (('left',0),('right',1)):
        old=f'{side}_clearance = jnp.min({side}_centers[:, 2] - sole_radii[{idx}]) - {side}_foot_keypoints[2]'
        new=f'{side}_clearance = -0.002 * jsp.logsumexp(-({side}_centers[:, 2] - sole_radii[{idx}]) / 0.002) - {side}_foot_keypoints[2]'
        code=change(code,old,new)
    target=NEW_VARIANT/'retarget_helpers';target.mkdir(parents=True,exist_ok=False)
    write(target/'__init__.py',(OLD_VARIANT/'retarget_helpers/__init__.py').read_text());write(target/'_utils.py',code)
    report=dict(old_report);report['helper_sha256']=sha(target/'_utils.py');report['original_sole_helper_sha256']=sha(helper)
    report['smooth_temperature_m']=.002;report['smooth_bias_upper_bound_m']=.002*math.log(4)
    report['warning']='Experimental smooth minimum of four sphere clearances, temperature 2mm and weight 3, assumed plane/low-speed mask. Not an author-v2 loss or physical acceptance.'
    write(NEW_VARIANT/'report.json',json.dumps(report,indent=2,allow_nan=False))
    write(HERE/driver.name,change(driver.read_text(),"VARIANT=ROOT/'pyroki_sole_w3_variant_20261030'","VARIANT=ROOT/'pyroki_sole_smooth_variant_20261030'"))
    write(HERE/audit.name,change(audit.read_text(),"helper=ROOT/'pyroki_sole_w3_variant_20261030/retarget_helpers/_utils.py'","helper=ROOT/'pyroki_sole_smooth_variant_20261030/retarget_helpers/_utils.py'"))
    out={'old_helper_sha256':sha(helper),'new_helper_sha256':sha(target/'_utils.py'),'old_variant_sha256':sha(OLD_VARIANT/'report.json'),
         'new_variant_sha256':sha(NEW_VARIANT/'report.json'),'driver_sha256':sha(HERE/driver.name),'audit_sha256':sha(HERE/audit.name),
         'script_sha256':sha(Path(__file__)),'temperature_m':.002,'smooth_bias_upper_bound_m':.002*math.log(4),'weight':3.,'windows':5,
         'warning':'Single S1 tuning clip, fixed weight and source, smooth-min replaces hard min only; plane/low-speed labels assumed. Not independent physical or paper validation.'}
    write(HERE/'manifest.json',json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
