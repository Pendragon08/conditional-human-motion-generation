"""Freeze S4-derived alignment variant and prepare S2/S3 fixed-window checks."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def replace_once(s,a,b):
    if s.count(a)!=1:raise ValueError('unexpected source pattern: '+a)
    return s.replace(a,b)

def main():
    experiment=ROOT/'pyroki_s4_world_offset_variant_20261030'
    audit=experiment/'independent_comparison.json'
    if sha(audit)!='f6e005d6a91c443f8d456b846aaccb60aab2c556cdcaab250a507d3ea3b8c662':raise ValueError('tuning audit changed')
    result=json.loads(audit.read_text());helper=experiment/'retarget_helpers/_utils.py'
    if sha(helper)!=result['helper_variant_sha256'] or not result['only_declared_world_residual_change']:raise ValueError('variant changed')
    variant=OUT/'variant';(variant/'retarget_helpers').mkdir(parents=True)
    (variant/'retarget_helpers/__init__.py').open('x').close()
    text=replace_once(helper.read_text(),'    return transform, solution[var_joints], offset, solution[var_Ts_world_root]\n',
                      '    return transform, solution[var_joints]\n')
    with (variant/'retarget_helpers/_utils.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
    source_variant=ROOT/'pyroki_grab_temporal_20261030/report.json';base=json.loads(source_variant.read_text())
    base['helper_sha256']=sha(variant/'retarget_helpers/_utils.py')
    base['warning']='Fixed exploratory world-offset global alignment. Same legacy objective otherwise; no paper v2 or physical acceptance.'
    with (variant/'report.json').open('x',encoding='utf8') as f:json.dump(base,f,indent=2,allow_nan=False)
    audit_template=ROOT/'pyroki_s1_sole_w3_pilot_20261030/audit_pyroki_windowed_pilot.py'
    if sha(audit_template)!='184060d22de5a0045c756461ccbdb83e78da09a6e3f7315f75144799c435689d':raise ValueError('audit template changed')
    jobs=[('s2','pyroki_grab_windowed_20261030','grab_human_robot_pilot_20261030/pilot_arrays.npz'),
          ('s3','pyroki_grab_s3_windowed_20261030','grab_s3_human_source_20261030/human_source.npz')]
    manifests=[]
    for label,base_name,source_name in jobs:
        previous=ROOT/base_name;prior=json.loads((previous/'report.json').read_text());driver=previous/'pyroki_grab_windowed_pilot.py'
        if sha(driver)!=prior['script_sha256'] or sha(ROOT/source_name)!=prior['source_sha256']:raise ValueError('baseline code or source changed')
        directory=OUT/label;directory.mkdir()
        s=driver.read_text()
        s=replace_once(s,"VARIANT=ROOT/'pyroki_grab_temporal_20261030'","VARIANT=ROOT/'pyroki_world_offset_holdout_20261030/variant'")
        s=replace_once(s,'    starts=list(range(1,113,4)); starts.append(112)','    starts=list(range(1,18,4))')
        s=replace_once(s,'29 overlapping seven-frame windows','5 overlapping seven-frame windows')
        with (directory/'pyroki_grab_windowed_pilot.py').open('x',encoding='utf8',newline='\n') as f:f.write(s)
        a=audit_template.read_text()
        a=replace_once(a,"ROOT/'grab_s1_pour_human_source_20261030/human_source.npz'",f"ROOT/'{source_name}'")
        a=replace_once(a,"ROOT/'pyroki_sole_w3_variant_20261030/retarget_helpers/_utils.py'","ROOT/'pyroki_world_offset_holdout_20261030/variant/retarget_helpers/_utils.py'")
        with (directory/'audit_pyroki_windowed_pilot.py').open('x',encoding='utf8',newline='\n') as f:f.write(a)
        manifests.append({'subject':label,'source_sha256':sha(ROOT/source_name),'baseline_dir':base_name,
                          'driver_sha256':sha(directory/'pyroki_grab_windowed_pilot.py'),'audit_sha256':sha(directory/'audit_pyroki_windowed_pilot.py')})
    manifest={'script_sha256':sha(__file__),'fixed_s4_audit_sha256':sha(audit),'fixed_debug_helper_sha256':sha(helper),
              'return_removed_helper_sha256':base['helper_sha256'],'source_variant_report_sha256':sha(source_variant),
              'variant_report_sha256':sha(variant/'report.json'),'audit_template_sha256':sha(audit_template),'jobs':manifests,
              'limitations':'Two 5-window CPU source checks; same weight set and approximate legacy solver; original source FPS and flat ground remain assumptions. No training acceptance.'}
    with (OUT/'manifest.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2,allow_nan=False)
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
