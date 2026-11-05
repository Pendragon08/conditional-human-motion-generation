"""Create an isolated legacy PyRoki variant changing only contact residuals."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def replace_once(data,old,new):
    if data.count(old)!=1:raise ValueError('expected exactly one occurrence: '+old)
    return data.replace(old,new)


def write_exclusive(path,text):
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)


def main():
    original=ROOT/'pyroki_grab_pilot_20261030'
    report=json.loads((original/'report.json').read_text())
    helper=PROJECT/'TrajGen/sample/retarget_helpers/_utils.py'
    pilot=original/'pyroki_grab_legacy_pilot.py'
    if sha(helper)!=report['helper_sha256'] or sha(pilot)!=report['script_sha256']:
        raise ValueError('original code differs from completed pilot')
    source=helper.read_text()
    source=replace_once(source,'is_left_foot_contact * (left_foot_pos - left_foot_keypoints) ** 2',
                        'is_left_foot_contact * (left_foot_pos - left_foot_keypoints)')
    source=replace_once(source,'is_right_foot_contact * (right_foot_pos - right_foot_keypoints) ** 2',
                        'is_right_foot_contact * (right_foot_pos - right_foot_keypoints)')
    d=HERE/'retarget_helpers';d.mkdir(exist_ok=False)
    write_exclusive(d/'__init__.py','')
    write_exclusive(d/'_utils.py',source)
    code=pilot.read_text()
    code=replace_once(code,"sys.path.insert(0,str(PROJECT/'TrajGen/sample'))",
                      "sys.path.insert(0,str(Path(__file__).resolve().parent))")
    code=replace_once(code,"sha(PROJECT/'TrajGen/sample/retarget_helpers/_utils.py')",
                      "sha(Path(__file__).resolve().parent/'retarget_helpers/_utils.py')")
    code=replace_once(code,'CPU legacy v1 PyRoki with assumed z=0',
                      'CPU legacy v1 PyRoki with linear foot residual variant and assumed z=0')
    write_exclusive(HERE/'pyroki_grab_legacy_pilot.py',code)
    audit=(original/'audit_pyroki_grab_pilot.py').read_text()
    audit=replace_once(audit,"run=ROOT/'pyroki_grab_pilot_20261030'; report=run/'report.json'; artifact=run/'raw_pilot.npz'",
                       "run=Path(__file__).resolve().parent; report=run/'report.json'; artifact=run/'raw_pilot.npz'")
    write_exclusive(HERE/'audit_pyroki_grab_pilot.py',audit)
    watcher=(original/'watch_pyroki_grab_audit.py').read_text()
    write_exclusive(HERE/'watch_pyroki_grab_audit.py',watcher)
    manifest={'baseline_helper_sha256':sha(helper),'variant_helper_sha256':sha(d/'_utils.py'),
              'baseline_pilot_sha256':sha(pilot),'variant_pilot_sha256':sha(HERE/'pyroki_grab_legacy_pilot.py'),
              'baseline_report_sha256':sha(original/'report.json'),
              'description':'Change only the two squared foot-position residuals to linear residuals; least-squares already squares residuals. Identical data, weights and other objectives; pilot limited to seven frames and assumed plane.'}
    write_exclusive(HERE/'variant_manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
