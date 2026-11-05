"""Instrument the legacy optimizer return values without changing its objective."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    base=ROOT/'pyroki_grab_temporal_20261030';base_report=json.loads((base/'report.json').read_text())
    source=base/'retarget_helpers/_utils.py'
    if sha(source)!=base_report['helper_sha256']:raise ValueError('base helper changed')
    old='    return transform, solution[var_joints]\n'
    new='    return transform, solution[var_joints], offset, solution[var_Ts_world_root]\n'
    text=source.read_text()
    if text.count(old)!=1:raise ValueError('unexpected helper return')
    package=OUT/'retarget_helpers';package.mkdir()
    (package/'__init__.py').open('x').close()
    with (package/'_utils.py').open('x',encoding='utf8',newline='\n') as f:f.write(text.replace(old,new))
    result={'base_report_sha256':sha(base/'report.json'),'base_helper_sha256':sha(source),
            'instrumented_helper_sha256':sha(package/'_utils.py'),'script_sha256':sha(__file__),
            'change':'Only return shared offset and unshifted root in addition to original outputs; objective and solver unchanged.',
            'limitations':'Diagnostic instrumentation of legacy approximation, not paper v2 code or physical acceptance.'}
    with (OUT/'manifest.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
