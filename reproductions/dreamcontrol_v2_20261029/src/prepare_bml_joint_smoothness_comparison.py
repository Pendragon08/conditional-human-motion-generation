"""Prepare a fixed joint-smoothness comparison for both BML candidates.

Only the old helper's joint first-difference coefficient changes (1.2 -> 4.8).
This is an exploratory alternative, not a paper parameter or a quality gate.
The previous solver inputs, results and code remain unchanged.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD=ROOT/'bml_global_candidates_20261031'
VARIANT=ROOT/'pyroki_grab_temporal_20261030'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('unexpected template: '+old)
    return text.replace(old,new)


def write(path,text):
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)


def main():
    original=json.loads((OLD/'manifest.json').read_text())
    receipt=json.loads((VARIANT/'report.json').read_text())
    helper=VARIANT/'retarget_helpers/_utils.py'
    if sha(helper)!=receipt['helper_sha256']:raise ValueError('old helper changed')
    new_helper=HERE/'retarget_helpers';new_helper.mkdir()
    init=VARIANT/'retarget_helpers/__init__.py'
    write(new_helper/'__init__.py',init.read_text())
    old='jnp.array([1.2]),';new='jnp.array([4.8]),'
    write(new_helper/'_utils.py',replace_once(helper.read_text(),old,new))
    adapted={**receipt,'helper_sha256':sha(new_helper/'_utils.py')}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(adapted,f,indent=2,allow_nan=False);f.write('\n')
    records=[]
    for case in original['cases']:
        name=case['label'];old_dir=OLD/name;new_dir=HERE/name;new_dir.mkdir()
        if sha(old_dir/'human.npz')!=case['human_sha256'] or sha(old_dir/'run.py')!=case['solver_sha256']:
            raise ValueError('prior source or solver changed')
        (new_dir/'human.npz').symlink_to(old_dir/'human.npz')
        code=replace_once((old_dir/'run.py').read_text(),
                          "VARIANT = ROOT / 'pyroki_grab_temporal_20261030'",
                          "VARIANT = ROOT / 'bml_joint_smoothness_20261031'")
        write(new_dir/'run.py',code)
        shell=replace_once((old_dir/'run.sh').read_text(),
                           'bml_global_candidates_20261031/'+name,
                           'bml_joint_smoothness_20261031/'+name)
        write(new_dir/'run.sh',shell)
        records.append({**case,'solver_sha256':sha(new_dir/'run.py'), 'launcher_sha256':sha(new_dir/'run.sh')})
    out={'cases':records,'previous_manifest_sha256':sha(OLD/'manifest.json'),
         'previous_helper_sha256':sha(helper),'new_helper_sha256':sha(new_helper/'_utils.py'),
         'change':{'joint_smoothness_coefficient_before':1.2,'after':4.8},
         'training_ready':False,'limitations':['single fixed exploratory alternative on two short clips',
                     'no official threshold, no claim of author parameters or trained data',
                     'source time/model, caption, physical and quality validation incomplete']}
    with (HERE/'manifest.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
