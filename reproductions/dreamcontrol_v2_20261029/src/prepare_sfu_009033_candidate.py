"""Prepare audited full-rate SFU blend, downsampling and encoding scripts."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD = ROOT / 'accad_004347_fullrate_candidate_20261031'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(name, changes):
    source = OLD / name
    data = source.read_text()
    edits = []
    for old, new, expected in changes:
        count = data.count(old)
        if count != expected:
            raise ValueError((name, old, count, expected))
        data = data.replace(old, new)
        edits.append({'before': old, 'after': new, 'count': count})
    target = HERE / name
    with target.open('x', newline='\n') as output:
        output.write(data)
    return {'name': name, 'template_sha256': sha(source), 'output_sha256': sha(target), 'edits': edits}


common = [('accad_004347_pyroki_fullrate_20261031', 'sfu_009033_pyroki_fullrate_20261031', 1)]
export = common + [('accad_004347_fullrate_20261031', 'sfu_009033_fullrate_20261031', 1),
                   ("check['windows_verified']!=88", "check['windows_verified']!=106", 1),
                   ('[1,354]', '[1,426]', 1), ('np.arange(1,355)', 'np.arange(1,427)', 1),
                   ('filtered.shape!=(59,34)', 'filtered.shape!=(71,34)', 1),
                   ('len(out_frames)!=59', 'len(out_frames)!=71', 1),
                   ('out_frames[-1]!=354', 'out_frames[-1]!=426', 1),
                   ('features.shape!=(58,263)', 'features.shape!=(70,263)', 1),
                   ("'id':'004347'", "'id':'009033'", 1), ("'filter_input_frames':[6,354]", "'filter_input_frames':[6,426]", 1)]
audit = common + [('np.arange(1,355)', 'np.arange(1,427)', 1),
                  ('np.arange(6,355,6)', 'np.arange(6,427,6)', 1),
                  ('features.shape==(58,263)', 'features.shape==(70,263)', 1),
                  ('filtered.shape==(59,34)', 'filtered.shape==(71,34)', 1),
                  ('(6,180,354)', '(6,216,426)', 1), ('[6,180,354]', '[6,216,426]', 1)]
watch = [('accad_004347_fullrate_candidate_20261031', 'sfu_009033_fullrate_candidate_v2_20261031', 1),
         ('accad_004347_pyroki_fullrate_20261031', 'sfu_009033_pyroki_fullrate_20261031', 2)]
files = [build('export.py', export), build('audit.py', audit), build('run_accad_candidate_after_windows.sh', watch)]
for name in ('blend_robot_windows.py', 'resample_g1_27.py', 'feature263.py', 'g1_fk.py'):
    source = OLD / name
    target = HERE / name
    with target.open('xb') as output:
        output.write(source.read_bytes())
    assert sha(source) == sha(target)
    files.append({'name': name, 'template_sha256': sha(source), 'output_sha256': sha(target), 'edits': []})
with (HERE / 'prepare_manifest.json').open('x') as output:
    json.dump({'files': files, 'training_ready': False,
               'limitations': ['Window filtering and 263D encoding are experimental.',
                               'Failure of solver or 263D self-check must not be bypassed.']}, output, indent=2)
    output.write('\n')
