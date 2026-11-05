"""Prepare three SSM human reconstructions using the audited earlier code.

Selected recordings are separate exact source paths; subject independence,
model equivalence, license and suitability for training are not assumed.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD = ROOT / 'humanml_mirror_time_witness_v2_20261030'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt = json.loads((OLD / 'report.json').read_text())
    source = OLD / 'ssm_joint_mirror_time_witness.py'
    if sha(source) != receipt['script_sha256']:
        raise ValueError('audited reconstruction script changed')
    raw = source.read_text(encoding='utf8')
    old = "for name in ('000008','002511'):"
    new = "for name in ('000069','003017','013303'):"
    if raw.count(old) != 1:
        raise ValueError('unexpected source selector')
    raw = raw.replace(old, new).replace('Two marker-aligned SSM candidates',
                                      'Three marker-aligned SSM candidates')
    out = HERE / 'reconstruct.py'
    with out.open('x', encoding='utf8', newline='\n') as f: f.write(raw)
    copied = {}
    for name, key in [('humanml_joint_mirroring.py', 'mirror_module_sha256'),
                      ('humanml_time_mapping.py', 'time_module_sha256')]:
        src = OLD / name
        if sha(src) != receipt[key]: raise ValueError('audited module changed')
        dst = HERE / name
        with dst.open('xb') as f: f.write(src.read_bytes())
        copied[name] = sha(dst)
    manifest = {'ids': ['000069', '003017', '013303'],
                'input_report_sha256': sha(OLD / 'report.json'),
                'input_script_sha256': sha(source), 'output_script_sha256': sha(out),
                'modules_sha256': copied, 'training_ready': False,
                'limitation': 'Different exact source paths; no claim of subject independence, model identity, quality or licensing.'}
    with (HERE / 'prepare_manifest.json').open('x', encoding='utf8') as f:
        json.dump(manifest, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__': main()
