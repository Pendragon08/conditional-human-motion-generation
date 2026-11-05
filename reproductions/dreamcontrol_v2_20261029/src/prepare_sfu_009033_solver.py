"""Adapt the audited full-rate window and audit launchers to SFU 009033."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD = ROOT / 'accad_004347_pyroki_fullrate_20261031'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(name, replacements):
    source = OLD / name
    text = source.read_text()
    changes = []
    for old, new in replacements:
        count = text.count(old)
        if not count:
            raise ValueError((name, old))
        text = text.replace(old, new)
        changes.append({'old': old, 'new': new, 'count': count})
    target = HERE / name
    with target.open('x', newline='\n') as output:
        output.write(text)
    return {'name': name, 'template_sha256': sha(source), 'output_sha256': sha(target), 'changes': changes}


common = [('accad_004347_fullrate_20261031', 'sfu_009033_fullrate_20261031'),
          ('accad_004347_pyroki_fullrate_20261031', 'sfu_009033_pyroki_fullrate_20261031')]
changes = [build('prepare.py', common + [('355', '427'), ("'expected_windows':88", "'expected_windows':106")]),
           build('prepare_audit.py', [common[0], ('355', '427'), ('354', '426'), ("== 88'", "== 106'")]),
           build('queue.sh', [common[1]])]
with (HERE / 'template_manifest.json').open('x') as output:
    json.dump({'files': changes, 'training_ready': False,
               'limitations': ['Old v1 solver and heuristic weights; no author-v2 or physical acceptance.',
                               'Declared source FPS, text and model identity require further validation.']}, output, indent=2)
    output.write('\n')
