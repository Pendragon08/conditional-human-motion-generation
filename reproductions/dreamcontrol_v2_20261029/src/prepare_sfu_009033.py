"""Prepare an isolated full-rate SFU candidate from the audited ACCAD template."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'accad_004347_fullrate_20261031'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make(name, replacements):
    source = OLD / name
    original = source.read_text()
    result = original
    changes = []
    for old, new, count in replacements:
        assert result.count(old) == count, (name, old, result.count(old), count)
        result = result.replace(old, new)
        changes.append({'old': old, 'new': new, 'count': count})
    destination = HERE / name
    with destination.open('x') as stream:
        stream.write(result)
    return {'file': name, 'source_sha256': sha(source), 'output_sha256': sha(destination), 'changes': changes}


edits = [
    make('reconstruct.py', [('004347', '009033', 1), ("item['end']!=60", "item['end']!=72", 1),
                            ('len(frames)!=355', 'len(frames)!=427', 1), ('points.shape!=(355,22,3)', 'points.shape!=(427,22,3)', 1)]),
    make('audit_human.py', [('np.arange(355)', 'np.arange(427)', 1), ('[0,177,354]', '[0,213,426]', 1)]),
]
with (HERE / 'prepare_manifest.json').open('x') as output:
    json.dump({'changes': edits, 'training_ready': False, 'limitations': ['Template adaptations require verification; source FPS and identity remain unaccepted.']}, output, indent=2)
    output.write('\n')
