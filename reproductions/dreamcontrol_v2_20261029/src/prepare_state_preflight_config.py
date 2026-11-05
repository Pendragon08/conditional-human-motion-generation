"""Prepare a CPU-only state-preflight config without changing existing runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
cfg = json.loads((ROOT / 'seven_candidate_technical_preview_20261031/preflight_config.json').read_text())
assert cfg['device'] == 'cpu'
cfg.update({'state_root': str(HERE), 'state_manifest': str(HERE / 'state_manifest.json'),
            'state_urdf': str(URDF), 'state_urdf_sha256': hashlib.sha256(URDF.read_bytes()).hexdigest(),
            'purpose': 'CPU-only technical validation, not an accepted training set'})
with (HERE / 'preflight_config.json').open('x') as output:
    json.dump(cfg, output, indent=2)
    output.write('\n')
