"""Prepare a new isolated preflight; preserve all prior configs and results."""
import json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'vertical_support_preflight_20261101/config.json').read_text())
cfg['run_dir']=str(HERE)
cfg['caption_archive_binding']={'archive':'/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip',
    'evidence_root':cfg['caption_evidence_root'],'evidence_manifest':cfg['caption_evidence_manifest'],
    'state_root':cfg['state_root'],'state_manifest':cfg['state_manifest']}
with (HERE/'config.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
