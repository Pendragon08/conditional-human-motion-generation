"""Create only a new isolated CPU preflight configuration."""
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
prior=ROOT/'source_bundle_preflight_20261101'/'config.json'
cfg=json.loads(prior.read_text())
cfg['run_dir']=str(HERE)
cfg['vertical_support_diagnostic']={
    'state_root':cfg['state_root'],'state_manifest':cfg['state_manifest'],
    'scene':'/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml',
    'asset_report':str(ROOT/'eight_static_robot_quality_20261031'/'report.json'),
    'asset_report_sha256':'46c5f3c6ef27a4e7159d1de90c7af69cc0339288f55e43cf6498a0f0ecb8a276'}
with (HERE/'config.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
