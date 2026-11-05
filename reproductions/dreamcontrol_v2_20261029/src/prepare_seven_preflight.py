"""Prepare an isolated CPU-only preflight using an already verified preview."""
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
old = ROOT / 'six_quality_guard_v2_20261031/six_candidate_preflight_only.json'
cfg = json.loads(old.read_text())
assert cfg['device'] == 'cpu'
cfg.update({'purpose': 'CPU preflight only; no data or caption is accepted for training',
            'dataset_root': str(HERE), 'run_dir': str(HERE),
            'manifest': str(HERE / 'experimental_manifest_not_for_training.json'),
            'stats': str(HERE / 'experimental_stats_not_for_training.npz')})
with (HERE / 'preflight_config.json').open('x') as output:
    json.dump(cfg, output, indent=2)
    output.write('\n')
