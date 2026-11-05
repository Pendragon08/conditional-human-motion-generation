"""Create a new CPU preflight configuration without touching prior datasets."""
import json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
TRAJ=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
cfg=json.loads((ROOT/'mixed_manual_caption_preview_20261101/config.json').read_text());cfg['run_dir']=str(HERE)
cfg['legacy_length_diagnostic']={'legacy_dataset':str(TRAJ/'data_loaders/humanml/data/dataset.py'),
    'legacy_dataset_sha256':'770d50ed52f4106b201fd9f37edc065fa18593731747ce78c7594f1b953822fa',
    'legacy_options':str(TRAJ/'dataset/humanml_opt.txt'),
    'legacy_options_sha256':'d72803f6112b04248f2a7168203a7fd455dcf6541a82ecb59d6c88e71c435ac2'}
with (HERE/'config.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
