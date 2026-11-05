import json
from pathlib import Path
T=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/step34_20261029')
for name in ['user_vibed_controlled_runner_source_20261103_v5','user_vibed_controlled_main_launch_20261103_v1',
 'user_vibed_controlled_training_smoke_20261103_v2','user_vibed_native_scene_documentation_20261103_v1',
 'user_vibed_gpu_witness_documentation_20261103_v1','user_vibed_native_training_source_20261103_v1']:
 p=T/name
 if not p.exists():continue
 print(name,json.dumps([x.name for x in p.iterdir()]))
 for f in p.iterdir():
  if f.is_file() and f.suffix in ['.md','.json'] and f.stat().st_size<20000:
   print(str(f),f.read_text()[:11000])
