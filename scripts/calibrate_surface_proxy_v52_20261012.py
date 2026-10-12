"""V50's first CPU attempt failed on local-only export path; preserve it.

This separate attempt reads execution-bound remote raw metadata, not public
export layout. No analytical control, metric or threshold is changed.
"""
from pathlib import Path
parent = Path(__file__).with_name('calibrate_surface_proxy_v50_20261012.py')
source = parent.read_text(encoding='utf8')
assert source.count('surface_proxy_calibration_v50_20261012') == 1
assert source.count('public/raw/val_native_replay_s11/metadata.json') == 1
source = source.replace('surface_proxy_calibration_v50_20261012', 'surface_proxy_calibration_v52_20261012')
source = source.replace('public/raw/val_native_replay_s11/metadata.json', 'raw/val_native_replay_s11/metadata.json')
source = source.replace('source_sha256=sha(__file__),', 'source_sha256=sha(__file__), parent_source_sha256=sha(parent),')
exec(compile(source, str(Path(__file__)), 'exec'), globals())
