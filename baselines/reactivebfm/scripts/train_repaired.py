"""Opt-in FP32 recovery entrypoint; original train_full.py remains unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'upstream'))
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--resume', type=Path, required=True)
arguments, _ = parser.parse_known_args()
configuration = json.loads(arguments.config.read_text())
if configuration['use_bf16']:
    raise ValueError('This documented recovery variant requires full FP32 model computation')
launch = json.loads((arguments.output/'recovery_launch.json').read_text())
if launch['configuration_sha256'] != hashlib.sha256(arguments.config.read_bytes()).hexdigest():
    raise ValueError('Launch configuration changed')
if launch['resume'] != str(arguments.resume):
    raise ValueError('Launch checkpoint changed')
manifest_path = Path(configuration['recovery']['original_manifest'])
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != configuration['recovery']['original_manifest_sha256']:
    raise ValueError('Original ready manifest changed after recovery planning')
manifest = json.loads(manifest_path.read_text())
for row in manifest['files']:
    path = Path(row['file'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    if path.stat().st_size != row['bytes'] or digest.hexdigest() != row['sha256']:
        raise ValueError('Recovery checkpoint size/SHA256 mismatch: '+str(path))
if str(arguments.resume) not in [r['file'] for r in manifest['files'] if Path(r['file']).name.startswith('model')]:
    raise ValueError('Resume path differs from verified manifest')
from numerical_guard import install_guards
install_guards(arguments.output, configuration['recovery']['maximum_consecutive_skips'],
               configuration['recovery']['save_after_resumed_updates'])
runpy.run_path(str(BASE/'scripts/train_full.py'), run_name='__main__')
