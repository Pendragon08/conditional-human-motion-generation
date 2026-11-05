"""Attach the experimental encoder threshold to hash-bound sampled-state evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'state_preflight_v1_20261031'
REENCODING = ROOT / 'seven_full_feature_reencoding_20261031/report.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original = OLD / 'state_manifest.json'
data = json.loads(original.read_text())
check = json.loads(REENCODING.read_text())
if sha(original) != check['state_manifest_sha256'] or check['foot_velocity_squared_threshold'] != .002:
    raise ValueError('re-encoding provenance mismatch')
rows = {row['sample_id']: row for row in check['cases']}
if len(rows) != len(data['records']):
    raise ValueError('missing or duplicate re-encoding records')
for rec in data['records']:
    witness = rows[rec['sample_id']]
    if (rec['feature_sha256'] != witness['feature_sha256'] or rec['state_sha256'] != witness['state_sha256'] or
        witness['max_abs_diff'] != 0 or witness['changed_foot_proxy_values'] != 0):
        raise ValueError('re-encoding result mismatch')
    rec['foot_velocity_squared_threshold'] = .002
data['prior_state_manifest_sha256'] = sha(original)
data['reencoding_report_sha256'] = sha(REENCODING)
with (HERE / 'state_manifest.json').open('x') as output:
    json.dump(data, output, indent=2)
    output.write('\n')
cfg = json.loads((OLD / 'preflight_config.json').read_text())
cfg['state_manifest'] = str(HERE / 'state_manifest.json')
with (HERE / 'preflight_config.json').open('x') as output:
    json.dump(cfg, output, indent=2)
    output.write('\n')
