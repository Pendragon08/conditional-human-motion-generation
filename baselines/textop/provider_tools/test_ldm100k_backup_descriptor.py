"""New isolated LDM-versus-VAE/endpoint descriptor rejection tests only."""
import copy
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from ldm100k_backup_contract import PATH, validate_descriptor

c = dict(path=PATH, manager_step=100000, bytes=1234, actual_optimizer_updates=100000, sha256='1' * 64)
b = dict(run='user_ldm100k_prefix_export_v1', receipt_sha256='2' * 64, exit_sha256='3' * 64, spec_sha256='4' * 64)
validate_descriptor(c, b)
count = 0
for key, value in [('path', PATH.replace('ldm_state', 'vae_state')), ('path', PATH.replace('100000', '300000')),
                   ('manager_step', 99700), ('manager_step', 300000), ('bytes', 0), ('bytes', True),
                   ('bytes', 513 * 1024**2), ('actual_optimizer_updates', 0),
                   ('actual_optimizer_updates', 100001), ('sha256', 'X' * 64)]:
    bad = copy.deepcopy(c)
    bad[key] = value
    try:
        validate_descriptor(bad, b)
    except AssertionError:
        count += 1
    else:
        raise AssertionError('Wrong state/endpoint descriptor accepted')
bad = copy.deepcopy(b)
bad['run'] = 'user_vae100k_export_v1'
try:
    validate_descriptor(c, bad)
except AssertionError:
    count += 1
else:
    raise AssertionError('VAE provenance accepted as LDM')
print(json.dumps(dict(status='LDM100K_BACKUP_NEW_ISOLATED_DESCRIPTOR_PASSED', negative_cases=count,
    actual_checkpoint_reads=0, actual_model_loads=0, actual_transfers=0,
    actual_backup_acceptance=False, dynamic_ownership_or_network_tested=False)))
