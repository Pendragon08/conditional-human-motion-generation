"""Own one bounded GPU inference group; common lock excludes all other model/SDK jobs."""
import argparse
import fcntl
import json
import os
import shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from ldm100k_verified_input import ROOT, read, sha, owned
from ldm100k_scope_acceptance import accepted_scope
from ldm100k_generation_acceptance import RUNS

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--mode', choices=['witness', 'diagnostic'], required=True)
a = p.parse_args()
assert os.getuid() == 1013
assert a.run in {'witness': RUNS,
                 'diagnostic': ['user_ldm100k_diagnostic32_v1']}[a.mode]
sp = owned(ROOT / 'configs/ldm100k-generation-spec.json')
s = read(sp)
assert s['schema'] == 'textop.ldm100k_heldout_generation.v1'
assert s['witnesses_authorized'] if a.mode == 'witness' else s['generation_authorized']
assert sha(__file__) == s['supervisor_sha256']
assert sha(ROOT / 'scripts/run_ldm100k_heldout_generation.py') == s['executor_sha256']
for name, digest in s['bound_sources'].items():
    assert sha(ROOT / name) == digest, name
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
accepted_scope(s['scope_binding'])
budget = s['accepted_output_upper_bytes'][a.mode]
assert type(budget) is int and budget > 0
assert shutil.disk_usage(ROOT).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + budget
prefix = 'ldm100k_generation_' + a.run
assert not (ROOT / 'ldm100k_user_handoff_20261028' / a.run).exists()
ep = ROOT / 'receipts' / (prefix + '_exit.json')
assert not ep.exists()
cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=1',
       'PYTHONIOENCODING=utf-8', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=4',
       'OPENBLAS_NUM_THREADS=4', str(ROOT / 'env/bin/python'),
       str(ROOT / 'scripts/run_ldm100k_heldout_generation.py'), '--run', a.run, '--mode', a.mode]
resources = ROOT / 'logs' / (prefix + '_resources.jsonl')
process = ROOT / 'receipts' / (prefix + '_process_exit.json')
proof = run_supervised(cmd, ROOT / 'logs' / (prefix + '.log'), resources, process,
                       lambda pid: sample_resources(ROOT, pid, s['minimum_ram_free_bytes'],
                                                    s['minimum_disk_free_bytes']),
                       s['timeout_seconds'][a.mode], 15)
rp = ROOT / 'ldm100k_user_handoff_20261028' / a.run / 'receipt.json'
rec = read(rp) if rp.exists() else {}
proof.update(run=a.run, mode=a.mode, status=rec.get('status'), source_sha256=sha(__file__),
             spec_sha256=sha(sp), receipt_sha256=sha(rp) if rp.exists() else None,
             resource_sha256=sha(resources), process_exit_sha256=sha(process),
             training_optimizer_updates_performed=0, new_sdk_tasks=0)
with ep.open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
print(json.dumps(proof), flush=True)
assert proof['supervision_passed'] and rec.get('output_writers_closed')
if a.mode == 'witness':
    from ldm100k_generation_acceptance import validate_witness
    validate_witness(rec, a.run, s)
else:
    assert rec.get('status') == 'USER_LDM100K_DIAGNOSTIC32_GENERATION_FINISHED'
    assert rec['unattempted_pairs'] == 0
