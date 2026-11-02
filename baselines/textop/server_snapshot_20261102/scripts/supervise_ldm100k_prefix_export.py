"""Bounded CPU-only export, serialized with all SDK/model jobs by the common lock."""
import argparse
import fcntl
import json
import os
import shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from ldm100k_prefix_completion import ROOT, read, sha, owned

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
a = p.parse_args()
assert os.getuid() == 1013 and a.run.isascii() and a.run.isidentifier()
sp = ROOT / 'configs/ldm100k-prefix-export-spec.json'
s = read(sp)
assert s['schema'] == 'textop.ldm100k_prefix_cpu_export.v1' and s['export_authorized']
assert sha(__file__) == s['supervisor_sha256']
assert sha(ROOT / 'scripts/export_ldm100k_prefix.py') == s['executor_sha256']
for n, h in s['bound_sources'].items():
    assert sha(ROOT / n) == h, n
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert shutil.disk_usage(ROOT).free >= s['minimum_disk_free_bytes']
parent = ROOT / 'ldm100k_user_handoff_20261028'
assert parent.is_dir()
owned(parent)
out = parent / a.run
assert not out.exists()
prefix = 'ldm100k_prefix_export_' + a.run
ep = ROOT / 'receipts' / (prefix + '_exit.json')
assert not ep.exists()
cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4',
       'OPENBLAS_NUM_THREADS=4', 'PYTHONIOENCODING=utf-8', 'PYTHONUNBUFFERED=1',
       str(ROOT / 'env/bin/python'), str(ROOT / 'scripts/export_ldm100k_prefix.py'), '--run', a.run]
resources = ROOT / 'logs' / (prefix + '_resources.jsonl')
process = ROOT / 'receipts' / (prefix + '_process_exit.json')
proof = run_supervised(cmd, ROOT / 'logs' / (prefix + '.log'), resources, process,
                       lambda pid: sample_resources(ROOT, pid, s['minimum_ram_free_bytes'],
                                                    s['minimum_disk_free_bytes']),
                       s['timeout_seconds'], 15)
rp = out / 'receipt.json'
rec = read(rp) if rp.exists() else {}
proof.update(run=a.run, status=rec.get('status'), receipt_sha256=sha(rp) if rp.exists() else None,
             spec_sha256=sha(sp), source_sha256=sha(__file__), resource_sha256=sha(resources),
             process_exit_sha256=sha(process), new_gpu_contexts=0, new_sdk_tasks=0,
             training_optimizer_updates_performed=0)
with ep.open('x') as handle:
    json.dump(proof, handle, indent=2, allow_nan=False)
print(json.dumps(proof), flush=True)
assert proof['supervision_passed'] and rec.get('status') == 'USER_LDM100000_PREFIX_CPU_EXPORT_PASSED'
assert rec.get('output_writers_closed')
