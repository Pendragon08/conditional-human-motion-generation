"""Account for completed authorized round, including public export duplicates.

Read-only with respect to experiments; writes a timestamped resource receipt.
Does not change bound V28/V33 budget helpers or their historical snapshots.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

from physics_guided_round_budget_v34_20261012 import snapshot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/round_resource_status_20261012'
caps = dict(measurement=4, comparison=12, ablation_g1=12, downstream=36, reserve=8)
result = snapshot(ROOT, ROOT / 'results/nonexistent_exclude_path')
directories = [ROOT / name for name in result['directories']]
for name in ('original_reference_attribution_v35_20261012', 'round_resource_status_20261012',
             'mocap_cache_binding_v36_20261012', 'endpoint_event_diagnostic_v37_20261012',
             'repair_budget_core_v38_20261012', 'execution_lineage_guard_v39_20261012',
             'construction_replay_transfer_v40_20261012', 'raw_replay_state_diagnostic_v41_20261012'):
    p = ROOT / 'results' / name
    if p.exists() and p not in directories:
        directories.append(p)
archives = [p for p in (ROOT / 'results').glob('*.tar.gz') if p.name.endswith('20261011.tar.gz') and
            p.name.startswith(('contact_', 'physics_guided_', 'reset_alignment_', 'reset_cache_'))]
result['output_bytes'] = sum(p.stat().st_size for d in directories for p in d.rglob('*') if p.is_file()) + sum(p.stat().st_size for p in archives)
sidecars = [ROOT / 'results/endpoint_event_diagnostic_v37_attempt1_failure.json']
result['output_bytes'] += sum(path.stat().st_size for path in sidecars if path.exists())
result['included_sidecars'] = [str(path.relative_to(ROOT)) for path in sidecars if path.exists()]
result['directories'] = [str(p.relative_to(ROOT)) for p in sorted(directories)]
result['archive_bytes'] = sum(p.stat().st_size for p in archives)
result['included_archives'] = [str(p.relative_to(ROOT)) for p in archives]
result['stage_caps_hours'] = caps
result['total_cap_hours'] = 72
result['output_cap_bytes'] = 15 * 1024 ** 3
result['data_free_bytes'] = shutil.disk_usage('/data').free
result['data_free_floor_bytes'] = 20 * 1024 ** 3
result['cap_checks'] = dict(total=result['charged_wall_hours_one_gpu_per_job'] <= 72,
    output=result['output_bytes'] < 15 * 1024 ** 3, data_free=result['data_free_bytes'] >= 20 * 1024 ** 3,
    **{stage: result['stage_hours'][stage] <= cap for stage, cap in caps.items()})
result['status'] = 'CAPS_PASS' if all(result['cap_checks'].values()) else 'CAPS_FAILED'
result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result['time'] = time.time()
result['gpu_jobs_launched_by_this_script'] = 0
result['note'] += ' Export archives and original-reference attribution also counted; current-resource receipt is not experiment efficacy.'
result['note'] += ' V36 cache certificate, V37 event diagnostic and V38 numeric-only budget core are also counted.'
OUT.mkdir(exist_ok=True)
destination = OUT / ('snapshot_%d.json' % time.time_ns())
destination.write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps({k: result[k] for k in ('status', 'charged_wall_hours_one_gpu_per_job', 'stage_hours', 'failed_entries', 'output_bytes', 'archive_bytes', 'data_free_bytes', 'cap_checks')}, indent=2))
print('RESOURCE_RECEIPT', destination)
