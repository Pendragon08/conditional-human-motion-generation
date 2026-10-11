"""Snapshot the whole authorized round, including diagnostics between candidates.

Never rewrites the live V28 snapshot. Call before a new serial experiment;
its caller excludes its own fresh output so each task is charged only once.
"""
import json
import math
from pathlib import Path

PREFIXES=('physics_guided_20261011','reset_alignment','condition_feedback_probe',
    'contact_sampling','contact_sensor','contact_geometry_calibration','contact_force_fixture',
    'contact_cpu_export','contact_cpu_validation','contact_slip_diagnostic','contact_point_fixture',
    'contact_point_human','contact_point_analysis','velocity_reference_fixture',
    'articulated_contact_fixture','contact_com_validation','contact_com_runtime',
    'metric_runtime_diagnostic','fixed_reference_com','condition_feedback_formal',
    'condition_feedback_guard','condition_feedback_components','mocap_reference_control',
    'mocap_reference_assembly')


def snapshot(root,exclude):
    root=Path(root).resolve();exclude=Path(exclude).resolve()
    directories=[p for p in (root/'results').iterdir() if p.is_dir() and p.resolve()!=exclude
        and p.name.startswith(PREFIXES) and p.name.endswith(('20261011','20261012'))]
    records=[]
    for directory in sorted(directories):
        ledger=directory/'budget_ledger.json'
        if not ledger.exists():continue
        rows=json.loads(ledger.read_text(encoding='utf8'))
        assert isinstance(rows,list)
        for row in rows:
            assert isinstance(row.get('seconds'),(int,float)) and math.isfinite(row['seconds']) and row['seconds']>=0
            assert row['stage'] in ('measurement','comparison','ablation_g1','downstream','reserve')
            records.append(dict(directory=directory.name,**row))
    return dict(entries=records,directories=[str(p.relative_to(root)) for p in sorted(directories)],
        output_bytes=sum(p.stat().st_size for d in directories for p in d.rglob('*') if p.is_file()),
        charged_wall_hours_one_gpu_per_job=sum(r['seconds'] for r in records)/3600,
        stage_hours={stage:sum(r['seconds'] for r in records if r['stage']==stage)/3600
            for stage in ('measurement','comparison','ablation_g1','downstream','reserve')},
        failed_entries=sum(r.get('exit_code',0)!=0 for r in records),
        note='Current serial-round ledger snapshot, including intervening diagnostics; original conservative debits retained. No live ledger or prior snapshot rewritten.')
