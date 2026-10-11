"""Conservative charge ledger for this physics-guided experiment, not old jobs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
prefixes=('physics_guided_20261011','reset_alignment','condition_feedback_probe','contact_sampling','contact_sensor','contact_force_fixture','contact_cpu_export','contact_cpu_validation')
directories=[p for p in (ROOT/'results').iterdir() if p.is_dir() and p.name.startswith(prefixes) and p.name.endswith(('20261011','20261012'))]
rows=[]
for directory in sorted(directories):
    ledger=directory/'budget_ledger.json'
    if ledger.exists():
        for row in json.loads(ledger.read_text()): rows.append(dict(directory=directory.name,**row))
stages={stage:sum(float(r['seconds']) for r in rows if r['stage']==stage)/3600 for stage in sorted({r['stage'] for r in rows})}
result=dict(charged_wall_hours_one_gpu_per_job=sum(float(r['seconds']) for r in rows)/3600,
    stage_hours=stages,attempt_ledger_entries=len(rows),failure_entries=sum(r.get('exit_code',0)!=0 for r in rows),
    limits_hours=dict(total=72,measurement=4,comparison=12,ablation_g1=12,downstream=36,reserve=8),
    note='Includes original conservative reserved debit; CPU fixture wall times also charged. Running attempt omitted until its terminal ledger entry exists.',
    output_bytes=sum(p.stat().st_size for d in directories for p in d.rglob('*') if p.is_file()),
    directories=[str(d.relative_to(ROOT)) for d in sorted(directories)],entries=rows)
out=ROOT/'results/contact_measurement_review_20261011/budget_snapshot_20261012.json'
out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('entries','directories')},indent=2))
