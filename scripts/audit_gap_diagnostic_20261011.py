"""Summarize frozen diagnostic without redefining acceptance thresholds."""
import csv
import json
from collections import Counter
from pathlib import Path

base = Path(__file__).resolve().parents[1] / 'results/gap_diagnostic_20261011/v3'
analysis = json.loads((base / 'analysis.json').read_text(encoding='utf-8'))
with (base / 'per_clip.csv').open(encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
with (base / 'threshold_sensitivity.csv').open(encoding='utf-8-sig') as f:
    sensitivity = list(csv.DictReader(f))
assert len(rows) == 192
rejected = sorted([r for r in rows if r['terminated'] == 'False' and r.get('tracking_gate') == 'False'],
                  key=lambda r: float(r['mean_gmpjpe_mm']), reverse=True)
assert len(rejected) == 15
accepted = Counter((r['force_n'], r['speed_mps'], r['mean_error_mm'], r['tail_error_mm'])
                   for r in sensitivity if r['pass_gate'] == 'True')
counts = list(accepted.values())
result = dict(
    interpretation='Descriptive diagnosis; not a new method or robot feasibility guarantee',
    h1=dict(status='supported_for_predeclared_operational_tracking_gate',
            finding='15 of 174 non-terminated first attempts fail the frozen mean/tail tracking gate',
            rejected_non_terminated=rejected,
            caution='Neither tracking failure nor this gate independently diagnoses a defective reference'),
    h2=dict(status='tracking_worsening_not_supported',
            finding='Reference boundary speed is higher, but boundary tracking error is lower by 34.64 mm',
            caution='Replanning uses executed-state conditioning and re-anchors predictions. Reference plan updates are not physical body discontinuities. No causal claim.'),
    contact_proxy=dict(status='needs_surface_geometry_or_manual_confirmation',
                      main_gate_count=85, sensitivity_min=min(counts), sensitivity_max=max(counts),
                      caution='Rigid-body center movement may be ankle/toe rotation; no true slip labels'),
    scope=dict(character='SMPL physically simulated character', tasks='HumanML seeded T2M, including ordinary and complex prompts',
               absent=['explicit spatial target retention', 'target robot actuator certification', 'second controller attribution', 'downstream retraining benefit']),
    next_experiment='Freeze matched prompts and target/phase conditions; compare raw and generic physical refinement before proposing a new correction method. Separate missing-scene prompts from support-compatible motions.')
(base / 'hypothesis_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k,v in result.items() if k!='h1'}, ensure_ascii=False, indent=2))
