"""Diagnose frozen validation outcomes; does not change any metric or gate."""
import json
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/contact_com_validation_v24_20261012'
OUT = BASE / 'quality_diagnosis.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary=json.loads((BASE/'validation_summary.json').read_text())
    assert summary['attempts']==432
    thresholds=json.loads((BASE/'frozen_thresholds.json').read_text())['thresholds']
    hashes={str((BASE/name).relative_to(ROOT)):sha(BASE/name) for name in ('validation_summary.json','frozen_thresholds.json')}
    entries=[]
    all_rows=[]
    for entry in summary['entries']:
        folder=BASE/'raw'/('val_%s_s%d'%(entry['mode'],entry['seed']))
        file=folder/'metrics.json'
        rows=json.loads(file.read_text())
        hashes[str(file.relative_to(ROOT))]=sha(file)
        assert len(rows)==72
        all_rows.extend(rows)
        for family in ('jump','dance'):
            group=[row for row in rows if row['family']==family]
            distributions={}
            for key in ('endpoint_error_m','heading_error_rad','endpoint_time_error_s',
                        'surface_slip_proxy_fraction','surface_slip_proxy_mean_m_s','jerk_p95_m_s3'):
                values=[row[key] for row in group if row.get(key) is not None]
                distributions[key]=dict(count=len(values), min=float(np.min(values)), median=float(np.median(values)), max=float(np.max(values))) if values else None
            entries.append(dict(seed=entry['seed'],mode=entry['mode'],family=family,attempted=len(group),
                **{key:sum(bool(row.get(key,False)) for row in group) for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},
                slip_proxy_failed=sum(row.get('surface_slip_proxy_fraction') is None or row['surface_slip_proxy_fraction']>thresholds['sliding_fraction'] for row in group),
                penetration_failed=sum(row.get('penetration_fraction',1)>thresholds['penetration_fraction'] for row in group),
                continuity_failed=sum(row.get('jerk_p95_m_s3') is None or row['jerk_p95_m_s3']>thresholds['continuity_jerk_p95_m_s3'] for row in group),
                distributions=distributions))
    result=dict(attempted=len(all_rows), entries=entries, hashes=hashes,
        diagnostic_scope='Validation outcomes only. Surface-slip is a collision-primitive proxy, not true foot slip. Failure correlation does not prove which data or controller defect caused it.',
        thresholds_changed=False, inputs_changed=False, confirmed_innovations=0,
        condition_feedback_route='Stopped pending physical-quality diagnosis if all joint outcomes are zero',
        candidate_implication='C1 has an old exploratory condition-error signal; C2/C3 are untested. No candidate can be declared effective from baseline calibration. No G1/downstream stage has passed its entry gate.',
        overleaf_results_sync=False)
    serialized=json.dumps(result,indent=2,ensure_ascii=False)
    if OUT.exists():
        assert json.loads(OUT.read_text())==result, 'Existing diagnosis cannot be silently changed'
    else:
        OUT.write_text(serialized,encoding='utf8')
    print('COM_VALIDATION_QUALITY_DIAGNOSIS',len(all_rows),json.dumps({key:sum(e[key] for e in entries) for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass','slip_proxy_failed','penetration_failed','continuity_failed')}))


if __name__=='__main__':
    main()
