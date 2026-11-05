"""Audit CPU preflight's saved output and exercise its invalid-asset path."""
import hashlib
import json
from pathlib import Path
from audit_vertical_support_preflight import audit_vertical_support

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cfg=json.loads((HERE/'config.json').read_text())
    manifest=json.loads(Path(cfg['manifest']).read_text())
    lines=(HERE/'preflight.log').read_text().splitlines()
    payload=json.loads(next(line for line in lines if line.startswith('{')))
    vertical=payload['vertical_support_diagnostic']
    prior=json.loads((ROOT/'vertical_support_bound_20261101'/'report.json').read_text())
    prior_cases={x['sample_id']:x for x in prior['cases']}
    assert payload['manifest_sha256']==sha(Path(cfg['manifest']))
    assert payload['feature_audit']['training_ready'] is False and vertical['training_ready'] is False
    assert vertical['scene_sha256']==prior['scene_sha256'] and len(vertical['cases'])==4
    for case in vertical['cases']:
        old=prior_cases[case['sample_id']]
        assert case['state_sha256']==old['state_sha256'] and case['frames']==old['frames']
        assert case['negative_beyond_1e_6']==old['below_minus_1e-6_intervals']
        assert abs(case['minimum_n_per_kg']-old['min_required_vertical_support_n_per_kg'])<1e-9
    bad=dict(cfg['vertical_support_diagnostic']);bad['asset_report_sha256']='0'*64
    try:audit_vertical_support(manifest['records'],**bad)
    except ValueError as exc:
        assert str(exc)=='asset report hash mismatch'
    else:raise AssertionError('tampered asset hash accepted')
    rejection=(HERE/'run_rejected.log').read_text()
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in rejection
    assert 'Ran 3 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    out={'preflight_log_sha256':sha(HERE/'preflight.log'),'config_sha256':sha(HERE/'config.json'),
         'diagnostic_module_sha256':sha(HERE/'audit_vertical_support_preflight.py'),
         'trainer_sha256':sha(HERE/'train_robot_prior.py'),'previous_bound_report_sha256':sha(ROOT/'vertical_support_bound_20261101'/'report.json'),
         'run_rejected_log_sha256':sha(HERE/'run_rejected.log'),'cases':vertical['cases'],
         'invalid_asset_hash_rejected':True,'training_run_rejected':True,'training_ready':False,
         'limitations':'Same MuJoCo asset and sample states as earlier analysis; bound is a necessary condition only.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))

if __name__=='__main__':main()
