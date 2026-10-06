"""Post-fit censoring/coverage audit; never train or modify the fitted results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/feedback_world_model_20261006'


def main():
    c = json.loads((ROOT/'configs/feedback_world_model_pilot_20261006.json').read_text())
    collection = json.loads((OUT/'collection.json').read_text())
    rows = json.loads((OUT/'reference_metrics.json').read_text())
    gate = json.loads((OUT/'predictive_gate.json').read_text())
    lo, hi = c['reference_splits']['test']
    expected = {(task, ref, cond) for task in c['tasks'] for ref in range(lo,hi)
                for cond in c['collection_conditions']}
    coverage = {}
    for horizon in c['evaluation_horizons']:
        observed = {(r['task'],r['reference'],r['condition']) for r in rows
                    if r['model']=='mlp_multi5_seed42' and r['horizon_steps']==horizon
                    and r['phase']=='nominal_and_dither' and r['windows']>0}
        coverage[str(horizon)] = {'expected_reference_condition_trials':len(expected),
            'observed_reference_condition_trials':len(observed),
            'zero_window_trials':[list(x) for x in sorted(expected-observed)]}
    durations = []
    for task in c['tasks']:
        for condition in c['collection_conditions']+[c['heldout_condition']]:
            records = [r for r in collection['records'] if r['split']=='test'
                       and r['task']==task and r['condition']==condition]
            durations.append({'task':task,'condition':condition,'trials':len(records),
                'fallen_trials':sum(r['fallen'] for r in records),
                'minimum_observed_duration_s':min(r['duration_s'] for r in records),
                'mean_observed_duration_s':sum(r['duration_s'] for r in records)/len(records),
                'meaning':'forecast windows terminate at fall; post-fall behavior is unobserved, not successful'})
    complete = not coverage['10']['zero_window_trials']
    report = {'status':'COMPLETE','horizon_coverage':coverage,'trial_duration_and_fall_records':durations,
        'primary_H10_coverage_complete':complete,
        'predictive_gate_after_coverage_audit':gate['status'] if complete else 'INCOMPLETE',
        'error_definition':'RMSE per Cartesian coordinate over 13 body sites (metres); Euclidean site RMSE is sqrt(3) times these values; all relative comparisons unchanged',
        'censoring_limitation':'Only observed pre-fall windows scored. Low forecast error does not mean low fall rate or successful recovery.',
        'full_innovation_validated':False}
    (OUT/'coverage_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COVERAGE_AUDIT_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='trial_duration_and_fall_records'}))


if __name__=='__main__':
    main()
