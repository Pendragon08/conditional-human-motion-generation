"""Post hoc reviewer diagnostic; never alter the pre-registered oracle gate."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/feedback_world_model_20261006'


def main():
    s=json.loads((OUT/'oracle_summary.json').read_text())
    rows=s['records']
    tasks=['Kick','Punch']
    keys=[(t,i) for t in tasks for i in range(46,66)]
    assert len(rows)==200 and len({(r['task'],r['reference'],r['arm'],r['candidate_seed']) for r in rows})==200
    values={arm:np.asarray([np.mean([r['recovery_tracking_AUC_rad_s'] for r in rows
        if r['arm']==arm and (r['task'],r['reference'])==key]) for key in keys]) for arm in ['ppo','simple','oracle']}
    rng=np.random.default_rng(20261006)
    trials=[]
    for _ in range(2000):
        ids=np.concatenate([rng.choice(np.arange(t*20,(t+1)*20),20,replace=True) for t in range(2)])
        trials.append(1-float(values['oracle'][ids].mean()/values['simple'][ids].mean()))
    ci=np.quantile(trials,[.025,.975]).tolist()
    breakdown=[]
    for task in tasks:
        for arm in ['ppo','simple','oracle']:
            rs=[r for r in rows if r['task']==task and r['arm']==arm]
            breakdown.append({'task':task,'arm':arm,'trials':len(rs),
                'mean_recovery_AUC_rad_s':float(np.mean([r['recovery_tracking_AUC_rad_s'] for r in rs])),
                'virtual_goal_success_rate':float(np.mean([r['virtual_target_success'] for r in rs])),
                'fall_rate':float(np.mean([r['fallen'] for r in rs])),
                'mean_abs_action_change':float(np.mean([r['mean_abs_action_change'] for r in rs])),
                'mean_oracle_decision_wall_s':float(np.mean([r['mean_oracle_decision_wall_s'] for r in rs
                    if r['mean_oracle_decision_wall_s'] is not None])) if arm=='oracle' else None})
    diagnostic={'status':'COMPLETE','type':'post-hoc reviewer diagnostic, not replacement gate',
        'original_formal_gate_preserved':s['oracle_upper_bound_gate'],
        'relative_oracle_AUC_improvement_vs_simple':1-float(values['oracle'].mean()/values['simple'].mean()),
        'reference_bootstrap95CI_vs_simple':ci,'task_breakdown':breakdown,
        'all200_trials_present_and_unique':True,
        'candidate_seed_stochasticity_not_fully_covered_by_reference_bootstrap':True,
        'oracle_scope':'five-candidate fixed-cost simulator oracle; not an upper bound on all world-model control designs',
        'learned_W_allowed_next':s['oracle_upper_bound_gate']=='PASS' and ci[0]>0,
        'no_full_innovation_claim':True}
    (OUT/'oracle_reviewer_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print(json.dumps(diagnostic,indent=2))


if __name__=='__main__':
    main()
