"""Locked reference-paired final control gate; no model selection or tuning."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
arms=['P0','P-cap','P-history','W-correct','W-shuffled-original']
out={'stage':'final_control_confirmation','bootstrap_unit':'reference','bootstrap_draws':10000,
 'rng_seed':20261007,'seeds':{},'innovation_validated':False,'explanation_arms_required_after_main_PASS':True}
for seed in [42,43,44]:
 summaries={arm:json.loads((a.root/f'eval_{arm}_seed{seed}_v1/summary.json').read_text()) for arm in arms}
 for arm,v in summaries.items():
  assert v['status']=='COMPLETE' and v['mode']=='formal' and v['checkpoint_completed_updates']==500
  assert v['PPO_updates']==v['aux_updates']==v['normalizer_updates']==0 and v['model_hashes_unchanged']
  assert v['seed']==seed and v['arm']==arm and v['num_envs']==2048 and v['steps']==500
  assert v['reference_manifest_sha256']==summaries['P0']['reference_manifest_sha256']
  assert [r['reference_file'] for r in v['rows']]==[r['reference_file'] for r in summaries['P0']['rows']]
 def metric(arm,name):return np.asarray([r[name] for r in summaries[arm]['rows']])
 draws=np.random.default_rng(20261007).integers(0,11,(10000,11))
 comparisons={};passed=True
 for control in ['P0','P-cap','P-history','W-shuffled-original']:
  base=metric(control,'recovery_joint_RMSE_AUC_rad_s');wm=metric('W-correct','recovery_joint_RMSE_AUC_rad_s')
  assert np.all(base>0) and np.isfinite(wm).all()
  gain=1-wm.mean()/base.mean();boot=1-wm[draws].mean(1)/base[draws].mean(1)
  ci=np.quantile(boot,[.025,.975]);criterion=bool(ci[0]>0 and (gain>=.1 if control!='W-shuffled-original' else True))
  row={'recovery_improvement':float(gain),'recovery_95CI':ci.tolist(),'recovery_gate':criterion}
  if control!='W-shuffled-original':
   cleanbase=metric(control,'clean_recovery_RMSE_rad');cleanwm=metric('W-correct','clean_recovery_RMSE_rad')
   cleanratio=cleanwm[draws].mean(1)/cleanbase[draws].mean(1);cci=np.quantile(cleanratio,[.025,.975])
   row.update({'clean_W_over_control':float(cleanwm.mean()/cleanbase.mean()),'clean_ratio_95CI':cci.tolist(),'clean_noninferiority_gate':bool(cci[1]<=1.05)})
   criterion &=row['clean_noninferiority_gate']
  passed &=criterion;comparisons[control]=row
 wm=summaries['W-correct'];base=summaries['P0']
 sd=wm['first_episode_macro_success']-base['first_episode_macro_success'];fd=wm['first_episode_macro_failure']-base['first_episode_macro_failure']
 def diff_ci(name):
  w=metric('W-correct',name);b=metric('P0',name)
  return np.quantile((w-b)[draws].mean(1),[.025,.975]).tolist()
 successgate=sd>=-.05;fallgate=fd<=0
 passed &=successgate and fallgate
 out['seeds'][str(seed)]={'passed':bool(passed),'comparisons':comparisons,
  'success_difference_vs_P0':sd,'success_difference_95CI':diff_ci('first_episode_success'),
  'failure_difference_vs_P0':fd,'failure_difference_95CI':diff_ci('first_episode_failure'),
  'success_point_gate':successgate,'failure_point_gate':fallgate}
out['main_control_gate_passed']=all(v['passed'] for v in out['seeds'].values())
out['status']='PASS_REQUIRES_EXPLANATION_ABLATIONS' if out['main_control_gate_passed'] else 'FAIL_STOP_CURRENT_CANDIDATE'
assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n')
print('WORLD_PPO_FINAL_CONTROL_GATE_COMPLETE',json.dumps(out),flush=True)
