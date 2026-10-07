"""Write and deploy the evaluation interface resource/source lock first."""
import hashlib,json,py_compile
from r3_world_ssh_transport import ROOT,sync
paths=['scripts/r3_world_ppo_eval_isaac_20261007.py','scripts/r3_world_ppo_formal_stage_20261007.sh',
 'scripts/r3_world_ppo_eval_interface_20261007.sh','scripts/launch_world_ppo_eval_interface_20261007.py',
 'docs/world_ppo_eval_invocation_20261007.md','plans/world_residual_ppo_control_metrics_20261007.md',
 'scripts/r3_world_ppo_control_formal_v3_20261007.py','scripts/r3_world_ppo_formal_queue_20261007.py',
 'scripts/r3_world_ppo_gate_20261007.py']
for name in paths:
 p=ROOT/name
 if p.suffix=='.sh':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
locked=paths[:4]+['scripts/r3_world_ppo_core.py','scripts/r3_world_ppo_residual_core.py',
 'scripts/r3_world_ppo_training_core_20261007.py','scripts/r3_world_ppo_control_formal_v3_20261007.py']
data={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in locked}
(ROOT/'configs/world_ppo_eval_sources_20261007.json').write_text(json.dumps(data,indent=2)+'\n')
spec={'base_spec_canonical_sha256':'47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4',
 'warm_dependencies_unchanged':True,'GPU':0,'existing_viewer_GPU':1,'shared_context_allowance_MiB':1500,
 'mode':'diagnostic','arms':['P-history','W-correct'],'num_envs':64,'steps':120,'seed':42,
 'PPO_updates':0,'aux_updates':0,'normalizer_updates':0,'formal_scientific_scores':False,
 'source_hashes':data}
(ROOT/'configs/env-world-ppo-eval-interface-20261007.json').write_text(json.dumps(spec,indent=2)+'\n')
with (ROOT/'docs/dreamcontrol_ssh_environment_ledger.md').open('a',encoding='utf-8') as f:
 f.write('\n## 2026-10-07 frozen control-evaluation interface\n\nconfigs/env-world-ppo-eval-interface-20261007.json is a resource delta only, no packages/rebuild. New control metrics locked in plans/world_residual_ppo_control_metrics_20261007.md before final scores. Fresh invocation docs/world_ppo_eval_invocation_20261007.md tests64×120 frozen capacity checkpoints, 0PPO/aux/norm, separately from formal500×8192. Balanced initialreference event is an explicit evaluation adaptation. Native scene traces are not videos. Actual interface readiness pending independent documented invocation.\n')
print(sync(paths+['configs/world_ppo_eval_sources_20261007.json','configs/env-world-ppo-eval-interface-20261007.json','docs/dreamcontrol_ssh_environment_ledger.md']))
