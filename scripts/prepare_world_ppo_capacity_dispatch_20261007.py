"""Source lock and declarative resource delta, before GPU invocation."""
import hashlib,json,py_compile
from r3_world_ssh_transport import ROOT,sync
paths=['scripts/r3_world_ppo_invariant_witness_20261007.py',
 'scripts/r3_world_ppo_control_capacity_v2_20261007.py','scripts/r3_world_ppo_capacity_20261007.sh',
 'scripts/launch_world_ppo_capacity_20261007.py','docs/world_ppo_capacity_invocation_20261007.md']
for name in paths:
 p=ROOT/name
 if p.suffix=='.sh':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
locked=paths[:3]+['scripts/r3_world_ppo_core.py','scripts/r3_world_ppo_residual_core.py',
 'scripts/r3_world_ppo_training_core_20261007.py']
data={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in locked}
(ROOT/'configs/world_ppo_capacity_sources_20261007.json').write_text(json.dumps(data,indent=2)+'\n')
spec={'base_spec_canonical_sha256':'47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4',
 'dependencies':'unchanged warm reuse','GPU':0,'viewer_GPU':1,'shared_context_allowance_MiB':1500,
 'fixture':{'CUDA':True,'PPO_updates':0,'simulation_steps':0,'seed':20261007},
 'capacity':{'arms':['P-history','W-correct'],'envs':8192,'PPO_diagnostic_updates':2,
  'steps_per_rollout':24,'seed':42,'W_aux_per_update':1,'aux_batch':1024},
 'formal_start':False,'source_hashes':data}
(ROOT/'configs/env-world-ppo-capacity-20261007.json').write_text(json.dumps(spec,indent=2)+'\n')
with (ROOT/'docs/dreamcontrol_ssh_environment_ledger.md').open('a',encoding='utf-8') as f:
 f.write('\n## 2026-10-07 full-size capacity and saved CUDA fixtures\n\n')
 f.write('User explicitly confirmed6013/live actual moving robot. New gate records direct human answer; main page failure preserved. Warm dependency spec unchanged; configs/env-world-ppo-capacity-20261007.json specifies zero-PPO/zero-sim CUDA tensors and two8192-env×2-update capacity stages. Preserves v1 five smokes, no formal auto-launch. Fresh doc invocation docs/world_ppo_capacity_invocation_20261007.md; all capacity readiness remains pending actual invocation and saved-tensor audit.\n')
sync(paths+['configs/world_ppo_capacity_sources_20261007.json','configs/env-world-ppo-capacity-20261007.json',
 'docs/dreamcontrol_ssh_environment_ledger.md'])
print('CAPACITY_FILES_DEPLOYED',json.dumps(data))
