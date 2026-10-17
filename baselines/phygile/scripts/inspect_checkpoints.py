"""Inspect only this run's trusted local checkpoints; verify finite model state."""
from pathlib import Path
import json, hashlib, shutil, torch
root=Path(__file__).resolve().parents[1]
runs=list((root/'logs/rsl_rl/phygile_moe_pilot').glob('*train20_nostate_seed42'));assert len(runs)==1
run=runs[0];out=root/'release_assets';out.mkdir(exist_ok=True)
records=[];states=[]
for i,updates in [(0,1),(499,500)]:
 p=run/f'model_{i}.pt'
 checkpoint=torch.load(p,map_location='cpu',weights_only=False)
 assert checkpoint['iter']==i
 state=checkpoint['model_state_dict'];states.append(state)
 assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
 name=f'phygile_initial_expert_{updates}updates_seed42.pt';shutil.copyfile(p,out/name)
 records.append({'file':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'iteration':i,'optimizer_updates':updates,'model_parameter_elements':sum(v.numel() for v in state.values()),'all_model_tensors_finite':True})
changed={}
for expert in range(12):
 keys=[k for k in states[0] if k.startswith(f'experts.{expert}.')]
 assert keys,expert
 changed[str(expert)]={'tensors':len(keys),'changed_tensors':sum(not torch.equal(states[0][k],states[1][k]) for k in keys)}
assert changed['0']['changed_tensors']>0
assert all(changed[str(i)]['changed_tensors']==0 for i in range(1,12))
cfg=root/'results/effective_configuration';cfg.mkdir(exist_ok=True)
for name in ['env.yaml','agent.yaml']:
 shutil.copyfile(run/'params'/name,cfg/name)
shutil.copyfile(run/'reference_validation.json',cfg/'reference_validation.json')
result={'checkpoints':records,'expert_parameter_changes':changed,'note':'Only the initial expert changed; not a complete curriculum MoE training run.'}
(root/'results/checkpoint_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
