import json,pathlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103';N=D/'native_universal_20261104'
def read(p):return json.loads(p.read_text())
gate=read(N/'all_four_reference_gate.json');sequence=read(Q/'sequence.json');progress=read(Q/'progress.json')
assert gate['verified_counts']=={'pick':500,'pour':0,'bimanual_box_lift':0,'box_push_pull':0}
assert not gate['training_allowed'] and sequence['policy']['wait_for_all2000_before_SuperMimic']
assert not sequence['policy']['wait_for_all2000_before_pick_ppo'] and sequence['SuperMimic_policy_mode']['initial_references']==2000
assert all(read(p)['wait_for_all_four_reference_libraries'] for p in [N/'status.json',D/'adaptation_status.json',D/'active_native_universal_run.json'])
assert next(t for t in progress['tasks'] if t['id']=='pick')['new_PPO_started']
ppo=read(R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1/status.json');assert ppo['status']=='ppo_running' and ppo['pid']==1436632
print(json.dumps({'status':'policy_readback_verified','counts':gate['verified_counts'],'SuperMimic_waits_for2000':True,'Pick_PPO_running':True,'Pick_PPO_updates':ppo['policy_updates']}))
