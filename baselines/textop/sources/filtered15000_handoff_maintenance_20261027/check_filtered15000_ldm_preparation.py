import pathlib,json,hashlib,ast,sys
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'scripts'));from full_ldm_history import rollout_probability
stages=[50000,50000,50000,150000]
cases=[(0,-1,0.),(4,0,0.),(50000,0,0.),(50001,1,.00002),(70000,1,.4),(90000,1,.8),(100000,1,.8),(150000,2,.8),(299999,3,.8)]
for step,stage,expected in cases:assert abs(rollout_probability(step,stage,stages)-expected)<1e-12
assert rollout_probability(299999,3,stages,False)==0.
paths=['scripts/filtered15000_ldm_runtime.py','scripts/full_ldm_history.py','provider_tools/activate_filtered15000_corpus.py']
for n in paths:ast.parse((B/n).read_text(encoding='utf-8'))
s=(B/'scripts/filtered15000_ldm_runtime.py').read_text(encoding='utf-8')
assert 'use_full_sample=True' in s and 'stages=[50000,50000,50000,150000]' in s and 'self.full_history_candidate(' in s
assert 'm.should_use_full_sample()' in s and 'm.max_steps==300000' in s
assert 'ModelVarType.FIXED_LARGE' in s and "spec['vae_inference']['path']" in s and 'restore_auxiliary_counters' in s
planpath=B/'configs/filtered15000-ldm-plan.json';plan=json.loads(planpath.read_text(encoding='utf-8'))
assert not plan['witness_authorized'] and not plan['train_authorized'];plan['runtime_source_sha256']=hashlib.sha256((B/paths[0]).read_bytes()).hexdigest()
plan['history_helper_sha256']=hashlib.sha256((B/paths[1]).read_bytes()).hexdigest();planpath.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
out=B/'results/filtered15000_handoff_tools_20261027/ldm_preparation_cpu_v1.json';assert not out.exists()
proof=dict(status='DISABLED_FULL_LDM_SCHEDULE_AND_HISTORY_SOURCE_CHECKED',rollout_probability_cases=len(cases)+1,
 native_total_steps=300000,full_sampling_history_branch_present=True,actual_later_stage_gpu_probe_completed=False,
 actual_gradients_or_sampling_executed=False,actual_models_loaded=0,new_gpu_contexts=0,
 source_sha256={n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in paths},
 remaining='Full entry/supervisor and real VAE200k inference/export/data/storage bindings plus actual author/fresh witness pending')
out.write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
