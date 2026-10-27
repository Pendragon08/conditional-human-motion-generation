"""Prepare native300k LDM math with real later-stage full-sampling history.

No model is instantiated; real filtered VAE200k and GPU witnesses are pending.
"""
import pathlib,ast,json,hashlib
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
s=(B/'scripts/quick_ldm_runtime.py').read_text(encoding='utf-8')
s=s.replace('Pinned DAR math with a real frozen quick VAE and private accepted corpus.','Full300k DAR with frozen filtered VAE200k; native later-stage DDPM history.')
s=s.replace('from quick_vae_runtime import Platform,restore_rng,batch_sha','from generator8278_vae_runtime import Platform,restore_rng,batch_sha')
s=s.replace('from robotmdar.train.manager import DARManager','from robotmdar.train.manager import DARManager\n  from full_ldm_history import make_paper_manager,full_history_candidate\n  DARManager=make_paper_manager(DARManager)\n  self.full_history_candidate=full_history_candidate;self.full_history_calls=0')
s=s.replace('stages=[1000,1000,1000]','stages=[50000,50000,50000,150000]')
s=s.replace('use_full_sample=False','use_full_sample=True')
s=s.replace('save_every=1000,eval_every=500,eval_steps=12','save_every=100000,eval_every=2000,eval_steps=12')
s=s.replace("ckpt=OmegaConf.create({'vae':spec['vae_checkpoint']['path'],'dar':None})","ckpt=OmegaConf.create({'vae':spec['vae_inference']['path'],'dar':None})")
s=s.replace('m.max_steps==3000 and not m.should_use_full_sample()','m.max_steps==300000 and m.use_full_sample and not m.should_use_full_sample()')
s=s.replace("previous=torch.cat([history,pred],dim=1).detach()", "use_full=(not is_eval) and m.should_use_full_sample()\n   previous=self.full_history_candidate(self,history,pred,text,tuple(x0.shape),use_full)\n   if use_full:self.full_history_calls+=1\n   m.extra['full_sampling_history_calls']=self.full_history_calls")
s=s.replace("'gradients_finite':finite,'actual_optimizer_updates'", "'full_sampling_history_generated':use_full,'full_sampling_history_calls':self.full_history_calls,\n    'rollout_history_selected':bool(getattr(m,'last_rollout_selected',False)) if not is_eval else False,\n    'gradients_finite':finite,'actual_optimizer_updates'")
s=s.replace("'QUICK_LDM_REAL_GRADIENT_MEMORY_RESUME_WITNESS_PASSED'","'FILTERED15000_FULL_LDM_MEMORY_RESUME_WITNESS_PASSED'")
# Add a real five-step history candidate probe, with no fictitious training clock.
needle=' def close(self):'
probe=''' def full_history_witness(self):
  m=self.manager;outer=rng_state(self.generators);old_idx=m.stage_idx;old_calls=self.full_history_calls
  before_step=m.step;before_clock=self.clock.state();before_vae=module_sha(self.vae);before_model=module_sha(self.model)
  mode=self.model.training;self.model.eval();self.vae.eval();calls=[0];original=self.model.forward
  def count_forward(*args,**kwargs):calls[0]+=1;return original(*args,**kwargs)
  self.model.forward=count_forward
  try:
   # This is a gate probe, explicitly not50k/100k completed optimization.
   m.stage_idx=0;assert not m.should_use_full_sample()
   m.stage_idx=1;assert m.should_use_full_sample()
   features,text=self.batch('val')[0];history=features[:12,:2].cuda();text=text[:12].cuda();fallback=torch.zeros_like(features[:12,2:]).cuda()
   result=self.full_history_candidate(self,history,fallback,text,(12,1,128),m.should_use_full_sample())
   assert result.shape==(12,10,69) and torch.isfinite(result).all() and torch.equal(result[:,:2],history)
   assert calls[0]==5 and not result.requires_grad
   assert m.rollout_probability()==0.0 # actual clock remains4, not a later step
  finally:
   self.model.forward=original;m.stage_idx=old_idx;self.full_history_calls=old_calls;self.model.train(mode);restore_rng(outer,self.generators)
  assert m.step==before_step and self.clock.state()==before_clock and equal(rng_state(self.generators),outer)
  assert module_sha(self.vae)==before_vae and module_sha(self.model)==before_model
  return dict(status='REAL_FULL_DDPM_HISTORY_HELPER_PROBE_PASSED',simulated_later_stage_gate=True,
   actual_training_step_unchanged=before_step,actual_optimizer_updates_unchanged=self.clock.updates,
   actual_denoiser_forward_calls=calls[0],sample_batch=12,history_shape=[12,10,69],finite=True,
   model_and_frozen_vae_unchanged=True,rng_restored=True,formal_later_stage_training_reached=False)
'''
s=s.replace(needle,probe+needle)
p=B/'scripts/filtered15000_ldm_runtime.py';assert not p.exists();ast.parse(s);p.write_text(s,encoding='utf-8')
spec=dict(schema='textop.filtered15000_full_ldm_plan.v1',witness_authorized=False,train_authorized=False,
 runtime_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),requires_real_filtered_vae_200000=True,
 requires_actual_filtered_corpus_and_restore=True,actual_gpu_witnesses_completed=False,actual_full_stage_probe_completed=False,
 stages=[50000,50000,50000,150000],total_primitive_steps=300000,batch_size=512,history=2,future=8,primitives=4,
 use_full_sample=True,maximum_rollout_probability=.8,release_native_rollout_cap=1.,
 posterior_variance='FIXED_LARGE',release_default_posterior_variance='FIXED_SMALL',
 planned_checkpoint_steps=[100000,200000,300000],
 vae_cache='Future CPU-verified inference-only VAE200k export; complete original state preserved separately',
 differences_disclosed=['Paper cap.8 vs native manager cap1','Full history sampling enabled vs release default false',
 'Disabled static pose stays disabled, including native stage>=2 edge case','FIXED_LARGE vs release default FIXED_SMALL',
 '100k checkpoint cadence for bounded persistent storage, full300k optimization clock unchanged'],
 remaining='Complete post-VAE receipt/CPU export/capacity/entry/supervisor binding, then real author and fresh full-history gradient/restore witnesses')
p=B/'configs/filtered15000-ldm-plan.json';assert not p.exists();p.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'DISABLED_NATIVE300K_FULL_HISTORY_RUNTIME_PREPARED','new_gpu_runs':0,'trained_models_loaded':0}))
