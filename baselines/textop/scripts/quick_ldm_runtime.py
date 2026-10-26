"""Pinned DAR math with a real frozen quick VAE and private accepted corpus."""
import hashlib,io,pathlib,random
import numpy as np
import torch
from generator_training_state import PrimitiveClock,equal,module_sha,rng_state
from quick_vae_runtime import Platform,restore_rng,batch_sha

class Runtime:
 def __init__(self,spec,base,out):
  from paper69_runtime import install_runtime,DifferentiableG1
  from paper69_data_runtime import install_data_sampling
  from omegaconf import OmegaConf
  random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);torch.set_num_threads(4)
  torch.cuda.set_per_process_memory_fraction(.65);assert torch.cuda.mem_get_info()[0]>20*1024**3
  install_runtime(base['robot_xml']);install_data_sampling()
  from robotmdar.dataloader.data import SkeletonPrimitiveDataset
  from robotmdar.model.mld_vae import AutoMldVae
  from robotmdar.model.mld_denoiser import DenoiserTransformer
  from robotmdar.train.manager import DARManager
  from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion,ModelMeanType,ModelVarType,LossType,get_named_beta_schedule
  from robotmdar.diffusion.resample import UniformSampler
  xml=pathlib.Path(base['robot_xml']);robot=OmegaConf.create({'asset':{'assetRoot':str(xml.parent),'assetFileName':xml.name}})
  self.datasets={}
  for split in ['train','val']:
   ds=SkeletonPrimitiveDataset(robot,512,69,2,8,4,spec['corpus_dir'],str(pathlib.Path(spec['corpus_dir'])/'action_statistics.json'),weighted_sample=split=='train',split=split)
   ds.skeleton=DifferentiableG1(str(xml),device='cuda');self.datasets[split]=ds
  self.generators={k:torch.Generator().manual_seed(spec['seed']+i) for i,k in enumerate(['train','val'])}
  self.vae=AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).cuda()
  self.model=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,clip_dim=512,history_shape=(2,69),noise_shape=(1,128),cond_mask_prob=.1).cuda()
  optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
  weights={'rec':1.,'kl':1e-4,'body_trans':.05,'body_rot':.01,'dof_pos':.03,'dof_vel':1e-5,'foot_contact':.01,'latent_rec':1.,'smooth':0.,'quantize_rot':0.,'quantize_trans':0.,'drift_xy':0.,'drift_yaw':0.}
  self.platform=Platform(out/'metrics.jsonl');self.requested_saves=[]
  m=DARManager(stages=[1000,1000,1000],use_rollout=True,use_static_pose=False,static_prob=.001,static_perturbation_scale=0.,use_full_sample=False,
   save_every=1000,eval_every=500,eval_steps=12,device='cuda',save_dir=spec['checkpoint_dir'],learning_rate=1e-4,anneal_lr=True,max_grad_norm=1.,use_ema=False,
   loss_weight=weights,ckpt=OmegaConf.create({'vae':spec['vae_checkpoint']['path'],'dar':None}),platform=self.platform)
  # The official manager caches its VAE beside DAR checkpoints; the HOME budget
  # explicitly includes this one full cross-filesystem copy, shared by all runs.
  m.hold_model(self.vae,self.model,optimizer,self.datasets['train'])
  m.save_model=lambda:self.requested_saves.append(m.step)
  assert not m.ema_models and m.max_steps==3000 and not m.should_use_full_sample()
  assert all(not p.requires_grad for p in self.vae.parameters());self.frozen_vae_sha=module_sha(self.vae)
  self.manager=m;self.clock=PrimitiveClock(m);self.consecutive_skips=0
  self.diffusion=GaussianDiffusion(betas=get_named_beta_schedule('cosine',5),model_mean_type=ModelMeanType.START_X,model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE,rescale_timesteps=False)
  self.sampler=UniformSampler(self.diffusion)
 def batch(self,split):
  b=self.datasets[split]._generate_batch_optimized(generator=self.generators[split])
  assert len(b)==4 and all(x.shape==(512,10,69) and y.shape==(512,512) for x,y in b);return b
 def run_batch(self,batch,is_eval=False):
  m=self.manager;self.model.train(not is_eval);self.vae.eval();previous=None;records=[]
  for features,text in batch:
   m.pre_step(is_eval=is_eval);features=features.cuda();text=text.cuda();future=features[:,2:]
   history=features[:,:2] if is_eval else m.choose_history(features[:,:2],previous,2)
   with torch.set_grad_enabled(not is_eval):
    t,importance=self.sampler.sample(512,device='cuda')
    latent,_=self.vae.encode(future_motion=future,history_motion=history);x0=latent.permute(1,0,2)
    noisy=self.diffusion.q_sample(x0,t,noise=torch.randn_like(x0))
    predlatent=self.model(noisy,self.diffusion._scale_timesteps(t),y={'history_motion_normalized':history,'text_embedding':text}).permute(1,0,2)
    pred=self.vae.decode(predlatent,history,nfuture=8)
    terms,extras=m.calc_loss(future,pred,latent,None,predlatent,importance,history_motion=history)
    assert all(torch.isfinite(v).all() for v in terms.values()),'Nonfinite loss; retain failure instead of clipping'
    finite=True
    if not is_eval:
     m.optimizer.zero_grad();terms['total'].backward();grads=[p.grad for p in self.model.parameters() if p.grad is not None];assert grads
     finite=bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
     if finite:m.grad_clip(self.model);m.optimizer.step();self.consecutive_skips=0
     else:self.consecutive_skips+=1
     assert all(p.grad is None and not p.requires_grad for p in self.vae.parameters())
   previous=torch.cat([history,pred],dim=1).detach()
   m.post_step(is_eval=is_eval,loss_dict={k:v.detach().cpu() for k,v in terms.items()},extras={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in extras.items()})
   if not is_eval:self.clock.after_post_step()
   records.append({'manager_step':m.step,'loss':float(terms['total']),'reconstruction_huber':float(terms['rec']),'latent_reconstruction_huber':float(terms['latent_rec']),
    'gradients_finite':finite,'actual_optimizer_updates':self.clock.updates,'skipped_updates':self.clock.skipped,'is_evaluation':is_eval})
   assert self.consecutive_skips<20,'Twenty consecutive official gradient skips'
  return records
 def sample_quality(self,batch):
  outer=rng_state(self.generators);torch.manual_seed(20261126);torch.cuda.manual_seed_all(20261126)
  mode=self.model.training;self.model.eval();self.vae.eval();previous=None;losses=[]
  with torch.no_grad():
   for features,text in batch:
    features=features[:12].cuda();text=text[:12].cuda();history=features[:,:2] if previous is None else previous[:,-2:]
    def guided(x,t,y):
     un=self.model(x,t.long(),y={**y,'uncond':True});yes=self.model(x,t.long(),y=y);return un+5*(yes-un)
    sample=self.diffusion.p_sample_loop(guided,(12,1,128),device='cuda',clip_denoised=False,model_kwargs={'y':{'text_embedding':text,'history_motion_normalized':history}},progress=False)
    pred=self.vae.decode(sample.permute(1,0,2),history,nfuture=8);assert torch.isfinite(pred).all()
    losses.append(float(torch.nn.functional.huber_loss(pred,features[:,2:])));previous=pred
  self.model.train(mode);restore_rng(outer,self.generators);assert equal(rng_state(self.generators),outer)
  return {'normalized_full_sampling_huber':sum(losses)/len(losses),'batch':12,'primitives':4,'diffusion_steps':5,'cfg':5,
   'posterior_future_not_used':True,'generated_history_after_first_primitive':True,'text_semantics_or_physical_execution_verified':False}
 def memory_witness(self):
  m=self.manager;records=self.run_batch(self.batch('train'));assert self.clock.updates==4 and not self.clock.skipped
  m._to_eval_steps=4
  state={'model':self.model.state_dict(),'optimizer':m.optimizer.state_dict(),'rng':rng_state(self.generators),'clock':self.clock.state(),'step':m.step,'pending_eval':m._to_eval_steps}
  memory=io.BytesIO();torch.save(state,memory);payload=memory.getvalue();saved=torch.load(io.BytesIO(payload),map_location='cpu')
  expected_batch=batch_sha(self.batch('train'));before=module_sha(self.model)
  with torch.no_grad():next(self.model.parameters()).add_(.01)
  self.model.load_state_dict(saved['model']);m.optimizer.load_state_dict(saved['optimizer']);m.step=saved['step'];self.clock.load(saved['clock']);m._to_eval_steps=saved['pending_eval'];restore_rng(saved['rng'],self.generators)
  assert module_sha(self.model)==before and equal(self.model.state_dict(),saved['model']) and equal(m.optimizer.state_dict(),saved['optimizer']) and equal(rng_state(self.generators),saved['rng'])
  assert batch_sha(self.batch('train'))==expected_batch;restore_rng(saved['rng'],self.generators)
  evaluation=self.run_batch(self.batch('val'),is_eval=True);assert self.clock.updates==4 and m.step==4 and not m.should_eval()
  assert module_sha(self.vae)==self.frozen_vae_sha
  return {'status':'QUICK_LDM_REAL_GRADIENT_MEMORY_RESUME_WITNESS_PASSED','actual_optimizer_updates':4,'primitive_attempts':4,'all_four_training_gradients_finite':True,
   'validation_optimizer_updates':0,'model_and_adam_roundtrip_exact':True,'all_rngs_restored_exactly':True,'next_batch_replayed_exactly':True,'pending_evaluation_completed_without_updates':True,
   'vae_frozen_and_unchanged':True,'ema_enabled':False,'memory_serialized_bytes':len(payload),'memory_serialized_sha256':hashlib.sha256(payload).hexdigest(),
   'persistent_diagnostic_checkpoint_created':False,'weights_reused_for_training':False,'train_records':records,'validation_records':evaluation}
 def close(self):
  self.platform.close()
  if self.manager._tqdm:self.manager._tqdm.close()
