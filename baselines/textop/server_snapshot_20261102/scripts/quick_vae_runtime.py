"""Official VAE math on the accepted quick corpus; audited counters and state only."""
import hashlib, io, json, pathlib, random
import numpy as np
import torch
from generator_training_state import PrimitiveClock, equal, module_sha, rng_state

class Platform:
 def __init__(self,path):self.stream=path.open('x')
 def report_scalar(self,name,value,step,group_name):
  self.stream.write(json.dumps({'name':name,'value':float(value),'step':step,'group':group_name})+'\n')
 def close(self):self.stream.flush();self.stream.close()

def restore_rng(state,generators):
 random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
 torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
 for k,v in state['generators'].items():generators[k].set_state(v.cpu())

def batch_sha(batch):
 h=hashlib.sha256()
 for pair in batch:
  for t in pair:h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

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
  from robotmdar.train.manager import MVAEManager
  xml=pathlib.Path(base['robot_xml']);robot=OmegaConf.create({'asset':{'assetRoot':str(xml.parent),'assetFileName':xml.name}})
  self.datasets={}
  for split in ['train','val']:
   ds=SkeletonPrimitiveDataset(robot,512,69,2,8,4,spec['corpus_dir'],str(pathlib.Path(spec['corpus_dir'])/'action_statistics.json'),weighted_sample=split=='train',split=split)
   ds.skeleton=DifferentiableG1(str(xml),device='cuda');self.datasets[split]=ds
  self.generators={k:torch.Generator().manual_seed(spec['seed']+i) for i,k in enumerate(['train','val'])}
  self.model=AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).cuda()
  optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4)
  weights={'rec':1.,'kl':1e-4,'body_trans':.05,'body_rot':.01,'dof_pos':.03,'dof_vel':1e-5,'foot_contact':.01,
   'latent_rec':1.,'smooth':0.,'quantize_rot':0.,'quantize_trans':0.,'drift_xy':0.,'drift_yaw':0.}
  self.platform=Platform(out/'metrics.jsonl');self.requested_saves=[]
  m=MVAEManager(stages=[1000,500,500],use_rollout=True,use_static_pose=True,static_prob=.001,static_perturbation_scale=0.,
   save_every=1000,eval_every=500,eval_steps=12,device='cuda',save_dir=str(out),learning_rate=1e-4,anneal_lr=True,
   max_grad_norm=1.,use_ema=False,loss_weight=weights,ckpt=OmegaConf.create({'vae':None,'dar':None}),platform=self.platform)
  m.hold_model(self.model,optimizer,self.datasets['train'])
  # The native post_step still schedules saves, but actual full-state I/O runs
  # after the four-primitive batch and pending native validation have closed.
  m.save_model=lambda:self.requested_saves.append(m.step)
  assert m.ema_models=={} and m.max_steps==2000
  self.manager=m;self.clock=PrimitiveClock(m);self.consecutive_skips=0
 def batch(self,split):
  b=self.datasets[split]._generate_batch_optimized(generator=self.generators[split])
  assert len(b)==4 and all(x.shape==(512,10,69) and y.shape==(512,512) for x,y in b)
  return b
 def run_batch(self,batch,is_eval=False):
  m=self.manager;self.model.train(not is_eval);previous=None;records=[]
  for features,text in batch:
   m.pre_step(is_eval=is_eval);features=features.cuda();future=features[:,2:]
   history=features[:,:2] if is_eval else m.choose_history(features[:,:2],previous,2)
   with torch.set_grad_enabled(not is_eval):
    z,dist=self.model.encode(future_motion=future,history_motion=history);pred=self.model.decode(z,history,nfuture=8)
    terms,extras=m.calc_loss(future,pred,dist,history_motion=history)
    assert all(torch.isfinite(v).all() for v in terms.values()),'Nonfinite loss: preserve failure, do not clip it'
    finite=True
    if not is_eval:
     m.optimizer.zero_grad();terms['total'].backward();grads=[p.grad for p in self.model.parameters() if p.grad is not None];assert grads
     finite=bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
     if finite:m.grad_clip(self.model);m.optimizer.step();self.consecutive_skips=0
     else:self.consecutive_skips+=1
   previous=pred.detach()
   m.post_step(is_eval=is_eval,loss_dict={k:v.detach().cpu() for k,v in terms.items()},extras={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in extras.items()})
   if not is_eval:self.clock.after_post_step()
   records.append({'manager_step':m.step,'loss':float(terms['total']),'reconstruction_huber':float(terms['rec']),
     'gradients_finite':finite,'actual_optimizer_updates':self.clock.updates,'skipped_updates':self.clock.skipped,'is_evaluation':is_eval})
   assert self.consecutive_skips<20,'Twenty consecutive official gradient skips; preserve state and diagnose'
  return records
 def quality(self,batch,rollout=False):
  outer=rng_state(self.generators);torch.manual_seed(20261125);torch.cuda.manual_seed_all(20261125)
  mode=self.model.training;self.model.eval();previous=None;losses=[]
  with torch.no_grad():
   for features,text in batch:
    features=features.cuda();future=features[:,2:];history=previous[:,-2:] if rollout and previous is not None else features[:,:2]
    z,_=self.model.encode(future_motion=future,history_motion=history);pred=self.model.decode(z,history,nfuture=8)
    assert torch.isfinite(pred).all();losses.append(float(torch.nn.functional.huber_loss(pred,future)));previous=pred
  self.model.train(mode);restore_rng(outer,self.generators)
  return {'normalized_reconstruction_huber':sum(losses)/len(losses),'four_primitives':True,'always_previous_prediction_history':rollout,
          'final_physical_endpoint_or_text_generation_evaluated':False}
 def memory_witness(self):
  m=self.manager;records=self.run_batch(self.batch('train'));assert self.clock.updates==4 and self.clock.skipped==0
  m._to_eval_steps=4
  state={'model':self.model.state_dict(),'optimizer':m.optimizer.state_dict(),'rng':rng_state(self.generators),
         'clock':self.clock.state(),'step':m.step,'pending_eval':m._to_eval_steps}
  memory=io.BytesIO();torch.save(state,memory);payload=memory.getvalue();saved=torch.load(io.BytesIO(payload),map_location='cpu')
  expected_batch=batch_sha(self.batch('train'));before=module_sha(self.model)
  with torch.no_grad():next(self.model.parameters()).add_(.01)
  self.model.load_state_dict(saved['model']);m.optimizer.load_state_dict(saved['optimizer']);self.clock.load(saved['clock']);m.step=saved['step'];m._to_eval_steps=saved['pending_eval'];restore_rng(saved['rng'],self.generators)
  assert module_sha(self.model)==before and equal(self.model.state_dict(),saved['model']) and equal(m.optimizer.state_dict(),saved['optimizer'])
  assert batch_sha(self.batch('train'))==expected_batch;restore_rng(saved['rng'],self.generators)
  evaluation=self.run_batch(self.batch('val'),is_eval=True);assert self.clock.updates==4 and m.step==4 and not m.should_eval()
  return {'status':'QUICK_VAE_REAL_GRADIENT_MEMORY_RESUME_WITNESS_PASSED','actual_optimizer_updates':4,'primitive_attempts':4,
    'all_four_training_gradients_finite':True,'validation_optimizer_updates':0,'model_and_adam_roundtrip_exact':True,
    'next_batch_and_rng_restored':True,'pending_evaluation_completed_without_updates':True,'ema_enabled':False,
    'memory_serialized_bytes':len(payload),'memory_serialized_sha256':hashlib.sha256(payload).hexdigest(),
    'persistent_checkpoint_created':False,'weights_reused_for_training':False,'train_records':records,'validation_records':evaluation,
    'file_resume_scope':'Existing unchanged generator_training_state.py file protocol; first real1000 checkpoint will be reloaded and checked'}
 def close(self):
  self.platform.close()
  if self.manager._tqdm:self.manager._tqdm.close()
