"""Bounded candidate-only Adam/AdamW resume check; never a formal training run.

Two four-primitive batches per stage, checkpoint after the first. Official loss,
manager, optimizer, clipping, EMA and schedule sampler are reused. Evaluation
frequency is diagnostic-only; model/stage/learning-rate sizes follow the plan.
"""
import argparse,gc,hashlib,importlib.metadata,json,math,os,pathlib,random,shutil,sys,time,traceback
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--base-spec',required=True)
p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
receipt={'status':'STARTING','dataset_accepted':False,'physical_filter_executed':False,
 'full_generator_training_started':False,'optimizer_updates':0,'stages':{},'output_writers_closed':False}
def save():
 receipt['elapsed_seconds']=time.monotonic()-started
 t=out/'receipt.json.partial';t.write_text(json.dumps(receipt,indent=2));t.replace(out/'receipt.json')
try:
 spec=json.loads(pathlib.Path(a.spec).read_text());base=json.loads(pathlib.Path(a.base_spec).read_text());plan=json.loads(pathlib.Path(a.plan).read_text())
 assert spec['dispatch_enabled'] and spec['status']=='DECLARED_CANDIDATE_DIAGNOSTIC_ONLY'
 assert os.getuid()==1013 and os.environ['CUDA_VISIBLE_DEVICES']=='1' and len(os.sched_getaffinity(0))<=4
 assert sys.version.split()[0]==base['python_version'] and {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
 assert hashlib.sha256(json.dumps(base,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==spec['warm_base_canonical_hash']
 assert shutil.disk_usage(out).free>spec['minimum_data_free_gib']*1024**3+spec['run_checkpoint_budget_bytes']
 import numpy as np,torch
 from omegaconf import OmegaConf
 from generator_training_state import sha,module_sha,PrimitiveClock,save_manager,load_manager,equal
 from paper69_runtime import install_runtime,DifferentiableG1
 from paper69_data_runtime import install_data_sampling
 view=pathlib.Path(spec['candidate_view']);assert sha(view/'view_receipt.json')==spec['view_receipt_sha256']
 meta=json.loads((view/'view_receipt.json').read_text());assert meta['status']=='UNFILTERED_CANDIDATE_VIEW_READY_NOT_ACCEPTED' and not meta['dataset_accepted']
 for name,entry in meta['artifacts'].items():assert sha(view/name)==entry['sha256']
 random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(spec['torch_memory_fraction']);assert torch.cuda.mem_get_info()[0]>spec['minimum_gpu_free_gib']*1024**3
 install_runtime(base['robot_xml']);install_data_sampling()
 from robotmdar.dataloader.data import SkeletonPrimitiveDataset
 from robotmdar.model.mld_vae import AutoMldVae
 from robotmdar.model.mld_denoiser import DenoiserTransformer
 from robotmdar.train.manager import MVAEManager,DARManager
 from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion,ModelMeanType,ModelVarType,LossType,get_named_beta_schedule
 from robotmdar.diffusion.resample import UniformSampler
 device=torch.device('cuda');xml=pathlib.Path(base['robot_xml']);robot_cfg=OmegaConf.create({'asset':{'assetRoot':str(xml.parent),'assetFileName':xml.name}})
 datasets={}
 for split in ['train','val']:
  datasets[split]=SkeletonPrimitiveDataset(robot_cfg,512,69,2,8,4,str(view),str(view/'action_statistics.json'),weighted_sample=split=='train',split=split)
  datasets[split].skeleton=DifferentiableG1(str(xml),device='cuda')
 generators={k:torch.Generator().manual_seed(spec['seed']+i) for i,k in enumerate(['train','val'])}
 weights={'rec':1.,'kl':1e-4,'body_trans':.05,'body_rot':.01,'dof_pos':.03,'dof_vel':1e-5,'foot_contact':.01,
  'latent_rec':1.,'smooth':0.,'quantize_rot':0.,'quantize_trans':0.,'drift_xy':0.,'drift_yaw':0.}
 diffusion=GaussianDiffusion(betas=get_named_beta_schedule('cosine',5),model_mean_type=ModelMeanType.START_X,
  model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE,rescale_timesteps=False)
 sampler=UniformSampler(diffusion)
 class Platform:
  def __init__(self,path):self.handle=path.open('x')
  def report_scalar(self,name,value,step,group_name):
   self.handle.write(json.dumps({'name':name,'value':float(value),'step':step,'group':group_name})+'\n')
  def close(self):self.handle.flush();self.handle.close()
 def new_vae():return AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).to(device)
 def construct(stage,label,vae_path=None):
  folder=out/stage;folder.mkdir(exist_ok=True);cfg=plan[stage]
  common=dict(stages=cfg['stages_from_released_USAGE'],use_rollout=True,use_static_pose=stage=='vae',static_prob=.001,
   static_perturbation_scale=0.,save_every=20000,eval_every=4,eval_steps=4,device='cuda',save_dir=str(folder),
   learning_rate=cfg['learning_rate'],anneal_lr=True,max_grad_norm=1.,use_ema=spec['ema_fixture_enabled'],ema_decay=spec['ema_decay'],
   loss_weight=weights,ckpt=OmegaConf.create({'vae':str(vae_path) if vae_path else None,'dar':None}),
   platform=Platform(folder/(label+'_metrics.jsonl')))
  if stage=='vae':
   model=new_vae();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
   manager=MVAEManager(**common);manager.hold_model(model,optimizer,datasets['train'])
  else:
   vae=new_vae();model=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,clip_dim=512,history_shape=(2,69),noise_shape=(1,128),cond_mask_prob=.1).to(device)
   optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
   manager=DARManager(**common,use_full_sample=False);manager.hold_model(vae,model,optimizer,datasets['train'])
  return manager,PrimitiveClock(manager)
 def batch(split):
  result=datasets[split]._generate_batch_optimized(generator=generators[split])
  assert len(result)==4 and all(x.shape==(512,10,69) and y.shape==(512,512) for x,y in result)
  return result
 def batch_hash(value):
  h=hashlib.sha256()
  for pair in value:
   for t in pair:h.update(t.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def probe():return {'python':random.random(),'numpy':np.random.random(3),'torch':torch.rand(3),'cuda':torch.rand(3,device=device).cpu()}
 def run_batch(stage,manager,clock,value,is_eval=False):
  model=manager.vae if stage=='vae' else manager.denoiser;model.train(not is_eval)
  if stage=='ldm':manager.vae.eval()
  previous=None;records=[]
  for index,(features,text) in enumerate(value):
   manager.pre_step(is_eval=is_eval);features=features.to(device);text=text.to(device)
   future=features[:,2:];history=features[:,:2] if is_eval else manager.choose_history(features[:,:2],previous,2)
   with torch.set_grad_enabled(not is_eval):
    latent,dist=manager.vae.encode(future_motion=future,history_motion=history)
    if stage=='vae':
     prediction=manager.vae.decode(latent,history,nfuture=8)
     terms,extras=manager.calc_loss(future,prediction,dist,history_motion=history)
    else:
     timesteps,importance=sampler.sample(512,device=device);x0=latent.permute(1,0,2)
     noisy=diffusion.q_sample(x0,timesteps,noise=torch.randn_like(x0))
     predicted_latent=manager.denoiser(noisy,diffusion._scale_timesteps(timesteps),y={'history_motion_normalized':history,'text_embedding':text}).permute(1,0,2)
     prediction=manager.vae.decode(predicted_latent,history,nfuture=8)
     terms,extras=manager.calc_loss(future,prediction,latent,None,predicted_latent,importance,history_motion=history)
    assert all(torch.isfinite(v).all() for v in terms.values())
    finite=True
    if not is_eval:
     manager.optimizer.zero_grad();terms['total'].backward()
     grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads
     finite=bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
     if finite:manager.grad_clip(model);manager.optimizer.step()
     if stage=='ldm':assert all(p.grad is None and not p.requires_grad for p in manager.vae.parameters())
   previous=prediction.detach()
   manager.post_step(is_eval=is_eval,loss_dict={k:v.detach().cpu() for k,v in terms.items()},
    extras={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in extras.items()})
   if not is_eval:clock.after_post_step()
   records.append({'primitive':index,'manager_step':manager.step,'loss':float(terms['total']),'gradients_finite':finite,
    'actual_optimizer_updates':clock.updates,'is_evaluation':is_eval})
   receipt['optimizer_updates']=sum(s.get('actual_optimizer_updates',0) for s in receipt['stages'].values())+clock.updates
  return records
 receipt.update(status='RUNNING_ACTUAL_OPTIMIZER_RESUME_DIAGNOSTIC',source_sha256=sha(__file__),spec_sha256=sha(a.spec),
  base_spec_sha256=sha(a.base_spec),plan_sha256=sha(a.plan),view_receipt_sha256=spec['view_receipt_sha256'],cpu_affinity=sorted(os.sched_getaffinity(0)),
  batch_size=512,primitives=4,feature_dim=69,source_hashes={name:sha(pathlib.Path(__file__).with_name(name)) for name in ['generator_training_state.py','paper69_runtime.py','paper69_data_runtime.py']},
  ema_diagnostic_fixture=True,release_ema_default=False,checkpoint_total_budget_bytes=spec['run_checkpoint_budget_bytes'])
 save();vae_path=None
 for stage,key in [('vae','vae'),('ldm','denoiser')]:
  torch.cuda.reset_peak_memory_stats();manager,clock=construct(stage,'before_resume',vae_path);model=getattr(manager,key)
  initial=module_sha(model);frozen_initial=module_sha(manager.vae) if stage=='ldm' else None
  first=run_batch(stage,manager,clock,batch('train'));assert clock.updates==4 and clock.skipped==0 and manager._to_eval_steps==4
  binding={'stage':stage,'feature_dim':69,'dof':29,'view_receipt_sha256':spec['view_receipt_sha256'],
   'base_spec_sha256':sha(a.base_spec),'state_spec_sha256':sha(a.spec),'robot_sha256':sha(xml),'source_hashes':receipt['source_hashes'],
   'candidate_only':True,'dataset_accepted':False,'formal_training':False,'frozen_vae_checkpoint_sha256':sha(vae_path) if vae_path else None}
  checkpoint=save_manager(manager,key,clock,generators,out/stage/'state_after4.pth',binding,
   spec['minimum_data_free_gib']*1024**3,spec['checkpoint_safety_budget_bytes'])
  expected_batch=batch_hash(batch('train'));expected_probe=probe();manager.platform.close()
  if manager._tqdm:manager._tqdm.close()
  del manager,clock,model;gc.collect();torch.cuda.empty_cache()
  manager,clock=construct(stage,'after_resume',vae_path);model=getattr(manager,key)
  restored=load_manager(manager,key,clock,generators,checkpoint['path'],binding,checkpoint['sha256'])
  assert batch_hash(batch('train'))==expected_batch and equal(probe(),expected_probe)
  # Restore again after the peek, so real validation and resumed training consume
  # the same RNG stream an uninterrupted manager would have consumed.
  load_manager(manager,key,clock,generators,checkpoint['path'],binding,checkpoint['sha256'])
  rejected=False
  try:load_manager(manager,key,clock,generators,checkpoint['path'],{**binding,'feature_dim':57},checkpoint['sha256'])
  except AssertionError:rejected=True
  assert rejected and manager._to_eval_steps==4 and clock.updates==4
  evaluation=run_batch(stage,manager,clock,batch('val'),is_eval=True)
  assert not manager.should_eval() and manager.step==4 and clock.updates==4
  second=run_batch(stage,manager,clock,batch('train'));assert clock.updates==8 and clock.skipped==0 and manager.step==8
  assert module_sha(model)!=initial and all(torch.isfinite(t).all() for t in model.state_dict().values())
  assert manager.optimizer.state and manager.ema_models
  adam_steps=[int(v['step'].item()) for v in manager.optimizer.state.values() if 'step' in v];assert adam_steps and set(adam_steps)=={8}
  if stage=='ldm':assert module_sha(manager.vae)==frozen_initial
  torch.cuda.synchronize()
  receipt['stages'][stage]={'actual_optimizer_updates':clock.updates,'completed_primitive_attempts':clock.attempts,'skipped_updates':clock.skipped,
   'train_batch_draws':2,'validation_batch_draws':1,'model_parameters_changed':True,'optimizer':type(manager.optimizer).__name__,
   'optimizer_state_tensors_allocated':True,'adam_parameter_step_min':min(adam_steps),'adam_parameter_step_max':max(adam_steps),
   'optimizer_weight_decay':manager.optimizer.param_groups[0]['weight_decay'],'ema_state_allocated':True,'checkpoint':checkpoint,
   'restore':restored,'next_batch_and_rng_probe_identical':True,'mismatched_binding_rejected':True,'pending_validation_completed_without_optimizer_updates':True,
   'initial_parameter_sha256':initial,'after_resume_parameter_sha256':module_sha(model),'train_before':first,'validation':evaluation,'train_after':second,
   'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(),
   'frozen_vae_unchanged':stage=='ldm','saved_checkpoint_manager_step':4,'final_diagnostic_manager_step':8}
  if stage=='vae':vae_path=pathlib.Path(checkpoint['path'])
  manager.platform.close()
  if manager._tqdm:manager._tqdm.close()
  del manager,clock,model;gc.collect();torch.cuda.empty_cache();save()
 receipt.update(status='TEXTOP_GENERATOR_STATE_WITNESS_PASSED',optimizer_updates=16,
  model_weights_changed_by_diagnostic=True,formal_generator_checkpoint_produced=False,output_writers_closed=True,
  limitation=spec['scope'],minimum_free_data_bytes_after=shutil.disk_usage(out).free)
 save();print(receipt['status'],flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
