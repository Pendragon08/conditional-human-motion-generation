"""Paper-sized neural forward/backward workload; never constructs an optimizer.

Official models and manager loss methods are reused. Four primitives execute
sequential backwards with detached history, matching graph lifetime, without
parameter updates. Random VAE and candidate data cannot certify a trained model.
"""
import argparse,gc,hashlib,importlib.metadata,json,os,pathlib,random,shutil,sys,time,traceback
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--base-spec',required=True)
p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_constructed':False,'optimizer_updates':0,'dataset_accepted':False,
 'physical_filter_executed':False,'full_generator_training_started':False,'model_weights_trained':False,'output_writers_closed':False}
started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def save():
 receipt['elapsed_seconds']=time.monotonic()-started;p=out/'receipt.json.partial';p.write_text(json.dumps(receipt,indent=2));p.replace(out/'receipt.json')
try:
 spec=json.loads(pathlib.Path(a.spec).read_text());base=json.loads(pathlib.Path(a.base_spec).read_text());plan=json.loads(pathlib.Path(a.plan).read_text())
 assert os.getuid()==1013 and os.environ['CUDA_VISIBLE_DEVICES']=='1' and len(os.sched_getaffinity(0))<=4
 assert sys.version.split()[0]==base['python_version'] and {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
 assert hashlib.sha256(json.dumps(base,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==spec['warm_base_canonical_hash']
 assert shutil.disk_usage(out).free>spec['minimum_data_free_gib']*1024**3
 view=pathlib.Path(spec['candidate_view']);assert sha(view/'view_receipt.json')==spec['view_receipt_sha256']
 meta=json.loads((view/'view_receipt.json').read_text());assert meta['status']=='UNFILTERED_CANDIDATE_VIEW_READY_NOT_ACCEPTED' and not meta['dataset_accepted']
 for name,entry in meta['artifacts'].items():assert sha(view/name)==entry['sha256']
 import numpy as np,torch
 from omegaconf import OmegaConf
 from paper69_runtime import install_runtime,DifferentiableG1
 from paper69_data_runtime import install_data_sampling
 random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(spec['torch_memory_fraction']);assert torch.cuda.mem_get_info()[0]>spec['minimum_gpu_free_gib']*1024**3
 install_runtime(base['robot_xml']);install_data_sampling()
 from robotmdar.dataloader.data import SkeletonPrimitiveDataset
 from robotmdar.model.mld_vae import AutoMldVae
 from robotmdar.model.mld_denoiser import DenoiserTransformer
 from robotmdar.train.manager import MVAEManager,DARManager,GeometryLoss
 from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion,ModelMeanType,ModelVarType,LossType,get_named_beta_schedule
 device=torch.device('cuda');xml=pathlib.Path(base['robot_xml']);robot_cfg=OmegaConf.create({'asset':{'assetRoot':str(xml.parent),'assetFileName':xml.name}})
 dataset=SkeletonPrimitiveDataset(robot_cfg,512,69,2,8,4,str(view),str(view/'action_statistics.json'),weighted_sample=True,split='train')
 dataset.skeleton=DifferentiableG1(str(xml),device='cuda')
 batch=[(features.to(device),text.to(device)) for features,text in dataset._generate_batch_optimized(generator=torch.Generator().manual_seed(spec['seed']))]
 assert len(batch)==4 and all(x.shape==(512,10,69) and y.shape==(512,512) for x,y in batch)
 weights={'rec':1.,'kl':1e-4,'body_trans':.05,'body_rot':.01,'dof_pos':.03,'dof_vel':1e-5,'foot_contact':.01,
          'latent_rec':1.,'smooth':0.,'quantize_rot':0.,'quantize_trans':0.,'drift_xy':0.,'drift_yaw':0.}
 assert [plan['vae']['transformer_layers'],plan['ldm']['transformer_layers'],plan['vae']['batch_size'],plan['ldm']['batch_size']]==[9,8,512,512]
 assert plan['vae']['loss']['kl_weight']==weights['kl'] and plan['vae']['loss']['dof_velocity_weight']==weights['dof_vel']
 context=GeometryLoss();context.dataset=dataset;context.rec_criterion=torch.nn.HuberLoss();context.loss_weight=weights
 def parameter_sha(module):
  h=hashlib.sha256()
  for name,value in module.named_parameters():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def gradients(module):
  values=[x.grad for x in module.parameters() if x.grad is not None];assert values
  assert bool(torch.stack([torch.isfinite(x).all() for x in values]).all())
  norm=float(torch.stack([x.detach().float().norm().square() for x in values]).sum().sqrt());assert norm>0 and __import__('math').isfinite(norm)
  return {'finite_gradient_tensors':len(values),'gradient_l2_norm':norm}
 receipt.update(status='RUNNING_VAE_BATCH512_BACKWARDS',source_sha256=sha(__file__),spec_sha256=sha(a.spec),
  base_spec_sha256=sha(a.base_spec),view_receipt_sha256=spec['view_receipt_sha256'],cpu_affinity=sorted(os.sched_getaffinity(0)),
  batch_size=512,primitives=4,feature_dim=69,gpu_name=torch.cuda.get_device_name(),candidate_sources_examined=2296,
  source_hashes={name:sha(pathlib.Path(__file__).with_name(name)) for name in ['paper69_runtime.py','paper69_data_runtime.py']})
 save();vae=AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).to(device)
 vae_initial=parameter_sha(vae);vae_bytes=sum(p.numel()*p.element_size() for p in vae.parameters());vae_cases=[]
 for mode in spec['vae_history_modes']:
  torch.cuda.reset_peak_memory_stats();previous=None;cases=[];vae.train();phase_started=time.monotonic()
  for index,(features,text) in enumerate(batch):
   vae.zero_grad(set_to_none=True)
   history=features[:,:2] if previous is None or mode=='ground_truth' else previous[:,-2:]
   future=features[:,2:];latent,dist=vae.encode(future_motion=future,history_motion=history);pred=vae.decode(latent,history,nfuture=8)
   terms,extras=MVAEManager.calc_loss(context,future,pred,dist,history_motion=history)
   assert all(torch.isfinite(v).all() for v in terms.values());terms['total'].backward();grad=gradients(vae)
   previous=pred.detach();cases.append({'primitive':index,'loss':float(terms['total']),**grad})
   del latent,dist,pred,terms,extras
  torch.cuda.synchronize();vae_cases.append({'history_mode':mode,'primitives':cases,'elapsed_seconds':time.monotonic()-phase_started,
      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 assert parameter_sha(vae)==vae_initial;vae.zero_grad(set_to_none=True)
 for parameter in vae.parameters():parameter.requires_grad=False
 vae.eval();previous=None;gc.collect();torch.cuda.empty_cache()
 receipt.update(status='RUNNING_LDM_BATCH512_BACKWARDS',vae=vae_cases,vae_parameters_unchanged=True);save()
 denoiser=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,clip_dim=512,history_shape=(2,69),noise_shape=(1,128),cond_mask_prob=.1).to(device)
 denoiser_initial=parameter_sha(denoiser);denoiser_bytes=sum(p.numel()*p.element_size() for p in denoiser.parameters())
 diffusion=GaussianDiffusion(betas=get_named_beta_schedule('cosine',5),model_mean_type=ModelMeanType.START_X,
    model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE,rescale_timesteps=False)
 torch.cuda.reset_peak_memory_stats();ldm_cases=[];phase_started=time.monotonic();denoiser.train()
 for index,(features,text) in enumerate(batch):
  denoiser.zero_grad(set_to_none=True);history=features[:,:2] if previous is None else previous[:,-2:];future=features[:,2:]
  with torch.no_grad():latent,_=vae.encode(future_motion=future,history_motion=history)
  timesteps=torch.randint(0,5,(512,),device=device);x0=latent.permute(1,0,2)
  noisy=diffusion.q_sample(x0,timesteps,noise=torch.randn_like(x0));conditions={'history_motion_normalized':history,'text_embedding':text}
  zpred=denoiser(noisy,timesteps,y=conditions);pred=vae.decode(zpred.permute(1,0,2),history,nfuture=8)
  terms,extras=DARManager.calc_loss(context,future,pred,latent,None,zpred.permute(1,0,2),torch.ones(512,device=device),history_motion=history)
  assert all(torch.isfinite(v).all() for v in terms.values());terms['total'].backward();grad=gradients(denoiser)
  assert all(p.grad is None for p in vae.parameters());previous=pred.detach();ldm_cases.append({'primitive':index,'loss':float(terms['total']),**grad})
  del latent,x0,noisy,zpred,pred,terms,extras
 torch.cuda.synchronize();ldm={'primitives':ldm_cases,'elapsed_seconds':time.monotonic()-phase_started,
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
 assert parameter_sha(denoiser)==denoiser_initial and parameter_sha(vae)==vae_initial
 denoiser.zero_grad(set_to_none=True);denoiser.eval();segments=[];sampling_started=time.monotonic();history=batch[0][0][:,:2]
 with torch.no_grad():
  def guided(x,t,**kwargs):
   cond=kwargs['y'];un=denoiser(x,t.long(),y={**cond,'uncond':True});yes=denoiser(x,t.long(),y=cond);return un+5*(yes-un)
  for index in range(2):
   cond={'history_motion_normalized':history,'text_embedding':batch[index][1]}
   latent=diffusion.p_sample_loop(guided,(512,1,128),device=device,clip_denoised=False,model_kwargs={'y':cond},progress=False)
   pred=vae.decode(latent.permute(1,0,2),history,nfuture=8);assert pred.shape==(512,8,69) and torch.isfinite(pred).all()
   segments.append(pred);history=pred[:,-2:].clone()
 torch.cuda.synchronize()
 receipt.update(status='TEXTOP_NEURAL_BATCH512_WITNESS_PASSED',vae=vae_cases,ldm=ldm,
   sampling={'batch_size':512,'steps':5,'cfg':5.,'segments':2,'total_frames_per_sample':16,'elapsed_seconds':time.monotonic()-sampling_started,
             'limitation':'Random untrained weights; batching latency is not single-stream deployment latency'},
   vae_parameters_unchanged=True,denoiser_parameters_unchanged=True,frozen_vae_has_no_gradients=True,
   vae_parameter_bytes=vae_bytes,denoiser_parameter_bytes=denoiser_bytes,
   estimated_additional_adam_and_ema_bytes={'vae':3*vae_bytes,'ldm':3*denoiser_bytes},
   optimizer_and_ema_states_actually_allocated=False,loss_implementation='Unmodified pinned MVAEManager.calc_loss and DARManager.calc_loss',
   vae_backward_passes=8,ldm_backward_passes=4,output_writers_closed=True,limitation=spec['limitation'])
 save();print(receipt['status'],flush=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
