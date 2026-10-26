"""Real trained conditional generation. Private output, no posterior future input."""
import argparse,hashlib,importlib.metadata,json,os,pathlib,random,shutil,sys,time,traceback
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
sp=ROOT/'configs/quick-generated-reference-spec.json';spec=json.loads(sp.read_text());base=json.loads((ROOT/'configs/paper69-env-spec.json').read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert a.run.isascii() and a.run.isidentifier() and spec['schema']=='textop.quick_generated_reference.v1' and spec['generation_enabled']
assert os.getuid()==1013 and set(os.sched_getaffinity(0))=={0,1,2,3} and os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
assert sha(__file__)==spec['executor_sha256'] and sys.version.split()[0]==base['python_version']
assert {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
for n,h in spec['bound_sources'].items():assert sha(ROOT/n)==h
assert spec['text_schedule']==[{'text':'walk forward','primitives':16},{'text':'wave your right hand','primitives':16}]
receipts={}
for stage,target in [('vae',2000),('ldm',3000)]:
 b=spec['trained_models'][stage];r=json.loads(pathlib.Path(b['receipt']).read_text());ex=json.loads(pathlib.Path(b['exit']).read_text())
 assert sha(b['receipt'])==b['receipt_sha256'] and sha(b['exit'])==b['exit_sha256'] and ex['evaluation_receipt_sha256']==sha(b['receipt'])
 assert ex['child_exit_code']==0 and ex['supervision_passed'] and r['output_writers_closed'] and r['completed_primitive_attempts']==target
 assert r['status']==('QUICK_VAE_2000_TRAINING_FINISHED' if stage=='vae' else 'QUICK_LDM_3000_TRAINING_FINISHED')
 assert sha(b['checkpoint']['path'])==b['checkpoint']['sha256'] and b['checkpoint'] in r['checkpoints'] and b['checkpoint']['manager_step']==target
 assert r['latest_file_resume_verification']['rng_and_generator_restored_exactly'];receipts[stage]=r
assert receipts['vae']['vae_quality_gate_passed'] and receipts['ldm']['vae_frozen_and_unchanged']
corpus=pathlib.Path(spec['corpus_dir']);meta=json.loads((corpus/'corpus_receipt.json').read_text());assert sha(corpus/'corpus_receipt.json')==spec['corpus_receipt_sha256']
assert sha(corpus/'meanstd.pkl')==meta['artifacts']['meanstd.pkl']['sha256']
out=ROOT/'generated_reference_private'/a.run;out.parent.mkdir(mode=0o700,exist_ok=True);out.mkdir(mode=0o700,exist_ok=False);started=time.monotonic()
r={'status':'STARTING_TRAINED_QUICK_GENERATION','quality_label':'QUICK_REPRODUCTION_NOT_PAPER_COMPLETE','optimizer_updates':0,'physics_steps':0,
 'posterior_future_used':False,'real_time_or_closed_loop_verified':False,'source_sha256':sha(__file__),'spec_sha256':sha(sp),'output_writers_closed':False,'text_schedule':spec['text_schedule'],
 'ldm_fixed_validation_sampling_huber_initial':receipts['ldm']['initial_full_sampling_quality']['normalized_full_sampling_huber'],
 'ldm_fixed_validation_sampling_huber_final':receipts['ldm']['final_full_sampling_quality']['normalized_full_sampling_huber'],
 'model_quality_accepted':False}
def save():
 r['elapsed_seconds']=time.monotonic()-started;tmp=out/'receipt.partial';tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(out/'receipt.json')
save()
try:
 import numpy as np,torch,clip
 from paper69_runtime import install_runtime,decode69,DifferentiableG1
 from generator_training_state import module_sha
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65);assert torch.cuda.mem_get_info()[0]>20*1024**3
 random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
 install_runtime(base['robot_xml'])
 from robotmdar.model.mld_vae import AutoMldVae
 from robotmdar.model.mld_denoiser import DenoiserTransformer
 from robotmdar.model.clip import encode_text
 from robotmdar.dtype.motion import get_zero_feature
 from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion,ModelMeanType,ModelVarType,LossType,get_named_beta_schedule
 vae=AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).cuda()
 denoiser=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,clip_dim=512,history_shape=(2,69),noise_shape=(1,128),cond_mask_prob=.1).cuda()
 for key,model,field in [('vae',vae,'vae'),('ldm',denoiser,'denoiser')]:
  saved=torch.load(spec['trained_models'][key]['checkpoint']['path'],map_location='cpu');model.load_state_dict(saved[field],strict=True);del saved
  model.eval()
  for param in model.parameters():param.requires_grad=False
 assert module_sha(vae)==receipts['vae']['trained_model_sha256'] and module_sha(denoiser)==receipts['ldm']['trained_denoiser_sha256']
 weight=next(pathlib.Path(x) for x in base['clip_weight_candidates'] if pathlib.Path(x).exists());assert sha(weight)==meta['clip_weight_sha256']
 encoder,_=clip.load(str(weight),device='cuda',jit=False);clip.model.convert_weights(encoder);encoder.eval()
 for param in encoder.parameters():param.requires_grad=False
 with torch.inference_mode():embedding=encode_text(encoder,[x['text'] for x in spec['text_schedule']])
 assert embedding.shape==(2,512) and torch.isfinite(embedding).all() and bool((embedding.norm(dim=1)>0).all());del encoder
 mean,std=torch.load(corpus/'meanstd.pkl',map_location='cpu');mean=mean.cuda();std=std.cuda();assert mean.shape==std.shape==(69,) and bool((std>0).all())
 neutral=get_zero_feature().cuda().reshape(1,1,69).repeat(1,2,1);history=(neutral-mean)/std;features=[history.cpu().numpy()]
 diffusion=GaussianDiffusion(betas=get_named_beta_schedule('cosine',5),model_mean_type=ModelMeanType.START_X,model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE,rescale_timesteps=False)
 latency=[];records=[];last=None
 with torch.inference_mode():
  for phase,entry in enumerate(spec['text_schedule']):
   for index in range(entry['primitives']):
    if last is not None:assert torch.equal(history,last[:,-2:]),'Only generated reference history after initial two neutral frames'
    y={'text_embedding':embedding[phase:phase+1],'history_motion_normalized':history}
    def guided(x,t,y):
     un=denoiser(x,t.long(),y={**y,'uncond':True});yes=denoiser(x,t.long(),y=y);return un+5*(yes-un)
    torch.cuda.synchronize();t0=time.perf_counter()
    latent=diffusion.p_sample_loop(guided,(1,1,128),device='cuda',clip_denoised=False,model_kwargs={'y':y},progress=False)
    pred=vae.decode(latent.permute(1,0,2),history,nfuture=8);torch.cuda.synchronize();elapsed=time.perf_counter()-t0
    assert pred.shape==(1,8,69) and torch.isfinite(pred).all();features.append(pred.cpu().numpy());latency.append(elapsed)
    records.append({'primitive':len(records),'phase':phase,'normalized_history_sha256':hashlib.sha256(history.cpu().numpy().tobytes()).hexdigest(),'wall_seconds':elapsed})
    last=pred;history=pred[:,-2:]
  normalized=np.concatenate(features,axis=1)[0];raw=torch.from_numpy(normalized).cuda()*std+mean
  motion=decode69(raw,abs_pose={'root_trans_offset':raw.new_tensor([0.,0.,.75]),'root_rot':raw.new_tensor([0.,0.,0.,1.])})
  qpos=torch.cat([motion['root_trans_offset'],motion['root_rot'][:,[3,0,1,2]],motion['dof']],dim=-1).cpu().numpy()
 assert normalized.shape==(258,69) and qpos.shape==(258,36) and np.isfinite(qpos).all()
 assert np.max(np.abs(np.linalg.norm(qpos[:,3:7],axis=1)-1))<1e-5
 assert shutil.disk_usage(out).free>2*1024**3+64*1024**2
 robot=DifferentiableG1(base['robot_xml']);assert len(robot.joint_names)==29
 dest=out/'generated.private.npz';np.savez_compressed(dest,qpos_29=qpos.astype(np.float32),normalized_features=normalized.astype(np.float32),features=raw.cpu().numpy().astype(np.float32),text_embeddings=embedding.cpu().numpy())
 private={'frames':258,'fps':50,'joint_names':robot.joint_names,'root_quaternion':'WXYZ','generation_records':records,'initial_history':'two neutral reference feature frames','future_conditioning':'none; noise+text+generated reference history','world_decode':'concatenated non-overlapping generated feature stream, one global integration','model_checkpoints':{k:v['checkpoint'] for k,v in spec['trained_models'].items()}}
 mp=out/'generation.private.json';mp.write_text(json.dumps(private,indent=2))
 r.update(status='TRAINED_QUICK_TEXT_REFERENCE_GENERATED',frames=258,generated_future_frames=256,initial_history_frames=2,primitives=32,fps=50,text_switch_at_generated_future_frame=128,
  artifact={'file':str(dest),'bytes':dest.stat().st_size,'sha256':sha(dest)},private_metadata_sha256=sha(mp),corpus_receipt_sha256=spec['corpus_receipt_sha256'],
  trained_vae_checkpoint_sha256=spec['trained_models']['vae']['checkpoint']['sha256'],trained_ldm_checkpoint_sha256=spec['trained_models']['ldm']['checkpoint']['sha256'],
  root_height_min_m=float(qpos[:,2].min()),root_height_max_m=float(qpos[:,2].max()),max_joint_difference_speed_rad_s=float(np.abs(np.diff(qpos[:,7:],axis=0)*50).max()),
  primitive_latency_mean_seconds=float(np.mean(latency)),primitive_latency_max_seconds=float(max(latency)),primitive_nominal_duration_seconds=.16,
  primitive_latencies_seconds=latency,physical_tracking_attempted=False,text_semantic_quality_verified=False,output_writers_closed=True,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
 save();print(json.dumps(r))
except BaseException as e:
 r.update(status='FAILED_QUICK_TEXT_GENERATION',error=repr(e),traceback=traceback.format_exc());save();raise
