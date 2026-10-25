"""Seeded69D codec/FK and official VAE/LDM numerical validation, not training."""
import argparse,datetime,hashlib,importlib,importlib.metadata,json,pathlib,random,sys,traceback,types
import numpy as np,torch,mujoco
from paper69_runtime import DifferentiableG1,encode69,decode69,to_qpos69,quat_matrix,install_runtime

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def finite_gradients(module):
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert sum(float(x.detach().abs().sum()) for x in grads)>0
    return len(grads)

def main(args):
    if args.output.exists():raise FileExistsError('Preserve existing witness: '+str(args.output))
    spec=json.loads(args.spec.read_text());versions={k:importlib.metadata.version(k) for k in spec['required_versions']}
    assert versions==spec['required_versions'] and sys.version.split()[0]==spec['python_version']
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
    if args.device=='cuda':
        assert torch.cuda.is_available();torch.cuda.manual_seed_all(0);torch.cuda.set_per_process_memory_fraction(spec['gpu_fraction'])
    device=torch.device(args.device)
    adapter=install_runtime(spec['robot_xml']);kin=DifferentiableG1(spec['robot_xml'],device)
    from unittest.mock import patch
    from robotmdar.train.manager import DARManager,MVAEManager
    curriculum=types.SimpleNamespace(use_rollout=True,stage_idx=2,step=300000,stages=[100000,100000,100000])
    with patch('torch.rand',return_value=torch.tensor([.85])):
        assert DARManager.should_rollout(curriculum) is False
        assert MVAEManager.should_rollout(curriculum) is True
    with patch('torch.rand',return_value=torch.tensor([.79])):assert DARManager.should_rollout(curriculum) is True
    fixtures=[]
    for p in sorted(pathlib.Path(spec['fixture_dir']).glob('*.private.npz')):
        with np.load(p,allow_pickle=False) as data:
            if len(data['qpos_29'])<35:continue
            fixtures.append({'file':p,'qpos':data['qpos_29'][:35].copy(),'contact':data['contact'][:35].copy(),'features':data['features_69'][:34].copy()})
        if len(fixtures)==2:break
    assert len(fixtures)==2
    q=torch.as_tensor(np.stack([x['qpos'] for x in fixtures]),device=device,dtype=torch.float32)
    contact=torch.as_tensor(np.stack([x['contact'] for x in fixtures]),device=device,dtype=torch.float32)
    motion={'root_trans_offset':q[...,:3],'root_rot':q[...,3:7][...,[1,2,3,0]],'dof':q[...,7:],'contact_mask':contact}
    features,pose=encode69(motion);decoded=decode69(features,pose);recovered,_=to_qpos69(decoded)
    rotation_error=float((quat_matrix(recovered[...,3:7])-quat_matrix(q[:,:-1,3:7])).abs().max())
    codec_error={'translation_m':float((recovered[...,:3]-q[:,:-1,:3]).abs().max()),'joint_rad':float((recovered[...,7:]-q[:,:-1,7:]).abs().max()),'rotation_matrix':rotation_error,'stored_numpy_feature':float((features-torch.as_tensor(np.stack([x['features'] for x in fixtures]),device=device)).abs().max())}
    assert max(codec_error.values())<1e-4,codec_error
    # Include composed nonzero root rotations and moving wrists, not only a zero pose.
    from paper69_runtime import rpy_quat
    probe=q[:,:12].clone();probe[...,3:7]=rpy_quat(torch.linspace(-.4,.4,12,device=device)[None].expand(2,-1),torch.linspace(.3,-.3,12,device=device)[None].expand(2,-1),torch.linspace(-.7,.7,12,device=device)[None].expand(2,-1));probe[...,7:]=torch.randn_like(probe[...,7:])*.35
    input_probe={'root_trans_offset':probe[...,:3],'root_rot':probe[...,3:7][...,[1,2,3,0]],'dof':probe[...,7:],'contact_mask':contact[:,:12]}
    predicted=kin.forward_kinematics(input_probe)
    reference_pos=[];reference_rot=[];mjdata=mujoco.MjData(kin.model)
    for sequence in probe.detach().cpu().numpy():
        positions=[];rotations=[]
        for frame in sequence:
            mjdata.qpos[:]=frame;mujoco.mj_forward(kin.model,mjdata);positions.append(mjdata.xpos[1:].copy());rotations.append(mjdata.xmat[1:].reshape(-1,3,3).copy())
        reference_pos.append(positions);reference_rot.append(rotations)
    fk_error={'position_m':float(np.abs(predicted['global_translation'].detach().cpu().numpy()-np.asarray(reference_pos)).max()),'rotation_matrix':float(np.abs(predicted['global_rotation_mat'].detach().cpu().numpy()-np.asarray(reference_rot)).max())}
    assert max(fk_error.values())<5e-5,fk_error
    from robotmdar.model.mld_vae import AutoMldVae
    from robotmdar.model.mld_denoiser import DenoiserTransformer
    from robotmdar.train.manager import GeometryLoss
    vae=AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).to(device)
    denoiser=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,clip_dim=512,history_shape=(2,69),noise_shape=(1,128),cond_mask_prob=.1).to(device)
    geometry=GeometryLoss()
    def reconstruct(f,need_denormalize=True,ret_fk=True,**kwargs):
        return kin.forward_kinematics(decode69(f))
    geometry.dataset=types.SimpleNamespace(skeleton=kin,reconstruct_motion=reconstruct)
    geometry.rec_criterion=torch.nn.HuberLoss()
    weights={'body_trans':.05,'body_rot':.01,'dof_pos':.03,'dof_vel':1e-5,'foot_contact':.01}
    def geom_loss(pred,target):
        terms,_=geometry.calc_geometry_loss(pred,target)
        return sum(weights[k]*terms[k] for k in weights),terms
    history=features[:,:2].detach();future=features[:,2:10].detach()
    latent,dist=vae.encode(future_motion=future,history_motion=history);out=vae.decode(latent,history,nfuture=8)
    assert list(latent.shape)==[1,2,128] and list(out.shape)==[2,8,69]
    gl,terms=geom_loss(out,future)
    kl=torch.distributions.kl_divergence(dist,torch.distributions.Normal(torch.zeros_like(dist.loc),torch.ones_like(dist.scale))).mean()
    vae_loss=torch.nn.functional.huber_loss(out,future)+1e-4*kl+gl
    assert torch.isfinite(vae_loss);vae_loss.backward();vae_grad_count=finite_gradients(vae)
    # The geometric loss itself must differentiate all six true wrist dimensions.
    probe_features=(future+.03*torch.randn_like(future)).detach().requires_grad_(True)
    geometric_only,_=geom_loss(probe_features,future);geometric_only.backward()
    wrist_indices=[i for i,name in enumerate(kin.joint_names) if 'wrist' in name]
    wrist_grad=probe_features.grad[..., [11+i for i in wrist_indices]].abs().mean((0,1))
    assert len(wrist_indices)==6 and torch.isfinite(wrist_grad).all() and (wrist_grad>0).all()
    for p in vae.parameters():p.grad=None;p.requires_grad=False
    vae.eval()
    # Existing official CLIP weights only; this witness never downloads a weight.
    import clip
    clip_impl=importlib.import_module('clip.clip');expected=clip_impl._MODELS['ViT-B/32'].split('/')[-2]
    weight=next((pathlib.Path(p) for p in spec['clip_weight_candidates'] if pathlib.Path(p).is_file()),None)
    assert weight and len(expected)==64 and sha(weight)==expected
    clip_model,_=clip.load(str(weight),device=device,jit=False);clip_model.eval()
    for p in clip_model.parameters():p.requires_grad=False
    with torch.no_grad():embedding=clip_model.encode_text(clip.tokenize(['walk forward','wave the right hand']).to(device)).float();z0,_=vae.encode(future_motion=future,history_motion=history)
    assert list(embedding.shape)==[2,512] and torch.isfinite(embedding).all()
    from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion,ModelMeanType,ModelVarType,LossType,get_named_beta_schedule
    diffusion=GaussianDiffusion(betas=get_named_beta_schedule('cosine',5),model_mean_type=ModelMeanType.START_X,model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE,rescale_timesteps=False)
    noise=torch.randn(2,1,128,device=device);timesteps=torch.tensor([1,3],device=device)
    noisy=diffusion.q_sample(z0.permute(1,0,2),timesteps,noise=noise)
    y={'history_motion_normalized':history,'text_embedding':embedding}
    denoiser.eval();clean=denoiser(noisy,timesteps,y=y)
    null=denoiser(noisy,timesteps,y={**y,'uncond':True})
    changed_history=denoiser(noisy,timesteps,y={**y,'history_motion_normalized':history+.1})
    sensitivity={'text':float((clean-null).abs().max()),'history':float((clean-changed_history).abs().max())}
    assert min(sensitivity.values())>1e-7,sensitivity
    future_pred=vae.decode(clean.permute(1,0,2),history,nfuture=8);geo_ldm,_=geom_loss(future_pred,future)
    ldm_loss=torch.nn.functional.huber_loss(clean,z0.permute(1,0,2))+torch.nn.functional.huber_loss(future_pred,future)+geo_ldm
    assert torch.isfinite(ldm_loss);ldm_loss.backward();ldm_grad_count=finite_gradients(denoiser)
    assert all(p.grad is None for p in vae.parameters()) and all(p.grad is None for p in clip_model.parameters())
    # Two finite feature segments with exact last2 history reuse. Not a quality test.
    with torch.no_grad():
        def guided(x,t,**kwargs):
            # Published sampler emits int32 steps; positional-table indexing
            # in the pinned PyTorch1.13 environment requires int64 indices.
            cond=kwargs['y'];step=t.long();un=denoiser(x,step,y={**cond,'uncond':True});yes=denoiser(x,step,y=cond);return un+5*(yes-un)
        generated=[];current_history=history
        for i in range(2):
            conditions={'history_motion_normalized':current_history,'text_embedding':embedding if i==0 else embedding.flip(0)}
            sampled=diffusion.p_sample_loop(guided,(2,1,128),device=device,clip_denoised=False,model_kwargs={'y':conditions},progress=False)
            segment=vae.decode(sampled.permute(1,0,2),current_history,nfuture=8);assert torch.isfinite(segment).all()
            generated.append(segment);current_history=segment[:,-2:].clone()
        assert list(torch.cat(generated,1).shape)==[2,16,69]
    if args.device=='cuda':torch.cuda.synchronize()
    record={'sentinel':'TEXTOP_PAPER69_NUMERICAL_WITNESS_PASSED','seed':0,'device':torch.cuda.get_device_name() if args.device=='cuda' else 'cpu','python':sys.version.split()[0],'versions':versions,'spec_sha256':sha(args.spec),'source_hashes':{p.name:sha(p) for p in [pathlib.Path(__file__),pathlib.Path(__file__).with_name('paper69_runtime.py')]},'robot_xml_sha256':sha(spec['robot_xml']),'clip_weight_sha256':expected,'fixture_sha256':[sha(x['file']) for x in fixtures],'adapter':adapter,'codec_error':codec_error,'fk_mujoco_error':fk_error,'vae':{'output_shape':list(out.shape),'latent_shape':list(latent.shape),'loss':float(vae_loss),'finite_gradient_parameters':vae_grad_count},'ldm':{'latent_shape':list(clean.shape),'loss':float(ldm_loss),'finite_gradient_parameters':ldm_grad_count,'conditioning_sensitivity':sensitivity,'frozen_vae':True,'frozen_clip':True},'geometry_terms':{k:float(v) for k,v in terms.items()},'wrist_geometry_gradient_abs_mean':wrist_grad.detach().cpu().tolist(),'sampling':{'steps':5,'variance':'FIXED_LARGE; beta_t at nonterminal sampling timesteps','cfg':5,'segments':2,'future_frames_per_segment':8,'history_frames':2,'shape':[2,16,69]},'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,'training_performed':False,'optimizer_updates':0,'normalization':'identity for numerical validation only; no corpus statistics fitted','dataset_accepted':False,'semantic_quality_evaluated':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(record,indent=2));print(record['sentinel'],json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    try:main(a)
    except BaseException:
        a.output.parent.mkdir(parents=True,exist_ok=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f');a.output.with_name(a.output.stem+'.failure_'+stamp+'.txt').write_text(traceback.format_exc());raise
