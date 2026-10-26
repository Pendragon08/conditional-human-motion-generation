"""Async trained reference-history generator; no executed-pose conditioning or GT future."""
import argparse,json,os,pathlib,time,traceback,random
from online_quick_protocol import ROOT,sha,write,load_spec
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();s,sh=load_spec()
assert a.run.isidentifier() and set(os.sched_getaffinity(0))=={0} and os.environ['CUDA_VISIBLE_DEVICES']=='1'
out=ROOT/'online_private'/a.run;assert out.is_dir()
r={'status':'STARTING','spec_sha256':sh,'optimizer_updates':0,'posterior_future_used':False,'executed_pose_conditioning':False,'writers_closed':False}
try:
    import numpy as np,torch,clip
    from paper69_runtime import install_runtime,decode69,DifferentiableG1
    from generator_training_state import module_sha
    from tracker_analytic_fk import TrackerFK,reference_arrays
    base=json.loads((ROOT/'configs/paper69-env-spec.json').read_text());g=json.loads((ROOT/'configs/quick-generated-reference-spec.json').read_text())
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(s['generator_torch_fraction'])
    random.seed(g['seed']);np.random.seed(g['seed']);torch.manual_seed(g['seed']);torch.cuda.manual_seed_all(g['seed'])
    install_runtime(base['robot_xml'])
    from robotmdar.model.mld_vae import AutoMldVae
    from robotmdar.model.mld_denoiser import DenoiserTransformer
    from robotmdar.model.clip import encode_text
    from robotmdar.dtype.motion import get_zero_feature
    from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion,ModelMeanType,ModelVarType,LossType,get_named_beta_schedule
    vae=AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).cuda()
    denoiser=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,clip_dim=512,history_shape=(2,69),noise_shape=(1,128),cond_mask_prob=.1).cuda()
    for key,model,field in [('vae',vae,'vae'),('ldm',denoiser,'denoiser')]:
        m=g['trained_models'][key];assert sha(m['checkpoint']['path'])==m['checkpoint']['sha256']
        proof=json.loads(pathlib.Path(m['receipt']).read_text());assert sha(m['receipt'])==m['receipt_sha256']
        saved=torch.load(m['checkpoint']['path'],map_location='cpu');model.load_state_dict(saved[field],strict=True);del saved
        assert module_sha(model)==proof['trained_model_sha256' if key=='vae' else 'trained_denoiser_sha256']
        model.eval()
        for param in model.parameters():param.requires_grad=False
    corpus=pathlib.Path(g['corpus_dir']);meta=json.loads((corpus/'corpus_receipt.json').read_text())
    assert sha(corpus/'corpus_receipt.json')==g['corpus_receipt_sha256'] and sha(corpus/'meanstd.pkl')==meta['artifacts']['meanstd.pkl']['sha256']
    weight=next(pathlib.Path(x) for x in base['clip_weight_candidates'] if pathlib.Path(x).exists());assert sha(weight)==meta['clip_weight_sha256']
    encoder,_=clip.load(str(weight),device='cuda',jit=False);clip.model.convert_weights(encoder);encoder.eval()
    for param in encoder.parameters():param.requires_grad=False
    mean,std=torch.load(corpus/'meanstd.pkl',map_location='cpu');mean=mean.cuda();std=std.cuda()
    neutral=get_zero_feature().cuda().reshape(1,1,69).repeat(1,2,1);history=(neutral-mean)/std;features=[history.cpu().numpy()]
    diffusion=GaussianDiffusion(betas=get_named_beta_schedule('cosine',5),model_mean_type=ModelMeanType.START_X,model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE,rescale_timesteps=False)
    input_names=DifferentiableG1(base['robot_xml']).joint_names
    fk=TrackerFK(s['urdf'],s['joint_names'],s['body_names']);assert sha(s['urdf'])==s['urdf_sha256']
    index=0;records=[];embeddings={};deadline=time.monotonic()+s['job_timeout_seconds'];last_published=None
    r['status']='READY_WAITING_REQUEST';write(out/'generator_receipt.json',r)
    with torch.inference_mode():
        while not (out/'stop.json').exists():
            assert time.monotonic()<deadline,'Generator bounded wall timeout'
            rp=out/('request_%03d.json'%index)
            if not rp.exists():time.sleep(.003);continue
            req=json.loads(rp.read_text());received=time.monotonic();assert req['index']==index and req['text']==['walk forward','wave your right hand'][req['phase']]
            assert req['chunks']==(4 if index==0 else 1)
            if req['phase'] not in embeddings:embeddings[req['phase']]=encode_text(encoder,[req['text']])
            first_new=sum(x.shape[1] for x in features)
            for _ in range(req['chunks']):
                y={'text_embedding':embeddings[req['phase']],'history_motion_normalized':history}
                def guided(x,t,y):
                    u=denoiser(x,t.long(),y={**y,'uncond':True});c=denoiser(x,t.long(),y=y);return u+5*(c-u)
                t=time.monotonic();h=__import__('hashlib').sha256(history.cpu().numpy().tobytes()).hexdigest()
                latent=diffusion.p_sample_loop(guided,(1,1,128),device='cuda',clip_denoised=False,model_kwargs={'y':y},progress=False)
                pred=vae.decode(latent.permute(1,0,2),history,nfuture=8);torch.cuda.synchronize();assert torch.isfinite(pred).all()
                features.append(pred.cpu().numpy());history=pred[:,-2:]
                records.append({'primitive':len(records),'phase':req['phase'],'history_sha256':h,'started_monotonic':t,'finished_monotonic':time.monotonic()})
            normalized=np.concatenate(features,axis=1)[0];raw=torch.from_numpy(normalized).cuda()*std+mean
            motion=decode69(raw,abs_pose={'root_trans_offset':raw.new_tensor([0.,0.,.75]),'root_rot':raw.new_tensor([0.,0.,0.,1.])})
            q=torch.cat([motion['root_trans_offset'],motion['root_rot'][:,[3,0,1,2]],motion['dof']],dim=-1).cpu().numpy()
            assert np.isfinite(q).all() and len(q)<s['buffer_capacity']
            if last_published is not None:assert np.array_equal(q[:len(last_published)],last_published),'Global decoding changed previously published reference poses'
            last_published=q.copy();arrays=reference_arrays(fk,q,input_names)
            target=out/('reference_%03d.private.npz'%index);temp=target.with_suffix('.partial')
            with temp.open('xb') as f:np.savez(f,**arrays,qpos_29=q,normalized_features=normalized)
            temp.replace(target)
            result=dict(req,received_monotonic=received,published_monotonic=time.monotonic(),npz=target.name,sha256=sha(target),frames=len(q),
                available_exclusive=len(q)-1,first_new_frame=first_new,last_frame_velocity_provisional=True)
            write(out/('response_%03d.json'%index),result);index+=1
            r.update(status='GENERATING_ON_DEMAND',responses=index,produced_frames=len(q),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated());write(out/'generator_receipt.json',r)
    write(out/'generation_records.private.json',records)
    r.update(status='ONLINE_GENERATOR_CLOSED',writers_closed=True,responses=index,produced_frames=len(q),records_sha256=sha(out/'generation_records.private.json'),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
except BaseException as e:r.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
finally:write(out/'generator_receipt.json',r)
