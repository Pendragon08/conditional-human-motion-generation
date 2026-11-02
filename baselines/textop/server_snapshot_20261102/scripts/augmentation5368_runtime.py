"""Frozen100k continuous TRAIN text-stream inference, pending real witnesses.

Import is standard-library only. Construction is allowed only in a separately
accepted, common-model-lock GPU job. No optimizer, teacher, future posterior,
heldout motion, clipping or zero-text fallback is used.
"""
import hashlib
import pathlib


class Runtime:
    def __init__(self,accepted,train_index):
        import sys
        import numpy as np
        import torch as t
        from filtered15000_vae_completion import ROOT,read,sha,owned
        from generator_training_state import module_sha
        self.np,self.t=np,t;self.sha,self.owned=sha,owned;self.module_sha=module_sha
        self.source=accepted['export']['stopped']['source_spec']
        self.index={x['canonical_source']:x for x in train_index}
        assert len(self.index)==len(train_index)==5361
        assert all(x['original_split']=='train' and not x['mirrored'] for x in train_index)
        assert t.cuda.device_count()==1
        self.device=t.device('cuda:0');t.set_num_threads(4)
        t.cuda.set_per_process_memory_fraction(.65,device=0)
        assert t.cuda.mem_get_info()[0]>=20*1024**3
        sys.path.insert(0,str(ROOT/'upstream/TextOpRobotMDAR'))
        from robotmdar.model.mld_vae import AutoMldVae
        from robotmdar.model.mld_denoiser import DenoiserTransformer
        from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion,ModelMeanType,ModelVarType,LossType,get_named_beta_schedule
        import robotmdar.diffusion.gaussian_diffusion as diffusion_module
        extract=diffusion_module._extract_into_tensor
        def extract_int64(array,timesteps,shape):return extract(array,timesteps.long(),shape)
        diffusion_module._extract_into_tensor=extract_int64
        import paper69_runtime as codec
        self.codec=codec
        self.vae=AutoMldVae(nfeats=69,latent_dim=[1,128],h_dim=512,ff_size=1024,num_layers=9,num_heads=4).cpu().eval()
        self.model=DenoiserTransformer(h_dim=512,ff_size=1024,num_layers=8,num_heads=4,
            clip_dim=512,history_shape=(2,69),noise_shape=(1,128),cond_mask_prob=.1).cpu().eval()
        ld=accepted['export']['inference'];vae=self.source['vae_inference']
        assert sha(ld['path'])==ld['sha256'] and sha(vae['path'])==vae['sha256']
        lp=t.load(ld['path'],map_location='cpu');vp=t.load(vae['path'],map_location='cpu')
        assert lp['schema']=='textop.ldm100k_prefix_inference.v1' and vp['schema']=='textop.filtered15000_vae_inference.v1'
        assert lp['completed_primitive_attempts']==vp['completed_primitive_attempts']==100000
        assert lp['source_full_state_sha256']==accepted['export']['stopped']['checkpoint']['sha256']
        assert vp['source_full_state_sha256']==accepted['export']['stopped']['policy']['vae_checkpoint_sha256']
        assert lp['architecture']==self.source['ldm'] and lp['vae_inference_sha256']==vae['sha256']
        assert lp['source_numeric_binding_sha256']==accepted['export']['stopped']['policy']['original_runtime_binding_sha256']
        assert lp['corpus_receipt_sha256']==vp['corpus_receipt_sha256']==self.source['corpus_receipt_sha256']
        assert vp['data_acceptance_sha256']==self.source['data_acceptance_sha256']
        self.model.load_state_dict(lp['denoiser'],strict=True);self.vae.load_state_dict(vp['vae'],strict=True)
        self.model_hash,self.vae_hash=module_sha(self.model),module_sha(self.vae)
        assert self.model_hash==lp['trained_denoiser_sha256'] and self.vae_hash==vp['trained_model_sha256']
        for model in [self.model,self.vae]:
            assert all(t.isfinite(x).all() for x in model.state_dict().values())
            for parameter in model.parameters():parameter.requires_grad_(False)
            model.to(self.device).eval()
        self.diffusion=GaussianDiffusion(betas=get_named_beta_schedule('cosine',5),model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,loss_type=LossType.MSE,rescale_timesteps=False)
        self.view_root=pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
        self.view=owned(self.source['corpus_dir'],self.view_root)
        assert sha(self.view/'view_receipt.json')==self.source['corpus_receipt_sha256']
        self.meta=read(self.view/'view_receipt.json')
        for name in ['meanstd.pkl','train_text_embed.pkl']:
            path=owned(self.view/name,self.view_root);item=self.meta['artifacts'][name]
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        self.mean,self.std=t.load(self.view/'meanstd.pkl',map_location='cpu')
        assert self.mean.shape==self.std.shape==(69,) and self.mean.dtype==self.std.dtype==t.float32
        assert t.isfinite(self.mean).all() and t.isfinite(self.std).all() and (self.std>0).all()
        self.embeddings=t.load(self.view/'train_text_embed.pkl',map_location='cpu')
        self.used_sources={};self.inference=[ld,vae]

    def initial_history(self,clip):
        np,t=self.np,self.t;row=self.index[clip['initial_source']]
        assert row['output_file']==clip['initial_source_path'] and row['output_sha256']==clip['initial_source_sha256']
        path=pathlib.Path(row['output_file'])
        roots=[x for x in [pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024'),self.view_root] if x in path.parents]
        assert len(roots)==1;self.owned(path,roots[0]);assert self.sha(path)==row['output_sha256']
        with np.load(path,allow_pickle=False) as motion:
            q=np.asarray(motion['qpos_29'],dtype=np.float32);c=np.asarray(motion['contact'],dtype=np.float32)
        assert q.shape==(row['frames50'],36) and c.shape==(len(q),2) and len(q)>=3
        assert np.isfinite(q).all() and np.isfinite(c).all() and ((c>=0)&(c<=1)).all()
        assert np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-3)
        # Only these three seed poses enter the generator. The remainder is
        # checked for source integrity but supplies no future conditioning.
        q,c=q[:3],c[:3]
        features,_=self.codec.encode69(dict(root_trans_offset=t.from_numpy(q[:,:3]),
            root_rot=t.from_numpy(q[:,[4,5,6,3]]),dof=t.from_numpy(q[:,7:]),contact_mask=t.from_numpy(c)))
        normalized=(features-self.mean)/self.std
        assert normalized.shape==(2,69) and t.isfinite(normalized).all()
        self.used_sources[str(path)]=row['output_sha256']
        return normalized.unsqueeze(0).to(self.device),q[0].copy()

    def generate(self,clip):
        np,t=self.np,self.t
        intervals=clip['target_control_intervals'];calls=clip['primitive_calls']
        assert clip['target_pose_frames']==intervals+1 and calls==(intervals+8)//8
        initial,pose=self.initial_history(clip);history=initial.clone();generated=[]
        t.manual_seed(clip['generation_seed']);t.cuda.manual_seed_all(clip['generation_seed'])
        chain=hashlib.sha256();command_index=0;actual_forwards=0;command_calls=[0]*len(clip['commands'])
        with t.no_grad():
            for primitive in range(calls):
                while primitive>=clip['commands'][command_index]['end_primitive_exclusive']:command_index+=1
                command=clip['commands'][command_index]
                assert command['start_primitive']<=primitive<command['end_primitive_exclusive']
                assert command['text'] in self.embeddings,'No missing text or zero-vector replacement permitted'
                text=self.embeddings[command['text']].float().reshape(1,512).to(self.device)
                assert t.isfinite(text).all()
                before=t.cuda.get_rng_state().cpu().numpy().tobytes()
                noise=t.randn((1,1,128),device=self.device);chain.update(before);chain.update(noise.cpu().numpy().tobytes())
                forwards=[0]
                def guided(x,timestep,y):
                    forwards[0]+=2
                    un=self.model(x,timestep.long(),y=dict(y,uncond=True))
                    yes=self.model(x,timestep.long(),y=dict(y,uncond=False))
                    return un+5*(yes-un)
                sample=self.diffusion.p_sample_loop(guided,(1,1,128),noise=noise,device=self.device,
                    clip_denoised=False,progress=False,model_kwargs={'y':dict(text_embedding=text,history_motion_normalized=history)})
                future=self.vae.decode(sample.permute(1,0,2),history,nfuture=8)
                assert future.shape==(1,8,69) and t.isfinite(future).all() and forwards[0]==10
                generated.append(future);history=future[:,-2:];actual_forwards+=forwards[0];command_calls[command_index]+=1
                chain.update(future.cpu().numpy().tobytes());chain.update(t.cuda.get_rng_state().cpu().numpy().tobytes())
            full=t.cat(generated,dim=1)
            features=t.cat([initial,full],dim=1)*self.std.to(self.device)+self.mean.to(self.device)
            absolute=dict(root_trans_offset=t.as_tensor(pose[:3],device=self.device),root_rot=t.as_tensor(pose[[4,5,6,3]],device=self.device))
            motion=self.codec.decode69(features,absolute);q,c=self.codec.to_qpos69(motion)
            q=q[0,2:2+intervals+1];c=c[0,2:2+intervals+1]
            assert q.shape==(intervals+1,36) and c.shape==(intervals+1,2)
            assert t.isfinite(q).all() and t.isfinite(c).all()
            assert t.allclose(q[:,3:7].norm(dim=-1),t.ones(len(q),device=self.device),atol=1e-3)
            arrays=dict(qpos_29=q.cpu().numpy().astype(np.float32),contact=c.cpu().numpy().astype(np.float32),fps=np.array(50,dtype=np.int64))
        assert sum(command_calls)==calls and all(n>0 for n in command_calls)
        return arrays,dict(clip_index=clip['clip_index'],seed=clip['generation_seed'],control_intervals=intervals,
            actual_pose_frames=len(q),actual_primitive_calls=calls,actual_denoiser_forward_calls=actual_forwards,
            command_native_call_counts=command_calls,noise_output_rng_chain_sha256=chain.hexdigest(),
            generated_terminal_pose_included=True,seed_history_in_duration=False,generated_tail_excluded=full.shape[1]-intervals-1,
            conditioning='accepted TRAIN text CFG5, initial two past features then generated-only history across text boundaries',
            future_posterior_encoded=False,parameter_updates=0,teacher_loaded=False,contact_or_joint_clamping=False)

    def verify_unchanged(self):
        assert self.module_sha(self.model)==self.model_hash and self.module_sha(self.vae)==self.vae_hash
        for x in self.inference:assert self.sha(x['path'])==x['sha256']
        for name in ['meanstd.pkl','train_text_embed.pkl']:assert self.sha(self.view/name)==self.meta['artifacts'][name]['sha256']
        for path,value in self.used_sources.items():assert self.sha(path)==value
