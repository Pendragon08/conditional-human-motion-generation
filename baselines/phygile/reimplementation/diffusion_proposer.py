"""Connect the reconstructed diffusion model to actual robot-native references.

The diagnostic backend has random generator weights and zero text conditioning.
It tests wiring only. The checkpoint backend accepts only provenance-bearing
training outputs and a real local DistilBERT encoder. Neither backend fabricates
paired training data, captions or learned generation quality.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from codec262 import SCHEMA
from features import BlockNormalizer
from model import ModelConfig,MotionDenoiser
from diffusion import X0Diffusion
from physics_prefix import Motion
from motion_clock import (validate_feature_contract, prepare_context, generation_grid,
                          resample_native, terminal_features)


class DiffusionProposer:
    def __init__(self,codec,normalization,device,seed,output,checkpoint=None,text_encoder=None,text=None):
        self.codec=codec;self.device=device;self.seed=seed;self.calls=[];self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        with np.load(normalization,allow_pickle=False) as d:
            self.normalizer=BlockNormalizer(d['mean'],d['std'])
        torch.manual_seed(seed)
        self.trained=checkpoint is not None
        if checkpoint:
            checkpoint=Path(checkpoint)
            state=torch.load(checkpoint,map_location=device,weights_only=False)
            config=state['config']
            if state['purpose']!='research_training' or state['optimizer_updates']<1:
                raise ValueError('Diagnostic/synthetic checkpoint cannot be labeled a trained language generator')
            self.feature_fps=validate_feature_contract(state.get('feature_contract'))
            self.canonicalize=True
            self.cfg_scale=2.5
            if state.get('codec_schema')!=SCHEMA or state.get('codec_skeleton_sha256')!=hashlib.sha256(codec.path.read_bytes()).hexdigest():
                raise ValueError('Checkpoint does not identify this explicit codec schema and skeleton')
            if not np.array_equal(state['normalization_mean'],self.normalizer.mean) or not np.array_equal(
                    state['normalization_std'],self.normalizer.std):
                raise ValueError('Checkpoint normalization mismatch')
            self.model=MotionDenoiser(ModelConfig(**config['model'])).to(device)
            self.model.load_state_dict(state['model_state_dict'],strict=True)
            schedule=config['schedule'];betas=torch.linspace(schedule['start'],schedule['end'],schedule['steps'])
            if text_encoder is None or not text:raise ValueError('Actual text encoder and prompt required')
            text_record=json.loads((Path(text_encoder)/'download.json').read_text())
            if text_record['revision']!=state['text_encoder_revision']:raise ValueError('Text encoder revision mismatch')
            for item in text_record['files']:
                if hashlib.sha256((Path(text_encoder)/item['name']).read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Text encoder file mismatch')
            from text import FrozenDistilBert
            self.encoder=FrozenDistilBert(text_encoder).to(device)
            self.tokens,self.text_valid=self.encoder([text])
            self.provenance={'backend':'diffusion_checkpoint','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'paired_training_updates':state['optimizer_updates'],'text':text,
                'feature_contract':state['feature_contract']}
        else:
            self.model=MotionDenoiser(ModelConfig()).to(device)
            # A chosen eight-step schedule keeps the engineering witness bounded;
            # it is not an inferred author training/inference schedule.
            betas=torch.linspace(.0001,.02,8)
            self.tokens=torch.zeros((1,1,768),device=device)
            self.text_valid=torch.ones((1,1),dtype=torch.bool,device=device)
            self.provenance={'backend':'untrained_diffusion_diagnostic','trained':False,
                'text_conditioning':'zero/unconditional; no text-to-motion claim',
                'schedule':'chosen linear 8-step engineering diagnostic'}
            # Preserve the explicit historical diagnostic's own clock/frame;
            # it is never inferred from the real paired training normalizer.
            self.feature_fps=50.;self.canonicalize=False;self.cfg_scale=1.
        self.model.eval().requires_grad_(False)
        self.diffusion=X0Diffusion(self.model,betas).to(device).eval()
        self.provenance['parameters']=sum(p.numel() for p in self.model.parameters())
        self.provenance['normalization_sha256']=hashlib.sha256(Path(normalization).read_bytes()).hexdigest()
        self.provenance.update(feature_fps=self.feature_fps,cfg_scale=self.cfg_scale,
            control_clock='explicit Motion.fps; no feature/control rate alias',
            terminal_root_pose='explicit output projection, not guaranteed by velocity inpainting')

    @torch.inference_mode()
    def __call__(self,prefix,suffix,new_frames,final,attempt):
        # A short retained context bounds inference cost; the external committed
        # prefix remains exact and all accepted actual states are retained there.
        prefix.compatible(suffix)
        future_times,control_times=generation_grid(new_frames,prefix.fps,self.feature_fps,final)
        f,meta,clock=prepare_context(self.codec,prefix,self.feature_fps,canonicalize=self.canonicalize)
        length=len(f)+len(future_times)-1
        known=np.zeros((length,262),np.float32);mask=np.zeros_like(known,dtype=bool)
        known[:len(f)]=f;mask[:len(f)]=True
        if final:
            known[-1]=terminal_features(self.codec,suffix,meta);mask[-1]=True
        values=torch.from_numpy(self.normalizer.transform(known))[None].to(self.device)
        known_mask=torch.from_numpy(mask)[None].to(self.device)
        generator=torch.Generator(device=self.device).manual_seed(self.seed+len(self.calls))
        raw=self.diffusion.sample((1,length,262),self.tokens,self.text_valid,scale=self.cfg_scale,
            known_values=values,known_mask=known_mask,generator=generator)[0].cpu().numpy()
        raw=self.normalizer.inverse(raw)
        decoded,_,projection=self.codec.decode(raw,meta,project=True,restore_world=True)
        # The boundary belongs to actual execution, including roundoff-sensitive
        # pose and velocity values. Sample the future on fixed control times.
        future=decoded[len(f)-1:].copy();future[0]=prefix.states[-1]
        continuation=resample_native(future,future_times,control_times)[1:]
        before=continuation.copy()
        # Explicit bounds are a diagnostic safety projection, not evidence of
        # physical feasibility. Raw network features and pre-bound states are kept.
        continuation[:,0:2]=np.clip(continuation[:,0:2],prefix.states[-1,0:2]-.5,prefix.states[-1,0:2]+.5)
        continuation[:,2]=np.clip(continuation[:,2],.45,1.1)
        continuation[:,7:13]=np.clip(continuation[:,7:13],-10,10)
        continuation[:,42:71]=np.clip(continuation[:,42:71],-30,30)
        if final:continuation[-len(suffix):]=suffix.states
        positions,_=self.codec.fk(continuation[:,:3],continuation[:,3:7],continuation[:,13:42])
        target=np.stack([positions[n] for n in prefix.body_names],1)
        if final:target[-len(suffix):]=suffix.body_positions
        candidate=prefix.append(Motion(continuation,target,prefix.fps,prefix.state_schema,prefix.body_names))
        index=len(self.calls)
        np.savez_compressed(self.output/f'proposal_{index:03d}.npz',raw_features_262=raw,
            codec_metadata_json=np.asarray(json.dumps(meta)),conditioned_features_262=known,
            conditioned_feature_mask=mask,
            decoded_native_before_bounds=before,candidate_native=candidate.states,
            feature_future_sample_times_s=future_times,control_future_sample_times_s=control_times,
            candidate_body_positions=candidate.body_positions,conditioned_actual_prefix=prefix.states)
        self.calls.append({'attempt':attempt,'final':final,'candidate_sha256':candidate.digest(),
            'input_prefix_sha256':prefix.digest(),'input_prefix_frames':len(prefix),
            'new_frames':new_frames,'network_context_frames':len(f),'projection':projection,
            'clock':{**clock,'generated_future_intervals':len(future_times)-1,
                'requested_duration_s':float(control_times[-1]),
                'generated_last_time_s':float(future_times[-1]),
                'unused_generated_tail_s':float(future_times[-1]-control_times[-1]),
                'time_stretched':False,'extrapolation':False,'cfg_scale':self.cfg_scale},
            'additional_bound_changed_values':int(np.count_nonzero(before!=continuation))})
        return candidate
