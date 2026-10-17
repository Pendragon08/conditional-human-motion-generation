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
                'paired_training_updates':state['optimizer_updates'],'text':text}
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
        self.model.eval().requires_grad_(False)
        self.diffusion=X0Diffusion(self.model,betas).to(device).eval()
        self.provenance['parameters']=sum(p.numel() for p in self.model.parameters())
        self.provenance['normalization_sha256']=hashlib.sha256(Path(normalization).read_bytes()).hexdigest()

    @torch.inference_mode()
    def __call__(self,prefix,suffix,new_frames,final,attempt):
        # A short retained context bounds inference cost; the external committed
        # prefix remains exact and all accepted actual states are retained there.
        context=prefix.states[-50:]
        if len(context)<2:context=np.repeat(context,2,axis=0)
        f,meta=self.codec.encode(context,50.,canonicalize=False)
        # The boundary velocity comes from actual execution, not an extrapolated
        # difference from the preceding reference candidate.
        f[-1,:3]=context[-1,10:13];f[-1,3:6]=context[-1,7:10]
        length=len(f)+new_frames
        known=np.zeros((length,262),np.float32);mask=np.zeros_like(known,dtype=bool)
        known[:len(f)]=f;mask[:len(f)]=True
        if final:
            terminal=np.repeat(suffix.states[-1:],2,axis=0)
            sf,_=self.codec.encode(terminal,50.,canonicalize=False)
            known[-len(suffix):]=sf[-1];mask[-len(suffix):]=True
        values=torch.from_numpy(self.normalizer.transform(known))[None].to(self.device)
        known_mask=torch.from_numpy(mask)[None].to(self.device)
        generator=torch.Generator(device=self.device).manual_seed(self.seed+len(self.calls))
        raw=self.diffusion.sample((1,length,262),self.tokens,self.text_valid,scale=1.,
            known_values=values,known_mask=known_mask,generator=generator)[0].cpu().numpy()
        raw=self.normalizer.inverse(raw)
        decoded,_,projection=self.codec.decode(raw,meta,project=True,restore_world=False)
        continuation=decoded[len(f):].copy()
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
        candidate=prefix.append(Motion(continuation,target,50.,prefix.state_schema,prefix.body_names))
        index=len(self.calls)
        np.savez_compressed(self.output/f'proposal_{index:03d}.npz',raw_features_262=raw,
            decoded_native_before_bounds=before,candidate_native=candidate.states,
            candidate_body_positions=candidate.body_positions,conditioned_actual_prefix=prefix.states)
        self.calls.append({'attempt':attempt,'final':final,'candidate_sha256':candidate.digest(),
            'input_prefix_sha256':prefix.digest(),'input_prefix_frames':len(prefix),
            'new_frames':new_frames,'network_context_frames':len(f),'projection':projection,
            'additional_bound_changed_values':int(np.count_nonzero(before!=continuation))})
        return candidate
