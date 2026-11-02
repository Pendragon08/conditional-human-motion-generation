"""Explicit paper LDM history settings around the released manager and DDPM.

The native manager caps rollout at1 and has a stage>=2 static-pose edge case.
The paper states a maximum LDM rollout probability of.8. Keep these differences
visible rather than silently attributing the adjusted settings to the release.
"""
def rollout_probability(step,stage_idx,stages,enabled=True):
 assert len(stages)==4 and stages==[50000,50000,50000,150000]
 assert isinstance(step,int) and step>=0
 if not enabled or stage_idx<1:return 0.0
 return max(0.0,min(.8,(step-stages[0])/float(stages[1])))

def make_paper_manager(base):
 import torch
 class PaperDARManager(base):
  def rollout_probability(self):return rollout_probability(self.step,self.stage_idx,self.stages,self.use_rollout)
  def should_rollout(self):
   prob=self.rollout_probability()
   # Preserve native early-stage RNG behavior: no random draw before stage1.
   self.last_rollout_selected=False if not self.use_rollout or self.stage_idx<1 else bool(torch.rand(1).item()<prob)
   return self.last_rollout_selected
  def should_static_pose(self):
   if not self.use_static_pose:return False
   return super().should_static_pose()
 return PaperDARManager

def full_history_candidate(runtime,history,prediction,text,latent_shape,use_full):
 import torch
 if not use_full:return torch.cat([history,prediction],dim=1).detach()
 assert latent_shape==(history.shape[0],1,128)
 # Native train_dar.py full-sample branch: fresh five-step DDPM, without CFG.
 # CFG5 belongs to generation/evaluation; conditioning masking remains native.
 with torch.no_grad():
  sample=runtime.diffusion.p_sample_loop(runtime.model,latent_shape,device=history.device,
   clip_denoised=False,model_kwargs={'y':{'text_embedding':text,'history_motion_normalized':history}},
   skip_timesteps=0,init_image=None,progress=False,dump_steps=None,noise=None,const_noise=False)
  assert torch.is_tensor(sample) and sample.shape==latent_shape and torch.isfinite(sample).all()
  future=runtime.vae.decode(sample.permute(1,0,2),history,nfuture=8)
  assert future.shape==(history.shape[0],8,69) and torch.isfinite(future).all()
  return torch.cat([history,future],dim=1).detach()
