"""Generate task-specific Pick candidates from the final trained diffusion prior.

The scene and spatial-gradient schedule are documented project adaptations. This
script only generates candidates: it never admits references or starts PPO.
"""
import json
import argparse
import sys
import time
from pathlib import Path
from types import MethodType
import numpy as np
import torch

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'stage2_prior29_20261103';S=R/'stage3_pick29_20261103'
sys.path[:0]=[str(D/'src'),str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage2_cmdm29 import build_model,restore_state,sha
from stage2_robot_prior import setup,preflight,diffusion_for
from stage3_pick29_core import SCENE,SCENE_SHA,BODY_JOINTS
from g1_fk import URDFFK


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir
    output.mkdir(exist_ok=False)
    cfg=json.loads((D/'config.json').read_text());cfg['gpu_memory_fraction']=.05
    _,device=setup(cfg)
    data,stats,provenance=preflight(cfg)
    checkpoint=D/'runs/grab_omni29_mdm_baseline_v1/step_000010000.pt'
    expected='109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9'
    assert sha(checkpoint)==expected
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['provenance']==provenance and saved['cursor']['step']==10000
    model,_=build_model(cfg);restore_state(model,saved['model']);model.to(device);model.eval()
    diffusion=diffusion_for(stats)
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
    fk=URDFFK(urdf);q=np.zeros((1,29))
    for side in ('left','right'):
        for part,value in [('hip_pitch',-.2),('knee',.42),('ankle_pitch',-.23),('elbow',.87),('shoulder_pitch',.35)]:
            q[0,BODY_JOINTS.index(side+'_'+part+'_joint')]=value
    q[0,BODY_JOINTS.index('left_shoulder_roll_joint')]=.16
    q[0,BODY_JOINTS.index('right_shoulder_roll_joint')]=-.16
    initial=fk.forward(np.array([[0,0,.84]]),np.array([[0,0,0,1.]]),q,BODY_JOINTS)[0]
    frames=120;batch=2
    world=np.broadcast_to(initial,(frames,22,3)).copy()
    grasp=np.array([.35,-.20,.94]);lift=np.array([.35,-.20,1.10])
    # Contact at 2s, lift by 4s; two candidates share the task but have different DDPM noise.
    for k in range(frames):
        t=k/20
        world[k,21]=(initial[21]*(1-t/2)+grasp*(t/2) if t<2 else
                     grasp*(1-(t-2)/2)+lift*((t-2)/2) if t<4 else lift)
    # Canonical X (left), Y (up), Z (forward) -> simulator Y, Z, X.
    rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    canonical=world@rotation
    target=torch.tensor(canonical,device=device,dtype=torch.float32)[None].repeat(batch,1,1,1)
    mask=torch.zeros(batch,frames,22,device=device,dtype=torch.bool)
    mask[:,:,21]=True
    mask[:,:,0]=True
    mask[:,::5,4:6]=True
    spatial_mean=stats['spatial_mean'].to(device).reshape(22,3)
    spatial_std=stats['spatial_std'].to(device).reshape(22,3)
    hint=((target-spatial_mean)/spatial_std)*mask[...,None]
    prompt='Pick up the object from the table.'
    with torch.no_grad():embedding=model.encode_text([prompt]*batch)
    y={'text':[prompt]*batch,'text_embedding':embedding,'hint':hint.reshape(batch,frames,66),
       'hint_mask':mask,'mask':torch.ones(batch,1,1,frames,device=device,dtype=torch.bool),
       'lengths':torch.full((batch,),frames,device=device,dtype=torch.long),
       'scale':torch.full((batch,),2.5,device=device)}
    mean=stats['feature_mean'].to(device);std=stats['feature_std'].to(device)
    count=(mask.sum((1,2))*3).clamp_min(1)
    guide_calls=0

    def per_sample_loss(x):
        features=x[:,:,0].permute(0,2,1)*std+mean
        points=recover_from_ric(features,22)
        return (((points-target)*mask[...,None]).square().sum((1,2,3)))/count

    def guide(self,x,t,model_kwargs=None,**kwargs):
        nonlocal guide_calls
        result=x.detach()
        # MSE gradient guidance with backtracking prevents an oversized tau increasing error.
        # Its tuned schedule is ours; the paper does not publish these exact hyperparameters.
        variance=torch.as_tensor(self.posterior_log_variance_clipped,device=device,dtype=x.dtype)[t].exp().clamp_min(.01)
        for _ in range(2):
            with torch.enable_grad():
                work=result.detach().requires_grad_(True)
                losses=per_sample_loss(work)
                gradient=torch.autograd.grad(losses.sum(),work)[0]
            gradient[...,0]=0
            alpha=variance*1000.
            good=torch.zeros(batch,device=device,dtype=torch.bool)
            original=result
            for attempt in range(6):
                trial=original-(gradient*alpha[:,None,None,None]).clamp(-.25,.25)
                with torch.no_grad():new_loss=per_sample_loss(trial)
                accepted=torch.isfinite(new_loss)&(new_loss<=losses.detach())&~good
                result=torch.where(accepted[:,None,None,None],trial,result)
                good|=accepted
                if bool(good.all()):break
                alpha=torch.where(good,alpha,alpha*.5)
        guide_calls+=1
        if guide_calls%100==0:
            status={'status':'sampling','denoising_steps_completed':guide_calls,'total':1000,'candidates':batch}
            (output/'progress.json').write_text(json.dumps(status))
        return result.detach()

    diffusion.guide=MethodType(guide,diffusion)
    began=time.monotonic();torch.manual_seed(20261103)
    with torch.no_grad():
        normalized=diffusion.p_sample_loop(ClassifierFreeSampleModel(model),(batch,263,1,frames),
            clip_denoised=False,model_kwargs={'y':y},device=device,progress=False)
        features=normalized[:,:,0].permute(0,2,1)*std+mean
        points=recover_from_ric(features,22)
    assert torch.isfinite(features).all() and torch.isfinite(points).all()
    results=[]
    for k in range(batch):
        p=output/('sample_%d.npz'%k)
        metadata={'caption':prompt,'fps':20,'frames':frames,'checkpoint_sha256':expected,
                  'scene_sha256':SCENE_SHA,'grasp_time_s':2.,'physical_acceptance':False,
                  'guidance':'MSE, two backtracked steps, tau1000, normalized coordinate update cap .25',
                  'canonical_to_world_rotation':rotation.tolist(),'canonical_to_world_translation':[0.,0.,0.],
                  'requested_constraint_points':[0,4,5,21],'note':'task-specific candidate, not an accepted RL reference'}
        with p.open('xb') as f:
            np.savez_compressed(f,features=features[k:k+1].cpu().numpy(),canonical_points=points[k:k+1].cpu().numpy(),
                target_points=target[k].cpu().numpy(),target_mask=mask[k:k+1].cpu().numpy(),
                metadata_utf8=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8))
        error=(points[k,:,21]-target[k,:,21]).norm(dim=-1)
        results.append({'path':str(p),'sha256':sha(p),'wrist_error_at_grasp_m':float(error[40]),
                        'max_wrist_error_m':float(error.max()),'mean_wrist_error_m':float(error.mean())})
    report={'status':'candidates_generated','candidates':results,'checkpoint_sha256':expected,
            'scene_sha256':SCENE_SHA,'scene':SCENE,'elapsed_s':time.monotonic()-began,
            'generated_reference_admission':False,'PPO_started':False,
            'scope':'task-conditioned 29DoF adaptation; 263D output still needs robot-space fitting and task filters'}
    with (output/'report.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
