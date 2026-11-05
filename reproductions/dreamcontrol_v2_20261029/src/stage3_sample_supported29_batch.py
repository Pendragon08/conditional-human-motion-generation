"""Generate three-task position-conditioned candidates from the final trained diffusion prior.

The scene and spatial-gradient schedule are documented project adaptations. This
script only generates candidates: it never admits references or starts PPO.
"""
import json
import hashlib
import shutil
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
from stage3_supported_task_conditions import build as task_conditions, TASKS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--count',type=int,required=True)
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--start-index',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261104)
    args=parser.parse_args()
    assert 1<=args.count<=1000 and 1<=args.batch_size<=4 and args.start_index>=0
    root_output=args.output_dir
    root_output.mkdir(exist_ok=False)
    records=root_output/'records.jsonl';records.open('x').close()
    def root_status(**value):
        temp=root_output/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(root_output/'status.json')
    root_status(status='initializing',requested_candidates=args.count,accepted_references=0,PPO_started=False)
    output=root_output
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
    initial_all,initial_rot=fk.forward(np.array([[0,0,.84]]),np.array([[0,0,0,1.]]),q,BODY_JOINTS,return_rotations=True)
    initial=initial_all[0]
    all_results=[]
    root_status(status='sampling',requested_candidates=args.count,generated_candidates=0,accepted_references=0,PPO_started=False)
    for start in range(args.start_index,args.start_index+args.count,args.batch_size):
        assert shutil.disk_usage('/data').free>8*1024**3+356*1024**2+100*1024**2,'sampling disk protection'
        output=root_output/('batch_'+str(start).zfill(6));output.mkdir(exist_ok=False)
        frames=180;batch=min(args.batch_size,args.start_index+args.count-start)
        candidate_ids=list(range(start,start+batch))
        conditions=task_conditions(args.task,candidate_ids,args.seed,initial,initial_rot[0])
        world=conditions['world'];object_positions=conditions['object_initial']
        scene=conditions['scene'];scene_sha=conditions['scene_sha256']
        # Canonical X (left), Y (up), Z (forward) -> simulator Y, Z, X.
        rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        canonical=world@rotation
        target=torch.tensor(canonical,device=device,dtype=torch.float32)
        mask=torch.tensor(conditions['mask'],device=device,dtype=torch.bool)
        spatial_mean=stats['spatial_mean'].to(device).reshape(22,3)
        spatial_std=stats['spatial_std'].to(device).reshape(22,3)
        hint=((target-spatial_mean)/spatial_std)*mask[...,None]
        prompt=conditions['prompt']
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
        began=time.monotonic();torch.manual_seed(args.seed+1048576+start)
        noise=torch.cat([torch.randn((1,263,1,frames),generator=torch.Generator(device=device).manual_seed(args.seed+i),
                                    device=device) for i in candidate_ids])
        with torch.no_grad():
            normalized=diffusion.p_sample_loop(ClassifierFreeSampleModel(model),(batch,263,1,frames),
                noise=noise,clip_denoised=False,model_kwargs={'y':y},device=device,progress=False)
            features=normalized[:,:,0].permute(0,2,1)*std+mean
            points=recover_from_ric(features,22)
        assert torch.isfinite(features).all() and torch.isfinite(points).all()
        results=[]
        for k in range(batch):
            p=output/('sample_%d.npz'%k)
            metadata={'caption':prompt,'fps':20,'frames':frames,'checkpoint_sha256':expected,
                      'candidate_id':candidate_ids[k],'initial_noise_seed':args.seed+candidate_ids[k],
                      'denoising_batch_seed':args.seed+1048576+start,
                      'initial_noise_sha256':hashlib.sha256(noise[k].cpu().numpy().tobytes()).hexdigest(),
                      'feature_content_sha256':hashlib.sha256(features[k].cpu().numpy().tobytes()).hexdigest(),
                      'task_object_position':object_positions[k].tolist(),
                      'object_xy_distribution':'uniform+-0.025m project tabletop pilot, not a published full distribution',
                      'scene_sha256':scene_sha,'grasp_time_s':2.,'physical_acceptance':False,
                      'guidance':'MSE, two backtracked steps, tau1000, normalized coordinate update cap .25',
                      'canonical_to_world_rotation':rotation.tolist(),'canonical_to_world_translation':[0.,0.,0.],
                      'requested_constraint_points':([0,10,11,21] if args.task=='pour' else [0,10,11,20,21]),
                      'task':args.task,'task_plan':conditions['plans'][k],
                      'task_conditions_source_sha256':sha(Path(__file__).with_name('stage3_supported_task_conditions.py')),'note':'task-specific candidate, not an accepted RL reference'}
            with p.open('xb') as f:
                np.savez_compressed(f,features=features[k:k+1].cpu().numpy(),canonical_points=points[k:k+1].cpu().numpy(),
                    target_points=target[k].cpu().numpy(),target_mask=mask[k:k+1].cpu().numpy(),
                    desired_object_positions=conditions['desired_object_positions'][k],
                    desired_wrist_quat_wxyz=conditions['desired_wrist_quat_wxyz'][k],planned_hand_closed=conditions['planned_hand_closed'][k],
                    metadata_utf8=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8))
            error=(points[k,:,21]-target[k,:,21]).norm(dim=-1)
            results.append({'path':str(p),'sha256':sha(p),'candidate_id':candidate_ids[k],
                            'initial_noise_sha256':metadata['initial_noise_sha256'],
                            'feature_content_sha256':metadata['feature_content_sha256'],
                            'task_object_position':object_positions[k].tolist(),
                            'wrist_error_at_grasp_m':float(error[40]),
                            'max_wrist_error_m':float(error.max()),'mean_wrist_error_m':float(error.mean())})
        report={'status':'candidates_generated','candidates':results,'checkpoint_sha256':expected,
                'scene_sha256':scene_sha,'scene':scene,'elapsed_s':time.monotonic()-began,
                'generated_reference_admission':False,'PPO_started':False,
                'scope':'task-conditioned 29DoF adaptation; 263D output still needs robot-space fitting and task filters'}
        with (output/'report.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report),flush=True)
        for item in results:
            with records.open('a') as f:f.write(json.dumps(item)+'\n')
        all_results.extend(results)
        root_status(status='sampling',requested_candidates=args.count,generated_candidates=len(all_results),
                    accepted_references=0,latest_batch=start,PPO_started=False)
    assert len(all_results)==args.count
    assert len({r['candidate_id'] for r in all_results})==args.count
    assert len({r['initial_noise_sha256'] for r in all_results})==args.count
    assert len({r['feature_content_sha256'] for r in all_results})==args.count
    final={'status':'sampling_completed_not_postoptimized','requested_candidates':args.count,
           'generated_candidates':len(all_results),'accepted_references':0,
           'records_sha256':sha(records),'checkpoint_sha256':expected,'scene_sha256':scene_sha,
           'PPO_started':False,'note':'candidate counts are not admitted reference counts'}
    with (root_output/'report.json').open('x') as f:json.dump(final,f,indent=2)
    root_status(**final)



if __name__=='__main__':main()
