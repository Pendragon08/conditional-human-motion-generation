"""Isolated old-Pick orientation-reward singularity experiment.

This is not connected to a task or claimed to reproduce DreamControl-v2.
The fallback is a design assumption: use the original initial target attitude
when the target direction or its horizontal projection is too small.
"""
import torch


def safe_target_orientation_error(env, asset_cfg, *, math_utils, slerp, epsilon=1e-6):
    if not isinstance(epsilon, (int,float)) or not 0 < epsilon < 1:
        raise ValueError('epsilon must be a small positive finite scalar')
    asset=env.scene[asset_cfg.name]
    position=asset.data.body_state_w[:,asset_cfg.body_ids[0],:3].clone()-env.scene.env_origins
    rotation=math_utils.quat_unique(asset.data.body_quat_w[:,asset_cfg.body_ids[0],:].clone())
    motion_times=env.episode_length_buf*env.step_dt+env.start_motion_times.clone().detach().to(device=env.device,dtype=torch.float32)-1.
    motion_ids=env.motion_ids.clone().detach().to(device=env.device,dtype=torch.long)
    motion=env.motion_lib.get_motion_state(motion_ids,motion_times)
    target=motion['grab_pos']+motion['offsets']; delta=target-position
    if not torch.isfinite(delta).all() or not torch.isfinite(rotation).all() or not torch.isfinite(motion_times).all():
        raise ValueError('non-finite reward inputs')
    valid=(torch.linalg.vector_norm(delta,dim=1)>epsilon)&(torch.linalg.vector_norm(delta[:,:2],dim=1)>epsilon)
    fallback=torch.zeros_like(delta);fallback[:,0]=1
    d=torch.where(valid[:,None],delta,fallback)
    a=d/torch.linalg.vector_norm(d,dim=1,keepdim=True)
    b=torch.stack((-a[:,1],a[:,0],torch.zeros_like(a[:,0])),dim=1)
    b=b/torch.linalg.vector_norm(b,dim=1,keepdim=True)
    z=torch.cross(a,b,dim=1);x=2*a-b;x=x/torch.linalg.vector_norm(x,dim=1,keepdim=True);y=torch.cross(z,x,dim=1)
    initial=torch.tensor([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]],device=position.device,dtype=position.dtype).expand(len(position),-1,-1)
    # Columns of original initial matrix are x=(0,0,-1), y=(0,1,0), z=(1,0,0).
    initial=initial.transpose(1,2)
    initial_quat=math_utils.quat_from_matrix(initial)
    target_quat=math_utils.quat_from_matrix(torch.stack([x,y,z],dim=2))
    target_quat=torch.where(valid[:,None],target_quat,initial_quat)
    blend=torch.clip((motion_times+1.)/1.5,0.,1.)
    desired=slerp(initial_quat,target_quat,blend.unsqueeze(1))
    angle=torch.abs(math_utils.quat_error_magnitude(desired,rotation))
    up=torch.tensor([0.,0.,1.],device=rotation.device,dtype=rotation.dtype).expand(len(position),-1)
    angle_post=torch.acos(torch.clamp(math_utils.quat_apply(rotation,up)[:,2],-1.,1.)).abs()
    result=torch.where(motion['is_closed'].bool(),angle_post,angle)
    if not torch.isfinite(result).all():raise ValueError('non-finite orientation reward')
    return result,valid
