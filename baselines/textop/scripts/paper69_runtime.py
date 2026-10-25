"""Isolated29DoF/69D adapter for the pinned RobotMDAR model/training code.

Own differentiable codec and FK; official Transformer models/managers are reused.
No dataset conversion, training launch, optimizer step, package install or disk
mutation occurs on import. Install before importing managers or dataloaders.
"""
import pathlib,sys
import mujoco
import numpy as np
import torch

DOF=29
FEATURES=69

def quat_mul(a,b):
    # Scalar-first unit quaternions; multiplication has no axis-angle singularity.
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    return torch.stack([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],-1)

def quat_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))

def rpy_quat(roll,pitch,yaw):
    cr,sr=torch.cos(roll/2),torch.sin(roll/2);cp,sp=torch.cos(pitch/2),torch.sin(pitch/2);cy,sy=torch.cos(yaw/2),torch.sin(yaw/2)
    return torch.stack([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy],-1)

def quat_rpy_xyzw(q):
    r=quat_matrix(q[..., [3,0,1,2]])
    roll=torch.atan2(r[...,2,1],r[...,2,2])
    pitch=torch.atan2(-r[...,2,0],(r[...,2,1].square()+r[...,2,2].square()).clamp_min(1e-12).sqrt())
    yaw=torch.atan2(r[...,1,0],r[...,0,0])
    return roll,pitch,yaw

def encode69(motion,skeleton=None):
    unbatched=motion['dof'].ndim==2
    data={k:v.unsqueeze(0) if unbatched else v for k,v in motion.items()}
    p,q,j,c=[data[k] for k in ['root_trans_offset','root_rot','dof','contact_mask']]
    assert j.shape[-1]==29 and p.shape[1]>=2
    roll,pitch,yaw=quat_rpy_xyzw(q)
    dy=yaw[:,1:]-yaw[:,:-1];dy=torch.atan2(torch.sin(dy),torch.cos(dy))
    dp=p[:,1:]-p[:,:-1];co,si=torch.cos(yaw[:,:-1]),torch.sin(yaw[:,:-1])
    local=torch.stack([co*dp[...,0]+si*dp[...,1],-si*dp[...,0]+co*dp[...,1],dp[...,2]],-1)
    phi=torch.stack([torch.sin(roll),torch.cos(roll)-1,torch.sin(pitch),torch.cos(pitch)-1],-1)[:,:-1]
    feature=torch.cat([phi,dy[...,None],c[:,:-1],local,p[:,:-1,2:3],j[:,:-1],j[:,1:]-j[:,:-1]],-1)
    assert feature.shape[-1]==69
    pose={'root_trans_offset':p[:,0], 'root_rot':q[:,0]}
    return (feature[0],{k:v[0] for k,v in pose.items()}) if unbatched else (feature,pose)

def decode69(feature,abs_pose=None):
    assert feature.shape[-1]==69
    unbatched=feature.ndim==2
    f=feature.unsqueeze(0) if unbatched else feature
    b,t,_=f.shape;assert t>0
    if abs_pose is None:
        initial=f.new_zeros(b,3);yaw0=f.new_zeros(b)
    else:
        initial=abs_pose['root_trans_offset'];q0=abs_pose['root_rot']
        if initial.ndim==1:initial=initial[None];q0=q0[None]
        yaw0=quat_rpy_xyzw(q0)[2]
    roll=torch.atan2(f[...,0],f[...,1]+1);pitch=torch.atan2(f[...,2],f[...,3]+1)
    yaw=yaw0[:,None]+torch.cat([f.new_zeros(b,1),f[:,:-1,4].cumsum(1)],1)
    co,si=torch.cos(yaw),torch.sin(yaw);d=f[...,7:10]
    world=torch.stack([co*d[...,0]-si*d[...,1],si*d[...,0]+co*d[...,1],d[...,2]],-1)
    integrated=initial[:,None]+torch.cat([f.new_zeros(b,1,3),world[:,:-1].cumsum(1)],1)
    p=torch.cat([integrated[...,:2],f[...,10:11]],-1)
    q=rpy_quat(roll,pitch,yaw)[...,[1,2,3,0]]
    out={'root_trans_offset':p,'root_rot':q,'dof':f[...,11:40],'contact_mask':f[...,5:7]}
    return {k:v[0] for k,v in out.items()} if unbatched else out

def to_qpos69(motion):
    assert motion['dof'].shape[-1]==29
    q=motion['root_rot'][...,[3,0,1,2]]
    return torch.cat([motion['root_trans_offset'],q,motion['dof']],-1),motion['contact_mask']

class DifferentiableG1:
    """MuJoCo model topology with differentiable quaternion hinge transforms.

    Keeps fixed sensor/link frames in the topology, rather than assuming every
    non-root body is an actuated joint. All body frames are returned in MuJoCo
    order without world. These geometry bodies are an explicit implementation
    choice: the paper does not release its final69D geometry-body configuration.
    """
    def __init__(self,xml,device='cpu'):
        self.xml=pathlib.Path(xml);self.device=torch.device(device)
        self.model=mujoco.MjModel.from_xml_path(str(self.xml));m=self.model
        assert m.nq==36 and m.nv==35
        self.root=int(m.jnt_bodyid[0]);assert self.root==1 and int(m.jnt_type[0])==0
        assert all(int(x)==3 for x in m.jnt_type[1:])
        self.joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(1,m.njnt)]
        assert len(self.joint_names)==29 and len(set(m.jnt_bodyid[1:]))==29
        self.body_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,m.nbody)]
        self.foot_id=[self.body_names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']]
        self.num_bodies=len(self.body_names);self.num_extend_dof=29
        self.parent_indices=torch.as_tensor(m.body_parentid[1:]-1,device=self.device)
        self.local_translations=torch.as_tensor(m.body_pos[1:].copy(),device=self.device,dtype=torch.float32)
        self.local_rotations=torch.as_tensor(m.body_quat[1:].copy(),device=self.device,dtype=torch.float32)
        self.hinges={int(m.jnt_bodyid[j]):j for j in range(1,m.njnt)}
        self.fk=self

    def forward_kinematics(self,motion,return_full=False):
        unbatched=motion['dof'].ndim==2
        data={k:v.unsqueeze(0) if unbatched else v for k,v in motion.items()}
        p=data['root_trans_offset'];q=data['root_rot'][...,[3,0,1,2]];dof=data['dof']
        assert dof.shape[-1]==29 and p.ndim==3
        m=self.model;positions=[None]*m.nbody;quats=[None]*m.nbody
        positions[1]=p;quats[1]=q
        for body in range(2,m.nbody):
            parent=int(m.body_parentid[body]);pq=quats[parent];pp=positions[parent]
            offset=p.new_tensor(m.body_pos[body]);rest=p.new_tensor(m.body_quat[body])
            base=quat_mul(pq,rest);local=offset
            if body in self.hinges:
                joint=self.hinges[body];index=int(m.jnt_qposadr[joint])-7
                angle=dof[...,index]-float(m.qpos0[m.jnt_qposadr[joint]])
                axis=p.new_tensor(m.jnt_axis[joint]);anchor=p.new_tensor(m.jnt_pos[joint])
                hinge=torch.cat([torch.cos(angle/2)[...,None],torch.sin(angle/2)[...,None]*axis],-1)
                rotation=quat_matrix(hinge)
                local=offset+torch.matmul(quat_matrix(rest), (anchor-torch.matmul(rotation,anchor[...,None]).squeeze(-1))[...,None]).squeeze(-1)
                base=quat_mul(base,hinge)
            positions[body]=pp+torch.matmul(quat_matrix(pq),local[...,None]).squeeze(-1)
            quats[body]=base
        pos=torch.stack(positions[1:],-2);rot=torch.stack(quats[1:],-2)
        if dof.shape[1]>1:
            vel=(dof[:,1:]-dof[:,:-1])*50
            vel=torch.cat([vel,vel[:,-1:]],1)
        else:vel=torch.zeros_like(dof)
        out={**data,'global_translation':pos,'global_translation_extend':pos,'global_rotation':rot[...,[1,2,3,0]],'global_rotation_extend':rot[...,[1,2,3,0]],'global_rotation_mat':quat_matrix(rot),'dof_pos':dof,'dof_vel':vel,'fps':50}
        return out

    def get_foot_id(self):return self.foot_id

def install_runtime(xml):
    for name in ['robotmdar.train.manager','robotmdar.dataloader.data']:
        if name in sys.modules:raise RuntimeError('Install the69D adapter before '+name)
    import robotmdar.dtype.motion as motion
    assert motion.FeatureVersion==3
    motion.DOF_DIM=29;motion.n_qpos=36;motion.motion_feature_dim=69;motion.motion_feature_dim_v3=69
    motion.motion_dict_to_feature=motion.motion_dict_to_feature_v3=encode69
    motion.motion_feature_to_dict=motion.motion_feature_to_dict_v3=decode69
    motion.motion_dict_to_qpos=to_qpos69
    reference=DifferentiableG1(xml)
    defaults={'left_hip_pitch_joint':-.1,'right_hip_pitch_joint':-.1,'left_knee_joint':.3,'right_knee_joint':.3,'left_ankle_pitch_joint':-.2,'right_ankle_pitch_joint':-.2,'left_shoulder_pitch_joint':.2,'right_shoulder_pitch_joint':.2,'left_shoulder_roll_joint':.2,'right_shoulder_roll_joint':-.2,'left_elbow_joint':.9,'right_elbow_joint':.9}
    def neutral():
        f=torch.zeros(1,69);f[:,5:7]=1;f[:,10]=.75
        f[:,11:40]=torch.tensor([defaults.get(name,0.) for name in reference.joint_names]);return f
    motion.get_zero_feature=motion.get_zero_feature_v3=neutral
    import robotmdar.skeleton.robot as skeleton
    class Skeleton69(DifferentiableG1):
        def __init__(self,device='cpu',cfg=None):
            if cfg is not None:
                specified=pathlib.Path(cfg.asset.assetRoot)/cfg.asset.assetFileName
                assert specified.resolve()==pathlib.Path(xml).resolve(),'Refusing mismatched23DoF skeleton configuration'
            super().__init__(xml,device)
    skeleton.RobotSkeleton=Skeleton69
    from robotmdar.train.manager import DARManager
    def ldm_rollout(self):
        if not self.use_rollout or self.stage_idx<1:return False
        probability=min(.8,max(0.,(self.step-self.stages[0])/max(float(self.stages[1]),1e-6)))
        return torch.rand(1).item()<probability
    DARManager.should_rollout=ldm_rollout
    # The released sampler creates int32 timestep tensors. The pinned1.13
    # PyTorch requires int64 for coefficient and positional-table indexing.
    import robotmdar.diffusion.gaussian_diffusion as diffusion
    extract=diffusion._extract_into_tensor
    def extract_int64(array,timesteps,shape):return extract(array,timesteps.long(),shape)
    diffusion._extract_into_tensor=extract_int64
    from robotmdar.model.mld_denoiser import DenoiserTransformer
    forward=DenoiserTransformer.forward
    def forward_int64(self,x_t,timesteps,y=None):return forward(self,x_t,timesteps.long(),y)
    DenoiserTransformer.forward=forward_int64
    return {'dof':29,'feature_dim':69,'qpos_dim':36,'body_names':reference.body_names,'joint_names':reference.joint_names,'fk_geometry_bodies':'all MuJoCo bodies except world; fixed frames retained','ldm_max_rollout_probability':.8,'sampler_timestep_index_dtype':'int64 compatibility adapter','upstream_files_modified':False}
