"""Independent, versioned G1 262D codec. Conventions are not author-confirmed.

The paper leaves rotation frames, differencing and the initial orientation
implicit. This codec makes them explicit; initial root pose is required side
information. Generated rotations are projected onto URDF revolute joints and
redundant position/velocity blocks are diagnostics, never used as ground truth.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from features import pack_blocks

SCHEMA = 'phygile-g1-262-semantic-urdf-discrete-v1'
SELECTED = tuple(f'{side}_{part}' for part in
                 ('elbow_link', 'wrist_roll_link', 'knee_link', 'ankle_pitch_link',
                  'ankle_roll_link', 'rubber_hand') for side in ('left', 'right'))
FEET = tuple(f'{side}_{part}' for part in ('ankle_pitch_link', 'ankle_roll_link')
             for side in ('left', 'right'))


def finite(x, shape=None):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or (shape is not None and x.shape != shape):
        raise ValueError(f'Nonfinite or wrong shape: {x.shape}, expected {shape}')
    return x


def quat_matrix(q):
    q = finite(q)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError('Zero quaternion')
    w, x, y, z = np.moveaxis(q/norm, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3,3))


def matrix_quat(m):
    m = finite(m)
    out = []
    for a in m.reshape(-1,3,3):
        trace = np.trace(a)
        if trace > 0:
            s = np.sqrt(trace+1)*2
            q = [s/4, (a[2,1]-a[1,2])/s, (a[0,2]-a[2,0])/s, (a[1,0]-a[0,1])/s]
        else:
            i = int(np.argmax(np.diag(a))); j = (i+1)%3; k = (i+2)%3
            s = np.sqrt(max(1+a[i,i]-a[j,j]-a[k,k], 0))*2
            if s < 1e-12:
                raise ValueError('Invalid rotation matrix')
            v = np.zeros(3); v[i] = s/4; v[j] = (a[j,i]+a[i,j])/s; v[k] = (a[k,i]+a[i,k])/s
            q = [(a[k,j]-a[j,k])/s, *v]
        q = np.asarray(q); q /= np.linalg.norm(q)
        out.append(q if q[0] >= 0 else -q)
    return np.asarray(out).reshape(m.shape[:-2]+(4,))


def rotvec_matrix(v):
    v = finite(v); theta = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = v/np.maximum(theta, 1e-16)
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    k = np.stack([zero,-z,y,z,zero,-x,-y,x,zero], -1).reshape(v.shape[:-1]+(3,3))
    eye = np.broadcast_to(np.eye(3), k.shape)
    return eye + np.sin(theta)[...,None]*k + (1-np.cos(theta))[...,None]*(k@k)


def matrix_rotvec(m):
    q = matrix_quat(m); v = q[...,1:]; n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v*(2*np.arctan2(n,q[...,:1])/np.maximum(n,1e-16))


def rpy_matrix(rpy):
    r,p,y = rpy
    return rotvec_matrix([0,0,y]) @ rotvec_matrix([0,p,0]) @ rotvec_matrix([r,0,0])


def to6d(m):
    return np.concatenate([m[..., :,0],m[..., :,1]], -1)


def from6d(x):
    x = finite(x); a,b = x[...,:3], x[...,3:]
    na = np.linalg.norm(a,axis=-1,keepdims=True)
    if (na < 1e-8).any():
        raise ValueError('Degenerate first 6D rotation vector')
    a = a/na; b = b-(a*b).sum(-1,keepdims=True)*a
    nb = np.linalg.norm(b,axis=-1,keepdims=True)
    if (nb < 1e-8).any():
        raise ValueError('Collinear 6D rotation vectors')
    b = b/nb
    return np.stack([a,b,np.cross(a,b)],-1)


def derivative(x, fps):
    if len(x) < 2:
        return np.zeros_like(x)
    d = np.diff(x,axis=0)*fps
    return np.concatenate([d,d[-1:]],0)


class G1Codec:
    def __init__(self, spec_path):
        self.path = Path(spec_path)
        self.spec = json.loads(self.path.read_text(encoding='utf-8'))
        self.joint_names = tuple(self.spec['joint_names'])
        self.body_names = tuple(self.spec['body_names'])
        self.joints = self.spec['joints']
        self.movable = {j['name']: j for j in self.joints if j['type']=='revolute'}
        if len(self.joint_names)!=29 or len(self.body_names)!=30 or set(self.movable)!=set(self.joint_names):
            raise ValueError('Expected pinned 29-DoF G1 skeleton')
        self.low = np.array([float(self.movable[n]['limit']['lower']) for n in self.joint_names])
        self.high = np.array([float(self.movable[n]['limit']['upper']) for n in self.joint_names])

    def fk(self, root_pos, root_quat, joint_pos):
        root_pos = finite(root_pos); root_quat = finite(root_quat)
        joint_pos = finite(joint_pos, (len(root_pos),29))
        positions = {'pelvis': root_pos}; rotations = {'pelvis': quat_matrix(root_quat)}
        remaining = list(self.joints)
        while remaining:
            pending=[]
            for j in remaining:
                if j['parent'] not in positions:
                    pending.append(j); continue
                parent=rotations[j['parent']]
                origin=rpy_matrix(j['rpy'])
                motion=np.eye(3) if j['type']=='fixed' else rotvec_matrix(
                    joint_pos[:,self.joint_names.index(j['name']),None]*np.asarray(j['axis']))
                positions[j['child']]=positions[j['parent']]+np.einsum('tij,j->ti',parent,j['xyz'])
                rotations[j['child']]=parent@origin@motion
            if len(pending)==len(remaining):
                raise ValueError('Disconnected or cyclic URDF')
            remaining=pending
        return positions, rotations

    def encode(self, native, fps, canonicalize=True):
        native = finite(native)
        if native.ndim!=2 or native.shape[1]!=71 or len(native)<2 or not np.isfinite(fps) or fps<=0:
            raise ValueError('Expected at least two native root13/q29/dq29 frames and positive fps')
        pos=native[:,:3].copy(); rot=quat_matrix(native[:,3:7]); q=native[:,13:42]
        transform=np.eye(3); offset=np.zeros(3)
        if canonicalize:
            yaw=np.arctan2(rot[0,1,0],rot[0,0,0])
            transform=rotvec_matrix([0,0,-yaw]); offset=pos[0].copy(); offset[2]=0
            pos=np.einsum('ij,tj->ti',transform,pos-offset);rot=transform@rot
        positions, rotations=self.fk(pos,matrix_quat(rot),q)
        inverse=rot.transpose(0,2,1)
        local=np.stack([np.einsum('tij,tj->ti',inverse,positions[n]-pos) for n in SELECTED],1)
        velocities=np.stack([np.einsum('tij,tj->ti',inverse,derivative(positions[n],fps))
                             for n in ('pelvis',)+SELECTED],1)
        angular=matrix_rotvec(rot[1:]@rot[:-1].transpose(0,2,1))*fps
        angular=np.concatenate([angular,angular[-1:]],0)
        relative=np.stack([rotations[self.movable[n]['parent']].transpose(0,2,1)
                           @rotations[self.movable[n]['child']] for n in self.joint_names],1)
        foot=np.stack([(positions[n][:,2]<.05)&(np.linalg.norm(derivative(positions[n],fps)[:,:2],axis=-1)<.01)
                       for n in FEET],1)
        hand=np.stack([positions[f'{s}_rubber_hand'][:,2]<.10 for s in ('left','right')],1)
        features=pack_blocks({'root_angular_velocity':angular,'root_linear_velocity':derivative(pos,fps),
            'root_height':pos[:,2:3],'local_body_positions':local,'body_rotations_6d':to6d(relative),
            'local_body_velocities':velocities,'foot_contacts':foot,'hand_contacts':hand})
        metadata={'schema':SCHEMA,'fps':float(fps),'initial_root_pos':pos[0].tolist(),
            'initial_root_quat':matrix_quat(rot[0]).tolist(),'initial_joint_pos':q[0].tolist(),
            'canonical_rotation':transform.tolist(),'canonical_offset':offset.tolist(),
            'skeleton_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
            'rotation_order':list(self.joint_names),'position_order':list(SELECTED),
            'conventions':'forward differences; last repeated; canonical-world root velocities; '
              'root-local RIC/selected velocities; parent-local child rotations including URDF origin; column 6D',
            'author_exact_mapping_verified':False}
        return features,metadata

    def decode(self, features, metadata, project=False, restore_world=True):
        x=finite(features)
        if x.ndim!=2 or x.shape[1]!=262 or len(x)<2 or metadata.get('schema')!=SCHEMA:
            raise ValueError('Explicit compatible schema and at least two [T,262] frames required')
        if metadata['skeleton_sha256']!=hashlib.sha256(self.path.read_bytes()).hexdigest():
            raise ValueError('Skeleton mismatch')
        fps=float(metadata['fps'])
        if not np.isfinite(fps) or fps<=0:
            raise ValueError('Invalid fps')
        rot=np.empty((len(x),3,3));rot[0]=quat_matrix(metadata['initial_root_quat'])
        pos=np.empty((len(x),3));pos[0]=metadata['initial_root_pos']
        for t in range(1,len(x)):
            rot[t]=rotvec_matrix(x[t-1,:3]/fps)@rot[t-1]
            pos[t]=pos[t-1]+x[t-1,3:6]/fps
        pos[:,2]=x[:,6]
        relative=from6d(x[:,43:217].reshape(-1,29,6))
        values=[]; residual=[]
        for i,n in enumerate(self.joint_names):
            j=self.movable[n]; r=rpy_matrix(j['rpy']).T@relative[:,i]
            axis=np.asarray(j['axis']); skew=np.stack([r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]],-1)/2
            angle=np.arctan2(skew@axis,(np.trace(r,axis1=1,axis2=2)-1)/2)
            residual.append(np.linalg.norm(rotvec_matrix(angle[:,None]*axis)-r,axis=(1,2)))
            values.append(angle)
        q=np.stack(values,-1); residual=np.stack(residual,-1)
        violations=(q<self.low-1e-5)|(q>self.high+1e-5)
        # A codec must preserve an existing out-of-limit reference for auditing.
        # Only the explicitly requested generator projection clips joint limits.
        if not project and residual.max()>1e-4:
            raise ValueError('Features are not URDF-consistent revolute joint rotations')
        if project:q=np.clip(q,self.low,self.high)
        canonical_pos,canonical_rot=self.fk(pos,matrix_quat(rot),q)
        ric=np.stack([np.einsum('tij,tj->ti',rot.transpose(0,2,1),canonical_pos[n]-pos) for n in SELECTED],1)
        report={'rotation_projection_residual_max':float(residual.max()),
                'joint_limit_violations':int(violations.sum()),
                'ric_redundancy_residual_m':float(np.linalg.norm(ric-x[:,7:43].reshape(-1,12,3),axis=-1).mean()),
                'projected':bool(project),'root_pose_side_information_required':True}
        if restore_world:
            inv=np.asarray(metadata['canonical_rotation']).T
            pos=np.einsum('ij,tj->ti',inv,pos)+np.asarray(metadata['canonical_offset']);rot=inv@rot
        quat=matrix_quat(rot);omega=matrix_rotvec(rot[1:]@rot[:-1].transpose(0,2,1))*fps
        omega=np.concatenate([omega,omega[-1:]],0)
        native=np.concatenate([pos,quat,derivative(pos,fps),omega,q,derivative(q,fps)],-1)
        positions,rotations=self.fk(pos,quat,q)
        body_pos=np.stack([positions[n] for n in self.body_names],1)
        body_rot=np.stack([rotations[n] for n in self.body_names],1)
        motion={'fps':np.array([fps],np.float32),'joint_names':np.asarray(self.joint_names),
            'body_names':np.asarray(self.body_names),'joint_pos':q.astype(np.float32),
            'joint_vel':derivative(q,fps).astype(np.float32),'body_pos_w':body_pos.astype(np.float32),
            'body_quat_w':matrix_quat(body_rot).astype(np.float32),
            'body_lin_vel_w':derivative(body_pos,fps).astype(np.float32),
            'body_ang_vel_w':np.concatenate([matrix_rotvec(body_rot[1:]@body_rot[:-1].transpose(0,1,3,2))*fps,
                matrix_rotvec(body_rot[-1:]@body_rot[-2:-1].transpose(0,1,3,2))*fps],0).astype(np.float32)}
        return native,motion,report

    def native_from_motion(self, motion):
        if tuple(motion['body_names'].tolist())!=self.body_names or tuple(motion['joint_names'].tolist())!=self.joint_names:
            raise ValueError('Name/order mismatch; no implicit index remapping')
        root=self.body_names.index('pelvis')
        return np.concatenate([motion['body_pos_w'][:,root],motion['body_quat_w'][:,root],
            motion['body_lin_vel_w'][:,root],motion['body_ang_vel_w'][:,root],
            motion['joint_pos'],motion['joint_vel']],-1)
