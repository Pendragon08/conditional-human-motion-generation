"""Reference-free actor inputs and private, interpolated 29DoF reward references."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

BODY_JOINTS = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint',
    'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint',
    'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]
SCENE = {'table_center': [.45, 0., .4], 'table_size': [.6, .6, .8],
         'object_center': [.35, -.20, .90], 'object_size': [.05, .05, .20],
         'object_mass': .1, 'success_height': .95, 'ground_z': 0.}
SCENE_SHA = hashlib.sha256(json.dumps(SCENE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_mul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((w, v), -1)


def quat_apply(q, v):
    uv = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


def interpolate_quat(a, b, t):
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    dot = (a * b).sum(-1, keepdim=True).clamp(-1, 1)
    angle = torch.acos(dot.clamp(-.999999, .999999))
    spherical = (torch.sin((1-t)*angle)*a + torch.sin(t*angle)*b) / torch.sin(angle).clamp_min(1e-7)
    linear = (1-t)*a + t*b
    result = torch.where(dot > .9995, linear, spherical)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def actor_observation(angular_velocity, gravity, joint_pos_rel, joint_vel_rel,
                      previous_action, root_pos, root_quat, object_pos, object_quat):
    """Only live sensors/action history. No reference, phase, goal motion or motion ID argument."""
    if joint_pos_rel.shape[-1] != 29 or joint_vel_rel.shape[-1] != 29 or previous_action.shape[-1] != 31:
        raise ValueError('expected 29 body joints and 31 actions')
    inverse = quat_conjugate(root_quat)
    obj_pos = quat_apply(inverse, object_pos-root_pos)
    obj_quat = quat_mul(inverse, object_quat)
    obj_quat = torch.where(obj_quat[..., :1] < 0, -obj_quat, obj_quat)
    obs = torch.cat((angular_velocity, gravity, joint_pos_rel, joint_vel_rel,
                     previous_action, obj_pos, obj_quat), dim=-1)
    if obs.shape[-1] != 102 or not torch.isfinite(obs).all():
        raise ValueError('invalid live actor observation')
    return obs


class FixedFrameMap:
    """Restore URDF fixed frames that PhysX merges, with their actual fixed offsets."""
    def __init__(self, urdf, physical_body_names, requested_frames, device):
        joints={j.find('child').get('link'):j for j in ET.parse(urdf).getroot().findall('joint')}
        indices=[];offsets=[];rotations=[];ancestors=[]
        for frame in requested_frames:
            node=frame;pos=np.zeros(3);rot=np.eye(3)
            while node not in physical_body_names:
                j=joints[node]
                if j.get('type')!='fixed':raise ValueError('missing actuated frame '+node)
                origin=j.find('origin')
                xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
                r,p,y=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
                cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
                rr=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                             [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
                pos=xyz+rr@pos;rot=rr@rot;node=j.find('parent').get('link')
            indices.append(physical_body_names.index(node));offsets.append(pos);rotations.append(rot);ancestors.append(node)
        self.indices=indices
        self.offsets=torch.tensor(np.array(offsets),device=device,dtype=torch.float32)
        self.rotations=torch.tensor(np.array(rotations),device=device,dtype=torch.float32)
        self.ancestors=ancestors

    def positions(self, body_positions, body_quaternions):
        q=body_quaternions[:,self.indices]
        return body_positions[:,self.indices]+quat_apply(q,self.offsets.expand(q.shape[0],-1,-1))


class ReferenceBank29:
    """CPU-file validation followed by a compact GPU bank; never exposed to actor inputs."""
    def __init__(self, manifest_path, asset_sha, device):
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        if manifest['asset_sha256'] != asset_sha or manifest['scene_sha256'] != SCENE_SHA:
            raise ValueError('reference asset/scene binding mismatch')
        calibration_path=Path(manifest['calibration_report'])
        if sha(calibration_path)!=manifest['calibration_report_sha256']:
            raise ValueError('native calibration hash mismatch')
        calibration=json.loads(calibration_path.read_text())
        if not (calibration['status']=='smoke_completed' and calibration['geometry_matches_pinned_urdf']
                and calibration['asset_sha256']==asset_sha and calibration['scene_sha256']==SCENE_SHA
                and calibration['action_dimension']==31 and calibration['actor_observation_dimension']==102):
            raise ValueError('native calibration is not applicable to this scene/robot')
        items = manifest['references']
        if len(items) < 2:
            raise ValueError('at least two generated references required')
        self.lengths, self.tg = [], []
        records = []
        for item in items:
            source = Path(item['path'])
            if sha(source) != item['sha256']:
                raise ValueError('reference hash mismatch')
            report_path = Path(item['postopt_report'])
            if sha(report_path) != item['postopt_report_sha256']:
                raise ValueError('post-optimization report hash mismatch')
            report = json.loads(report_path.read_text())
            original_path=Path(item['kinematic_report'])
            original=json.loads(original_path.read_text())
            if not (report['eligible_for_calibrated_scene_reference'] and all(report['native_checks'].values())
                    and report['asset_sha256']==asset_sha and report['scene_sha256']==SCENE_SHA
                    and sha(original_path)==report['kinematic_report_sha256']
                    and all(original['checks'].values()) and original['output_sha256']==item['sha256']):
                raise ValueError('generated reference failed scene checks')
            with np.load(source, allow_pickle=False) as z:
                record = {k: z[k].astype(np.float32) for k in
                          ('q', 'root_pos', 'root_quat', 'keypoints', 'hand_closed', 'right_wrist_quat')}
                names = z['joint_names'].tolist()
                fps = float(z['fps'])
            n = len(record['q'])
            shapes = {'q': (n,29), 'root_pos': (n,3), 'root_quat': (n,4),
                      'keypoints': (n,22,3), 'hand_closed': (n,2), 'right_wrist_quat': (n,4)}
            if names != BODY_JOINTS or fps != 20 or not 2 <= n <= 200:
                raise ValueError('reference joint order/time mismatch')
            for k, shape in shapes.items():
                if record[k].shape != shape or not np.isfinite(record[k]).all():
                    raise ValueError('invalid reference array '+k)
            for k in ('root_quat','right_wrist_quat'):
                if np.max(np.abs(np.linalg.norm(record[k],axis=-1)-1)) > 1e-4:
                    raise ValueError('invalid quaternion')
            tg = float(item['grasp_time_s'])
            if not 0 < tg < (n-1)/20 or not np.isin(record['hand_closed'], [0,1]).all():
                raise ValueError('invalid grasp schedule')
            self.lengths.append(n); self.tg.append(tg); records.append(record)
        maxn = max(self.lengths)
        self.data = {k: torch.tensor(np.stack([np.pad(r[k], ((0,maxn-len(r[k])),)+((0,0),)*(r[k].ndim-1),
                                                                    mode='edge') for r in records]), device=device)
                     for k in records[0]}
        self.lengths = torch.tensor(self.lengths, device=device, dtype=torch.long)
        self.tg = torch.tensor(self.tg, device=device, dtype=torch.float32)
        self.manifest_sha256 = sha(path)
        self.count = len(records)

    def query(self, ids, seconds):
        clock = torch.minimum(seconds.clamp_min(0)*20, (self.lengths[ids]-1).float())
        lo = clock.floor().long(); hi = torch.minimum(lo+1, self.lengths[ids]-1)
        weight = clock-lo
        result = {}
        for key, value in self.data.items():
            a,b = value[ids,lo],value[ids,hi]
            t = weight.reshape((-1,)+(1,)*(a.ndim-1))
            result[key] = (a if key == 'hand_closed' else interpolate_quat(a,b,t)
                           if key in ('root_quat','right_wrist_quat') else torch.lerp(a,b,t))
        result['after_grasp'] = seconds >= self.tg[ids]
        return result
