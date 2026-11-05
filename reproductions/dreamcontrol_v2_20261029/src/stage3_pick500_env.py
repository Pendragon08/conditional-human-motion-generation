"""Same live Pick policy/task, with reference-bound object placement on reset.

Formal runs require500 distinct admitted reference contents. Bounded integration
tests explicitly request a smaller minimum and cannot admit a formal run.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass
import stage3_pick29_env as base
from stage3_pick29_core import ReferenceBank29, FixedFrameMap, BODY_JOINTS, sha


class ReferenceBank500(ReferenceBank29):
    def __init__(self,manifest_path,asset_sha,device,minimum_references=500):
        m=json.loads(Path(manifest_path).read_text());items=m['references']
        if minimum_references<2 or len(items)<minimum_references:
            raise ValueError('not enough admitted generated references')
        if m.get('object_pose_policy')!='per_reference_fixed' or m.get('task')!='pick':
            raise ValueError('reference-bound Pick object placement missing')
        super().__init__(manifest_path,asset_sha,device)
        fingerprints=set();noise=set();features=set();ids=set();positions=[];rotations=[]
        for item in items:
            report=json.loads(Path(item['postopt_report']).read_text())
            kin=json.loads(Path(item['kinematic_report']).read_text())
            p=np.array(item['task_object_position'],dtype=float);q=np.array(item['task_object_quaternion_wxyz'],dtype=float)
            if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or abs(np.linalg.norm(q)-1)>1e-6:
                raise ValueError('invalid reference object state')
            if not np.all(np.abs(p[:2]-[.35,-.2])<=.0600001) or p[2]!=.9:
                raise ValueError('object outside validated tabletop sampling domain')
            if not np.allclose(p,report['task_object_position'],atol=1e-12) or not np.allclose(p,kin['task_object_position'],atol=1e-12):
                raise ValueError('native/kinematic/manifest object placement differs')
            with np.load(item['path'],allow_pickle=False) as z:
                if not np.allclose(p,z['task_object_position'],atol=1e-12) or not np.allclose(q,z['task_object_quaternion_wxyz'],atol=1e-12):
                    raise ValueError('reference file object placement differs')
                meta=json.loads(z['metadata_utf8'].tobytes())
                content=b''.join(np.ascontiguousarray(z[k]).tobytes() for k in ['q','root_pos','root_quat','hand_closed'])
            fingerprints.add(hashlib.sha256(content).hexdigest())
            noise.add(meta['initial_noise_sha256']);features.add(meta['feature_content_sha256']);ids.add(meta['candidate_id'])
            if (meta['initial_noise_sha256']!=kin['source_initial_noise_sha256'] or
                    meta['feature_content_sha256']!=kin['source_feature_content_sha256']):
                raise ValueError('generated source provenance differs')
            positions.append(p);rotations.append(q)
        if any(len(x)!=len(items) for x in [fingerprints,noise,features,ids]):
            raise ValueError('duplicate generated source or robot-state content')
        self.object_positions=torch.tensor(np.array(positions),device=device,dtype=torch.float32)
        self.object_quaternions=torch.tensor(np.array(rotations),device=device,dtype=torch.float32)
        self.formal_reference_count_satisfied=len(items)>=500


def startup(env,env_ids=None):
    env._reference_bank=ReferenceBank500(env.cfg.reference_manifest,sha(base.ASSET),env.device,
                                        minimum_references=env.cfg.minimum_reference_count)
    env._reference_ids=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    env._reference_cache_step=-1
    env._body29_ids=env.scene['robot'].find_joints(BODY_JOINTS,preserve_order=True)[0]
    env._keypoint_frames=FixedFrameMap(base.URDF,env.scene['robot'].body_names,base.KEYPOINT_LINKS,env.device)
    assert len(env._body29_ids)==29


def reset_scene(env,env_ids):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    base.reset_scene(env,env_ids)
    bank=env._reference_bank;selected=env._reference_ids[env_ids]
    obj=env.scene['object'];state=obj.data.default_root_state[env_ids].clone()
    state[:,:3]=bank.object_positions[selected]+env.scene.env_origins[env_ids]
    state[:,3:7]=bank.object_quaternions[selected];state[:,7:]=0.
    obj.write_root_state_to_sim(state,env_ids=env_ids)


@configclass
class EventsCfg(base.EventsCfg):
    initialize=EventTermCfg(func=startup,mode='startup')
    reset=EventTermCfg(func=reset_scene,mode='reset')


@configclass
class Pick500EnvCfg(base.Pick29EnvCfg):
    events: EventsCfg=EventsCfg()
    minimum_reference_count: int=500
