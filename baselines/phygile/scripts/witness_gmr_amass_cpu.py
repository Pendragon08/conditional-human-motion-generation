"""Seeded CPU environment witness. This is not a motion-quality experiment."""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import numpy as np
import torch
import mujoco
import smplx
import mink


def main(args):
    if args.output.exists():raise FileExistsError('Existing witness preserved')
    spec=json.loads(args.spec.read_text())
    actual={name:md.version(name) for name in spec['required_versions']}
    if actual!=spec['required_versions']:raise ValueError('Environment version drift: '+str(actual))
    sys.path.insert(0,str(args.gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting
    sys.path.insert(0,str(args.ingestion_root/'reimplementation'))
    from codec262 import G1Codec
    codec=G1Codec(args.ingestion_root/'configs/g1_codec_skeleton.json')
    xml=args.gmr_root/'assets/unitree_g1/g1_mocap_29dof.xml'
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    rng=np.random.default_rng(42)
    joints=rng.uniform(codec.low,codec.high,size=(16,29))
    pos=np.zeros((16,3));pos[:,2]=1
    quat=np.zeros((16,4));quat[:,0]=1
    positions,rotations=codec.fk(pos,quat,joints)
    missing=[n for n in codec.body_names if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)<0]
    pos_error=rot_error=0.0
    if not missing:
        for t in range(16):
            data.qpos[:7]=np.r_[pos[t],quat[t]]
            for i,name in enumerate(codec.joint_names):
                joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                if joint<0:raise ValueError('Missing G1 joint '+name)
                data.qpos[model.jnt_qposadr[joint]]=joints[t,i]
            mujoco.mj_forward(model,data)
            for name in codec.body_names:
                body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
                pos_error=max(pos_error,float(np.max(np.abs(data.xpos[body]-positions[name][t]))))
                rot_error=max(rot_error,float(np.max(np.abs(data.xmat[body].reshape(3,3)-rotations[name][t]))))
    torch.manual_seed(42)
    x=torch.randn(8,8,device='cpu');sentinel=float((x@x.T).sum())
    good=not missing and pos_error<1e-5 and rot_error<1e-5
    result={'status':'CPU_WITNESS_PASSED' if good else 'G1_FK_INCOMPATIBLE',
            'python':sys.version,'versions':actual,'seed':42,'device':'cpu',
            'gmr_class_imported':True,'fk_pose_count':16,
            'missing_codec_bodies_in_gmr_model':missing,
            'fk_max_position_abs_m':pos_error if not missing else None,
            'fk_max_rotation_matrix_abs':rot_error if not missing else None,
            'torch_cpu_kernel_sum':sentinel,
            'gmr_xml_sha256':hashlib.sha256(xml.read_bytes()).hexdigest(),
            'spec_hash':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8],
            'human_model_forward_pass_executed':False,'gmr_retargeting_executed':False,
            'environment_fully_ready_for_humanml_processing':False,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    if not good:raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gmr-root','ingestion-root','spec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
