"""CPU exploratory chain: old human prior -> old PyRoki, synthetic settings.

Not trained on robot data, not v2 author code, not a physics or training test.
Only writes to a fresh directory; no original source or checkpoint is modified.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

# Reuse the existing CPU tokenizer dependencies after the main environment;
# the caller supplies the pinned OpenAI CLIP source on PYTHONPATH.
sys.path.append('/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages')
import clip
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import torch
import yourdfpy
from diffusion import gaussian_diffusion as gd
import model.cmdm as cmdm
from utils.model_util import get_model_args
from explicit_hint_control import attach_explicit_hint_mask
from robot_diffusion_stats import configure_robot_stats
from robot_prior_sampling import sample_robot_prior
from postopt_generated_bundle import postopt_generated_bundle

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
HELPER=ROOT/'pyroki_grab_temporal_20261030'
CHECKPOINT=PROJECT/'TrajGen/save/omnicontrol_ckpt/model_humanml3d.pt'
CLIP=Path('/home/ouyanghao/.cache/clip/ViT-B-32.pt')
URDF=PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HELPER))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,create_conn_tree,solve_retargeting,foot_detect


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


class StubRotation2xyz:
    def __init__(self,**kwargs):self.smpl_model=torch.nn.Identity()


def main():
    torch.set_num_threads(1)
    if any(d.platform!='cpu' for d in jax.devices()):raise RuntimeError('CPU required')
    helper_receipt=json.loads((HELPER/'report.json').read_text())
    pins={CHECKPOINT:'886b722e95496f4ceafd704c4e1d0578888733d12ce67429677b80a91a9c36d0',
          CLIP:'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
          URDF:helper_receipt['urdf_sha256'],HELPER/'retarget_helpers/_utils.py':helper_receipt['helper_sha256']}
    if any(not p.is_file() or sha(p)!=h for p,h in pins.items()):raise ValueError('fixed inputs missing or changed')
    clip_impl=importlib.import_module('clip.clip')
    if clip_impl._MODELS['ViT-B/32'].split('/')[-2]!=pins[CLIP]:raise ValueError('CLIP cache differs from package')
    args=SimpleNamespace(dataset='humanml',latent_dim=512,layers=8,cond_mode='both_text_spatial',
                         cond_mask_prob=.1,arch='trans_enc',emb_trans_dec=False)
    cmdm.Rotation2xyz=StubRotation2xyz
    with patch.object(clip_impl,'_download',return_value=str(CLIP)): model=cmdm.CMDM(**get_model_args(args,None))
    missing,unexpected=model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True),strict=False)
    if unexpected or any(not key.startswith('clip_model.') for key in missing):raise ValueError('checkpoint mismatch')
    attach_explicit_hint_mask(model);model.eval()
    def fake_stats(path):
        n=66 if 'raw' in str(path).lower() else 263
        return np.ones(n,dtype=np.float32) if 'Std' in str(path) else np.zeros(n,dtype=np.float32)
    with patch.object(gd.np,'load',side_effect=fake_stats):
        diffusion=gd.GaussianDiffusion(betas=np.linspace(.001,.02,4),model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_SMALL,loss_type=gd.LossType.MSE,dataset='humanml')
    configure_robot_stats(diffusion,np.zeros(263),np.ones(263),np.zeros(66),np.ones(66))
    targets=torch.zeros((1,8,22,3));mask=torch.zeros((1,8,22),dtype=torch.bool)
    targets[0,5,20]=torch.tensor([.2,0.,.1]);mask[0,5,20]=True
    sample=sample_robot_prior(model,diffusion,texts=['a person reaches forward'],targets=targets,target_mask=mask,
                              cfg_scale=2.5,spatial_step_size=.05,spatial_steps=1,seed=772)
    meta={'sampling':sample['metadata'],'human_pretrained_checkpoint_sha256':pins[CHECKPOINT],
          'synthetic_statistics':True,'training_or_physical_acceptance':False}
    raw=json.dumps(meta,allow_nan=False).encode('utf8');prior=HERE/'synthetic_prior.npz'
    with prior.open('xb') as f:np.savez_compressed(f,canonical_points=sample['canonical_points'].cpu().numpy(),
        normalized_features=sample['normalized_features'].cpu().numpy(),metadata_utf8=np.frombuffer(raw,dtype=np.uint8))
    del model, sample
    urdf=yourdfpy.URDF.load(str(URDF),load_meshes=False,load_collision_meshes=True)
    robot=pk.Robot.from_urdf(urdf);collision=pk.collision.RobotCollision.from_urdf(urdf)
    hi,ri=get_humanoid_retarget_indices_g1_27dof();tree=create_conn_tree(robot,ri)
    ground=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),height_data=jnp.zeros((100,100),dtype=jnp.float32))
    names=list(robot.joints.actuated_names)
    limits={j.attrib['name']:(float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper']))
            for j in ET.parse(URDF).getroot().findall('joint') if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    lower,upper=zip(*(limits[n] for n in names))
    weights=dict(helper_receipt['weights']);weights['root_smoothness']=2.;weights['foot_skating']=2.
    output=postopt_generated_bundle(input_path=prior,input_sha256=sha(prior),output_path=HERE/'synthetic_robot.npz',
        sample_index=0,solver_provenance={'helper_sha256':pins[HELPER/'retarget_helpers/_utils.py'],'urdf_sha256':pins[URDF]},
        min_free_bytes=8*1024**3,solver=solve_retargeting,foot_detector=foot_detect,robot=robot,robot_collision=collision,
        heightmap=ground,human_indices=hi,robot_indices=ri,connection_mask=tree,weights=weights,
        canonical_to_solver_rotation=[[0,0,1],[1,0,0],[0,1,0]],canonical_to_solver_translation=[0,0,0],
        pick_point_solver=[0,0,0],foot_displacement_sq_threshold=.002,joint_names=names,joint_lower=lower,joint_upper=upper,fps=20.)
    if any(sha(p)!=h for p,h in pins.items()):raise RuntimeError('fixed inputs changed during smoke')
    with np.load(HERE/'synthetic_robot.npz',allow_pickle=False) as f:
        result={'prior_sha256':sha(prior),'robot_output_sha256':sha(HERE/'synthetic_robot.npz'),
                'checkpoint_sha256':pins[CHECKPOINT],'clip_sha256':pins[CLIP],'helper_sha256':pins[HELPER/'retarget_helpers/_utils.py'],
                'script_sha256':sha(__file__),'root_shape':list(f['root_wxyz_xyz'].shape),'joint_shape':list(f['joint_pos'].shape),
                'sample_indices':f['input_frame_indices'].tolist(),'postopt_metadata':output['metadata']['postopt'],
                'training_ready':False,'limitations':['Old human pretrained model, synthetic stats/hints and four diffusion steps, not v2.',
                    'Dummy ground, old solver, no real robot checkpoint, semantics, source or physical acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
