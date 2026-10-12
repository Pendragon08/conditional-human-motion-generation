"""Frozen CLoSD adapter. Generated targets are never evaluation targets.

Experimental prefix/goal interventions are instance wrappers; upstream stays
clean. All first attempts, including prefix failures, are saved.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from physics_guided_core import foot_geometry, heading, metrics

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--manifest',required=True)
p.add_argument('--split',default='val',choices=['train','val','test'])
p.add_argument('--mode',default='planned',choices=['planned','native','replay'])
p.add_argument('--seed',type=int,default=11)
p.add_argument('--round',type=int,default=0)
p.add_argument('--noise-round',type=int)
p.add_argument('--limit-sources',type=int,default=0)
p.add_argument('--correction-file')
p.add_argument('--replay-file')
p.add_argument('--thresholds')
p.add_argument('--request-file')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
config=json.loads((root/'configs/physics_guided_20261011.json').read_text())
manifest=json.loads(Path(a.manifest).read_text())
sources=[r for r in manifest['rows'] if r['split']==a.split]
if a.limit_sources:
    sources=sources[:a.limit_sources]
cases=[]
for source in sources:
    for off in config['offsets_m']:
        row=source.copy(); row['offset_m']=off
        row['goal_joint_body']='Pelvis' if row['goal_joint']=='pelvis' else 'R_Wrist'
        row['case_id']=row['clip_id']+':'+str(off)
        cases.append(row)
out=Path(a.output).resolve()
assert not out.exists(), 'Use a fresh output; never replace an attempt'
out.mkdir(parents=True)
started=time.time()
spec=importlib.util.spec_from_file_location('pg_closd_entry',root/'scripts/r3_closd_isolated_entry.py')
entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from closd.env.tasks.closd_t2m import CLoSDT2M
from closd.env.tasks.closd_task import CLoSDTask
from closd.utils.rep_util import smpl_2_mujoco

thresholds=json.loads(Path(a.thresholds).read_text())['thresholds'] if a.thresholds else config['base_thresholds']
cache={path:np.load(path,allow_pickle=True).item()['data_dict'] for path in {r['cache_file'] for r in cases}}
mean=np.load(manifest['mean']); std=np.load(manifest['std'])
raw=np.stack([cache[r['cache_file']][r['cache_key']]['motion'][:80] for r in cases])
assert raw.shape==(len(cases),80,263)
assert np.isfinite(raw).all()
normed=(raw-mean)/std
correction=json.loads(Path(a.correction_file).read_text()) if a.correction_file else None
replay=np.load(a.replay_file,allow_pickle=False) if a.replay_file else None
store={}


class CollectionComplete(Exception):
    pass


def set_conditions(self,env_ids):
    # Reuse state reset, but never sample another caption or prefix on failure.
    CLoSDTask.update_mdm_conditions(self,env_ids)
    for i in env_ids.detach().cpu().tolist():
        r=cases[i]
        self.hml_prompts[i]=r['caption']; self.hml_tokens[i]=r['tokens']
        self.db_keys[i]=r['clip_id']; self.hml_lengths[i]=80
    self.hml_prefix_from_data[env_ids]=torch.as_tensor(normed[:,:40].transpose(0,2,1)[:,:,None,:],
                                      device=self.device,dtype=torch.float32)[env_ids]


CLoSDT2M.update_mdm_conditions=set_conditions
original_create=entry.official.create_rlgpu_env


def create(*args,**kwargs):
    env=original_create(*args,**kwargs); task=env.task
    n=task.num_envs; t=config['episode_steps_30hz']; b=task.num_bodies
    assert n==len(cases) and b==24 and abs(task.dt-1/30)<1e-6
    assert task.multi_target_cond, 'This experiment requires the released goal-conditioned DiP'
    assert task.context_len_20fps==20 and task.planning_horizon_30fps==60
    assert task.mdm.training is False
    # Freeze one dataset-derived world coordinate frame per case and run.
    init=task._rigid_body_pos[:,None].repeat(1,30,1,1)
    _,_,recon=task.rep.pose_to_hml(init,fix_ik_bug=True)
    reference_input=torch.as_tensor(normed,dtype=torch.float32,device=task.device)
    dataset_pose=task.rep.hml_to_pose(reference_input,recon,sim_at_hml_idx=0)
    dataset_sim=dataset_pose[:,:,smpl_2_mujoco].matmul(task.to_isaac_mat.T)
    assert dataset_sim.shape==(n,120,24,3)
    assert torch.isfinite(dataset_sim).all()
    original_goal=np.stack([dataset_sim[i,-1,task._body_names.index(r['goal_joint_body'])].detach().cpu().numpy()
                            for i,r in enumerate(cases)])
    original_goal[:,0]+=np.array([r['offset_m'] for r in cases])
    original_heading=heading(dataset_sim[:,-1].detach().cpu().numpy(),list(task._body_names))
    proposed_goal=original_goal.copy(); proposed_heading=original_heading.copy()
    if correction:
        assert correction['case_ids']==[r['case_id'] for r in cases]
        assert correction['original_goal_sha256']==hashlib.sha256(original_goal.tobytes()).hexdigest()
        proposed_goal=np.array(correction['proposed_goal'],dtype=np.float32)
        proposed_heading=np.array(correction['proposed_heading'],dtype=np.float32)
        assert np.max(np.linalg.norm(proposed_goal-original_goal,axis=1))<=.15001
        previous=json.loads((Path(correction['feedback_source'])/'metadata.json').read_text())
        assert previous['original_goal_sha256']==correction['original_goal_sha256']
        assert np.max(np.linalg.norm(proposed_goal-np.array(previous['proposed_goal']),axis=1))<=config['position_step_m']+1e-5
        from physics_guided_core import wrap_angle
        assert np.max(np.abs(wrap_angle(proposed_heading-np.array(previous['proposed_heading']))))<=np.deg2rad(config['heading_step_deg'])+1e-5
        assert np.max(np.abs(wrap_angle(proposed_heading-original_heading)))<=np.deg2rad(config['heading_total_deg'])+1e-5
    else:
        proposed_goal=proposed_goal.astype(np.float32)
    replay_world=None
    if replay is not None:
        replay_world=torch.as_tensor(replay['reference'],dtype=torch.float32,device=task.device).transpose(0,1)
        assert replay_world.shape==(n,t,b,3)
        assert np.isfinite(replay['reference']).all()
        assert np.allclose(replay['original_goal'],original_goal,atol=1e-5)
    goal=torch.as_tensor(proposed_goal,device=task.device)
    goal_heading=torch.as_tensor(proposed_heading,dtype=torch.float32,device=task.device)
    metadata=dict(cases=cases,names=list(task._body_names),dt=float(task.dt),episode_steps=t,context_steps=60,
                  original_goal=original_goal.tolist(),original_heading=original_heading.tolist(),
                  original_goal_sha256=hashlib.sha256(original_goal.tobytes()).hexdigest(),
                  proposed_goal=proposed_goal.tolist(),proposed_heading=proposed_heading.tolist(),
                  foot_geometry=foot_geometry(entry.asset_paths[id(task)]),seed=a.seed,mode=a.mode,round=a.round,
                  source_commit=config['source_commit'],coordinate_system='Isaac world, z up, mujoco body order',
                  goal_provenance='dataset motion, fixed before generation, endpoint at 4 seconds',
                  prefix_feedback='actual prefix and actual anchor' if a.mode=='native' else 'planned prefix and planned anchor',
                  time_alignment='action k uses reference frame k; next observation requests k+1; 120 frames with no duplicate frame0',
                  semantics='dataset text, independent generated semantics not yet verified',
                  weights_frozen=True)
    store.update(metadata=metadata,steps=0,finished=np.zeros(n,bool),terminated=np.zeros(n,bool),
                 valid=np.zeros((t+5,n),bool),pending=None,
                 actual=np.zeros((t+5,n,b,3),np.float32),reference=np.zeros((t+5,n,b,3),np.float32),
                 quaternion=np.zeros((t+5,n,b,4),np.float32),linvel=np.zeros((t+5,n,b,3),np.float32),
                 angvel=np.zeros((t+5,n,b,3),np.float32),contact_force=np.zeros((t+5,n,b,3),np.float32),
                 sample_count=0,provided_frame=None,reference_frame=np.full(t+5,-1,dtype=np.int32))
    generated={}
    def plan():
        # Official pose_buffer is updated after observations. Append the refreshed
        # current state here so both prefixes terminate at the same instant.
        context=torch.cat([task.pose_buffer[:,1:],task._rigid_body_pos[:,None]],dim=1) if a.mode=='native' else dataset_sim[:,30:60]
        anchor=context[:,-1,0]
        heading_tip=anchor+torch.stack([torch.cos(goal_heading),torch.sin(goal_heading),torch.zeros_like(goal_heading)],dim=1)
        points=torch.stack([goal,heading_tip,anchor],dim=1)
        hml_context,translated,recon_data=task.rep.pose_to_hml(context,points,fix_ik_bug=True)
        target=torch.zeros((n,task.num_joint_conditions,3),device=task.device)
        joint_names=[[r['goal_joint']] for r in cases]
        for i,r in enumerate(cases): target[i,task.extended_goal_joint_names.index(r['goal_joint'])]=translated[i,0]
        heading_vector=translated[:,1]-translated[:,2]
        target[:,task.extended_goal_joint_names.index('heading'),0]=torch.atan2(heading_vector[:,0],heading_vector[:,2])
        prefix=hml_context.unsqueeze(2).permute(0,3,2,1)
        y=dict(mask=torch.ones((n,1,1,40),dtype=torch.bool,device=task.device),prefix_len=20,prefix=prefix,
               target_cond=target,pred_target_cond=target,target_joint_names=joint_names,
               is_heading=torch.ones(n,dtype=torch.bool,device=task.device),
               scale=torch.ones(n,device=task.device)*task.mdm_cfg_param,text=[r['caption'] for r in cases])
        # Same batch ordering/noise/RNG state for paired arms. Candidate number
        # changes random draws only for the random-candidate baseline.
        candidate_seed=a.seed+10000*(a.round if a.noise_round is None else a.noise_round)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(candidate_seed); torch.cuda.manual_seed_all(candidate_seed)
            y['text_embed']=task.mdm.encode_text(y['text'])
            noise=torch.randn(task.mdm_tensor_shape,device=task.device)
            sample=task.sample_fn(task.mdm,task.mdm_tensor_shape,clip_denoised=False,
                                 model_kwargs={'y':y},skip_timesteps=0,init_image=None,progress=False,
                                 dump_steps=None,noise=noise,const_noise=False,cond_fn=None)
        generated['noise_sha256']=hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest()
        generated['prefix_sha256']=hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest()
        xyz=task.rep.hml_to_pose(sample.squeeze(2).permute(0,2,1),recon_data,sim_at_hml_idx=19)
        # Match the official 29:90 horizon extraction exactly.
        generated['previous']=xyz[:,29]
        generated['next']=xyz[:,30:90]
        assert generated['next'].shape==(n,60,24,3)
        store['sample_count']+=1
        metadata['generation_noise_sha256']=generated['noise_sha256']
        metadata['generation_prefix_sha256']=generated['prefix_sha256']
        # Exact point transform inverse, including translation and rotation.
        back=task.rep.inverse_align_to_recon_data(translated,recon_data)
        _,points_hml=task.rep.pose_to_hml_xyz(context,points)
        metadata['target_world_roundtrip_max_m']=float((back-points_hml).abs().max())
        assert metadata['target_world_roundtrip_max_m']<1e-4
    def get_pred_pose():
        # The inherited frame_idx increments after observations and duplicates
        # frame0. Use an explicit adapter clock, shared by every comparison arm.
        frame=store['steps']+(1 if store['pending'] is not None else 0)
        store['provided_frame']=min(frame,t-1)
        if a.mode=='replay':
            selected=replay_world[:,min(frame,t-1)]
            previous=replay_world[:,max(min(frame-1,t-1),0)]
            return {'j3d':task.rep.sim_pose_to_ref_pose(selected),'j3d_prev':task.rep.sim_pose_to_ref_pose(previous),'dt':task.dt}
        if frame<60:
            return {'j3d':dataset_pose[:,frame],'j3d_prev':dataset_pose[:,max(frame-1,0)],'dt':task.dt}
        if not generated: plan()
        j=min(frame-60,59)
        previous=generated['previous'] if j==0 else generated['next'][:,j-1]
        return {'j3d':generated['next'][:,j],'j3d_prev':previous,'dt':task.dt}
    task.get_pred_pose=get_pred_pose
    pre=task.pre_physics_step
    def before(actions):
        assert store['provided_frame']==store['steps'], 'Applied reference/physics clock mismatch'
        store['reference_frame'][store['steps']]=store['provided_frame']
        store['applied_ref']=task.ref_body_pos.detach().cpu().numpy().copy()
        return pre(actions)
    task.pre_physics_step=before
    reward=task._compute_reward
    def record(actions):
        result=reward(actions)
        store['pending']={
            'actual':task._rigid_body_pos.detach().cpu().numpy().copy(),
            'reference':store['applied_ref'],
            'quaternion':task._rigid_body_rot.detach().cpu().numpy().copy(),
            'linvel':task._rigid_body_vel.detach().cpu().numpy().copy(),
            'angvel':task._rigid_body_ang_vel.detach().cpu().numpy().copy(),
            'contact_force':task._contact_forces.detach().cpu().numpy().copy()}
        return result
    task._compute_reward=record
    post=task.post_physics_step
    def after():
        result=post(); k=store['steps']; active=~store['finished']
        assert store['pending'] is not None
        for name,value in store['pending'].items():
            assert np.isfinite(value[active]).all(),name
            store[name][k]=value
        store['valid'][k]=active
        done=task.reset_buf.detach().cpu().numpy().astype(bool)&active
        term=task._terminate_buf.detach().cpu().numpy().astype(bool)&done
        store['terminated']|=term
        store['finished']|=done
        if k+1==t: store['finished'][:]=True
        store['steps']+=1; store['pending']=None
        if store['steps']%30==0:
            print('PHYSICS_GUIDED_PROGRESS',store['steps'],int(store['finished'].sum()),round(time.time()-started,2),flush=True)
        if store['finished'].all(): raise CollectionComplete()
        return result
    task.post_physics_step=after
    # Deterministic reference replay fixture; inverse coordinate mapping must
    # reproduce every body, not merely a root translation.
    back=task.rep.ref_pose_to_sim_pose(task.rep.sim_pose_to_ref_pose(dataset_sim))
    assert float((back-dataset_sim).abs().max())<1e-5
    metadata['body_coordinate_roundtrip_max_m']=float((back-dataset_sim).abs().max())
    return env


entry.official.create_rlgpu_env=create
overrides=['learning=im_big','robot=smpl_humanoid','env=closd_t2m',
           'exp_name=CLoSD_multitask_finetune','epoch=-1','test=True','no_virtual_display=True',
           'headless=True','no_log=True','debug=False','seed='+str(a.seed),
           'env.num_envs='+str(len(cases)),'env.episode_length=121','env.save_motion.save_hml_episodes=False',
           'env.dip.model_path=closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt',
           'env.dip.cfg_param=7.5','learning.params.config.player.games_num=10000000']
with initialize_config_dir(version_base=None,config_dir=str(entry.SOURCE/'closd/data/cfg')):
    cfg=compose(config_name='config',overrides=overrides)
(out/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg))
complete=False
try:
    entry.official.main.__wrapped__(cfg)
    raise RuntimeError('Runner returned before all first attempts')
except CollectionComplete:
    complete=True
finally:
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for task in entry.captured:
        if getattr(task,'viewer',None) is not None: task.gym.destroy_viewer(task.viewer); task.viewer=None
        if getattr(task,'sim',None) is not None: task.gym.destroy_sim(task.sim); task.sim=None
    print('PHYSICS_GUIDED_NATIVE_CLEANUP',flush=True)
if complete:
    fields=['actual','reference','quaternion','linvel','angvel','contact_force','valid']
    arrays={key:store[key][:store['steps']] for key in fields}
    arrays['terminated']=store['terminated']
    arrays['reference_frame']=store['reference_frame'][:store['steps']]
    assert np.array_equal(arrays['reference_frame'],np.arange(store['steps'])), 'Duplicate/skipped reference time'
    np.savez_compressed(out/'trajectories.npz',**arrays)
    metadata=store['metadata']; metadata['elapsed_seconds']=time.time()-started
    metadata['generation_batch_calls']=store['sample_count']
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[metrics(arrays,i,metadata,thresholds) for i in range(len(cases))]
    (out/'metrics.json').write_text(json.dumps(rows,indent=2))
    (out/'receipt.json').write_text(json.dumps(dict(status='complete',cases=len(cases),
        completed=sum(r['completed'] for r in rows),joint_pass=sum(r['joint_pass'] for r in rows),
        elapsed_seconds=time.time()-started,trajectories_sha256=hashlib.sha256((out/'trajectories.npz').read_bytes()).hexdigest()),indent=2))
    if a.request_file:
        from physics_guided_lineage import file_hash
        request=json.loads(Path(a.request_file).read_text())
        assert (request['mode'],request['seed'],request['round'])==(a.mode,a.seed,a.round)
        assert request['manifest']==file_hash(a.manifest)
        assert request['config']==file_hash(root/'configs/physics_guided_20261011.json')
        assert request['thresholds']==file_hash(a.thresholds)
        binding=dict(request=request,outputs={name:file_hash(out/name) for name in ('metadata.json','metrics.json','trajectories.npz')},
                     certification='bound_request_at_execution',confirmatory=False,
                     study_status='amended diagnostic run; thresholds retained')
        (out/'lineage.json').write_text(json.dumps(binding,indent=2))
    print('PHYSICS_GUIDED_COLLECTION_PASS',len(rows),sum(r['joint_pass'] for r in rows),flush=True)
