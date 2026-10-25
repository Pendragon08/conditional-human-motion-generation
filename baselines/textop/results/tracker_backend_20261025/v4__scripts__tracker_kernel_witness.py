"""Seeded real G1 FK/dynamics dispatch in existing IsaacLab, zero PPO updates.

The diagnostic intentionally has no ground: it certifies backend dispatch,
joint conventions and numerical gradients, not contact stability or tracking.
Private reference arrays stay in the task-local output, never in Git.
"""
import argparse,datetime,hashlib,importlib.metadata,json,os,pathlib,sys,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
spec=json.loads(pathlib.Path(args.spec).read_text());out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'physical_filter_executed':False,'full_filter_training_ready':False}
app=None;sim=None

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def save():
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))

try:
    assert args.headless and args.device=='cuda:1'
    assert '.'.join(map(str,sys.version_info[:2]))==spec['python_major_minor']
    for key,version in spec['required_versions'].items():assert importlib.metadata.version(key)==version
    assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
    receipt.update(spec_canonical_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8],
        spec_file_sha256=sha(args.spec),source_sha256=sha(__file__),urdf_sha256=sha(spec['robot_urdf']))
    save();app=AppLauncher(args).app;print('TEXTOP_ISAAC_APP_LAUNCHED',flush=True)
    import numpy as np,torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
    from isaaclab.utils import configclass
    from textop_tracker.robots.g1 import G1_CYLINDER_CFG
    torch.set_num_threads(4);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
    torch.cuda.set_per_process_memory_fraction(spec['resources']['torch_memory_fraction'],torch.device(args.device))
    robot_cfg=G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    robot_cfg.spawn.asset_path=spec['robot_urdf'];robot_cfg.spawn.usd_dir=str(out/'usd');robot_cfg.spawn.usd_file_name='g1.usd'
    @configclass
    class Scene(InteractiveSceneCfg):
        robot:ArticulationCfg=robot_cfg
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
    scene=InteractiveScene(Scene(num_envs=8,env_spacing=2.5));sim.reset();robot=scene['robot']
    try:
        import omni.replicator.core as rep
    except ModuleNotFoundError:
        rep=None
    if rep is not None:rep.orchestrator.set_capture_on_play(False)
    assert len(robot.joint_names)==29
    motion_root=pathlib.Path(spec['root'])/'data_preparation_20261025/available500_v1'
    records=[json.loads(line) for line in (motion_root/'prepared_manifest.private.jsonl').read_text().splitlines()]
    item=next(x for x in records if x['frames50']>=8)
    data=np.load(item['output_file'],allow_pickle=False);q=np.asarray(data['qpos_29'][:8],dtype=np.float32)
    assert q.shape==(8,36)
    indices=robot.find_joints(item['joint_names'],preserve_order=True)[0];assert len(indices)==29
    root=robot.data.default_root_state.clone();root[:,:3]=torch.as_tensor(q[:,:3],device=args.device)+scene.env_origins
    root[:,3:7]=torch.as_tensor(q[:,3:7],device=args.device);root[:,7:]=0
    jp=robot.data.default_joint_pos.clone();jp[:,indices]=torch.as_tensor(q[:,7:],device=args.device)
    robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(jp,torch.zeros_like(jp))
    sim.forward();scene.update(.005)
    assert torch.allclose(robot.data.joint_pos[:,indices],torch.as_tensor(q[:,7:],device=args.device),atol=1e-5)
    assert torch.allclose(robot.data.root_pos_w,root[:,:3],atol=1e-5)
    fk_pos=robot.data.body_pos_w-scene.env_origins[:,None,:]
    fk_quat=robot.data.body_quat_w
    assert torch.isfinite(fk_pos).all() and torch.isfinite(fk_quat).all()
    np.savez_compressed(out/'reference_fk.private.npz',qpos=q,body_pos_w=fk_pos.cpu().numpy(),body_quat_w=fk_quat.cpu().numpy())
    initial=robot.data.root_pos_w.clone()
    for step in range(spec['runtime_smoke']['control_steps']):
        targets=jp.clone();targets[:,indices[3]]+=.01 if step%2 else -.01
        robot.set_joint_position_target(targets);scene.write_data_to_sim()
        for _ in range(4):sim.step(render=False);scene.update(.005)
        assert torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all()
    delta=float(torch.max(torch.abs(robot.data.root_pos_w-initial)).item());assert delta>1e-4
    policy=torch.nn.Sequential(torch.nn.Linear(557,2048),torch.nn.ELU(),torch.nn.Linear(2048,1024),torch.nn.ELU(),
        torch.nn.Linear(1024,512),torch.nn.ELU(),torch.nn.Linear(512,29)).to(args.device)
    x=torch.randn(8,557,device=args.device);y=policy(x);y.square().mean().backward()
    assert y.shape==(8,29) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in policy.parameters())
    assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
    receipt.update(status='TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED',num_envs=8,physics_steps=32,dt=.005,control_hz=50,
        dof=29,body_count=len(robot.body_names),joint_names=robot.joint_names,body_names=robot.body_names,
        real_state_transition_max_m=delta,seeded_paper_MLP_forward_backward_passed=True,MLP_input_dim=557,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)),
        private_reference_sha256=item['output_sha256'],ground_present=False,tracking_success_evaluated=False,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save();print('TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED',flush=True)
except BaseException as error:
    receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
    if app is not None:
        from bounded_headless_close import close
        receipt['shutdown_helper_sha256']=sha(pathlib.Path(__file__).with_name('bounded_headless_close.py'))
        close(app,receipt,save,out,simulation=sim)
