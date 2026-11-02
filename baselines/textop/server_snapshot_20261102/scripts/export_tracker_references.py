"""Export complete private GMR trajectories with actual Isaac articulation FK.

Does not advance dynamics or accept motions. Name-based mapping is mandatory:
GMR has38 bodies while the merged Isaac articulation has30. Outputs remain
private; only aggregate receipt is suitable for public reporting.
"""
import argparse,hashlib,json,pathlib,shutil,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--manifest',action='append',required=True)
p.add_argument('--output',required=True);p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=128)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','exported_source_sequences':0,'exported_trajectories':0,'exported_frames':0,
         'physical_filter_executed':False,'dataset_accepted':False,'optimizer_updates':0,'output_writers_closed':False}
app=None;sim=None
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2))
try:
    assert a.headless and a.device=='cuda:1' and 1<=a.batch_size<=512
    spec=json.loads(pathlib.Path(a.spec).read_text());assert sha(spec['robot_urdf'])==spec['robot_urdf_sha256']
    records=[]
    for name in a.manifest:
        manifest=pathlib.Path(name);progress=json.loads((manifest.parent/'progress.json').read_text())
        assert progress['status']=='PREPROCESSING_FINISHED_FILTER_PENDING',('Input cohort not complete',name)
        assert progress['failed_sequences']==0
        rows=[json.loads(x) for x in manifest.read_text().splitlines()]
        assert len(rows)==progress['completed_sequences'];records+=rows
    assert len({x['canonical_source'] for x in records})==len(records),'Duplicate source across cohorts'
    if a.limit:records=records[:a.limit]
    assert records
    receipt.update(input_manifest_hashes=[sha(x) for x in a.manifest],target_sources=len(records),
                   source_sha256=sha(__file__),urdf_sha256=sha(spec['robot_urdf']))
    save();app=AppLauncher(a).app
    import numpy as np,torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
    from isaaclab.utils import configclass
    from textop_tracker.robots.g1 import G1_CYLINDER_CFG
    torch.set_num_threads(4);torch.manual_seed(20261025)
    torch.cuda.set_per_process_memory_fraction(.65,torch.device(a.device))
    robot_cfg=G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    robot_cfg.spawn.asset_path=spec['robot_urdf'];robot_cfg.spawn.usd_dir=str(out/'usd');robot_cfg.spawn.usd_file_name='g1.usd'
    @configclass
    class Scene(InteractiveSceneCfg):robot:ArticulationCfg=robot_cfg
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=a.device))
    scene=InteractiveScene(Scene(num_envs=a.batch_size,env_spacing=2.5));sim.reset();robot=scene['robot']
    assert len(robot.joint_names)==29 and len(robot.body_names)==30
    receipt.update(joint_names=robot.joint_names,body_names=robot.body_names,fps=50,
                   quaternion_convention='WXYZ, world-frame angular velocity',physics_steps=0)
    def angular_velocity(q,dt):
        prev=q[:-1];nxt=q[1:]
        w=np.sum(prev*nxt,axis=-1)
        v=prev[...,:1]*nxt[...,1:]-nxt[...,:1]*prev[...,1:]+np.cross(prev[...,1:],nxt[...,1:])
        sign=np.where(w<0,-1.,1.);w=w*sign;v=v*sign[...,None]
        norm=np.linalg.norm(v,axis=-1)
        adjacent=v*(2*np.arctan2(norm,w)/np.maximum(norm,1e-12))[...,None]/dt
        return np.concatenate([adjacent[:1],(adjacent[:-1]+adjacent[1:])*.5,adjacent[-1:]],axis=0)
    with (out/'manifest.private.jsonl').open('x') as manifest:
        for entry in records:
            assert sha(entry['output_file'])==entry['output_sha256']
            assert len(entry['joint_names'])==29
            order=[entry['joint_names'].index(n) for n in robot.joint_names]
            with np.load(entry['output_file'],allow_pickle=False) as data:
                for mirrored,key in [(False,'qpos_29'),(True,'mirrored_qpos_29')]:
                    assert shutil.disk_usage(out).free>2*1024**3,'Reserve reached; inputs and partial outputs retained'
                    q=np.asarray(data[key],dtype=np.float32);assert q.shape==(entry['frames50'],36) and len(q)>=2
                    jp=q[:,7:][:,order];positions=[];quaternions=[]
                    for start in range(0,len(q),a.batch_size):
                        n=min(a.batch_size,len(q)-start);indices=np.minimum(np.arange(a.batch_size)+start,len(q)-1)
                        root=robot.data.default_root_state.clone();root[:,:3]=torch.as_tensor(q[indices,:3],device=a.device)+scene.env_origins
                        root[:,3:7]=torch.as_tensor(q[indices,3:7],device=a.device);root[:,7:]=0
                        joints=torch.as_tensor(jp[indices],device=a.device)
                        robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(joints,torch.zeros_like(joints))
                        sim.forward();scene.update(.005)
                        assert torch.allclose(robot.data.joint_pos,joints,atol=1e-5)
                        positions.append((robot.data.body_pos_w-scene.env_origins[:,None,:])[:n].cpu().numpy())
                        quaternions.append(robot.data.body_quat_w[:n].cpu().numpy())
                    bp=np.concatenate(positions);bq=np.concatenate(quaternions)
                    assert np.isfinite(bp).all() and np.isfinite(bq).all()
                    assert np.max(np.abs(np.linalg.norm(bq,axis=-1)-1))<1e-4
                    name=hashlib.sha256((entry['canonical_source']+'|'+str(mirrored)).encode()).hexdigest()[:24]
                    dest=out/(name+'.private.npz')
                    np.savez_compressed(dest,fps=np.array([50]),joint_pos=jp,joint_vel=np.gradient(jp,.02,axis=0),
                        body_pos_w=bp,body_quat_w=bq,body_lin_vel_w=np.gradient(bp,.02,axis=0),body_ang_vel_w=angular_velocity(bq,.02))
                    manifest.write(json.dumps({'canonical_source':entry['canonical_source'],'split':entry['split'],
                        'mirrored':mirrored,'file':str(dest),'sha256':sha(dest),'frames':len(q),
                        'prepared_file':entry['output_file'],'prepared_sha256':entry['output_sha256'],
                        'annotations':entry['mirrored_annotations'] if mirrored else entry['annotations'],
                        'physical_filter_passed':False})+'\n');manifest.flush()
                    receipt['exported_trajectories']+=1;receipt['exported_frames']+=len(q)
            receipt['exported_source_sequences']+=1;save()
            if receipt['exported_source_sequences']%10==0:print(json.dumps(receipt),flush=True)
    receipt.update(status='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED',output_writers_closed=True,
        private_manifest_sha256=sha(out/'manifest.private.jsonl'),
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(a.device)))
    save();print('TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED',flush=True)
except BaseException as error:
    receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app,receipt,save,out,simulation=sim,successful_statuses=('TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED',))
