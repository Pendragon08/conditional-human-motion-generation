"""Headless adaptation of TextOp's official MuJoCo controller for dance.

The official observation, joint-order, actuator gains and PD functions are
compiled directly from the authenticated upstream file. The original viewer
and infinite-interactive loop are replaced with bounded, recorded test cases.
"""
import argparse, ast, hashlib, json, os, pathlib, time, traceback

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN = 'user_mixed5000_official_mujoco_dance_v1'
SUPERVISOR = ROOT/'scripts/supervise_mixed5000_official_mujoco_v1.py'
OFFICIAL = ROOT/'upstream/TextOpTracker/scripts/deploy_mujoco.py'
MATH = pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/IsaacLab/source/isaaclab/isaaclab/utils/math.py')
ASSET = pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml')


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def atomic(path, value):
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def official_functions(np, torch):
    from enum import Enum
    if sha(OFFICIAL)!='9e11c0ea8f869a88a9cb0255bbb1a63b3d3f50ee1eee891a92aff5595defbf04':
        raise ValueError('official controller source differs')
    if sha(MATH)!='3c58e51afa3528d482d167af80b00cce33a87829f9287547b1bcdb84bf4e1fa8':
        raise ValueError('Isaac math source differs')
    namespace=dict(np=np,torch=torch,Enum=Enum)
    math_tree=ast.parse(MATH.read_bytes())
    needed={'normalize','matrix_from_quat','quat_conjugate','quat_inv','quat_mul','quat_apply','subtract_frame_transforms'}
    selected=[x for x in math_tree.body if isinstance(x,ast.FunctionDef) and x.name in needed]
    if {x.name for x in selected}!=needed:raise ValueError('incomplete upstream quaternion operations')
    for node in selected:node.decorator_list=[]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(MATH),'exec'),namespace)
    tree=ast.parse(OFFICIAL.read_bytes());selected=[]
    assignments={'anchor_body','isaaclab_joint_names','mujoco_joint_names','stiffness_dict','damping_dict',
                 'scale_dict','joint_pos_config','isaaclab_to_mujoco_reindex','mujoco_to_isaaclab_reindex',
                 'joint_names','kps','kds','action_scale','default_angles'}
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in {'AnchorBody','MotionLoader'}:selected.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name not in {'__fix__add_marker_to_scene','update_joint_visualization'}:selected.append(node)
        elif isinstance(node,ast.Assign) and all(isinstance(x,ast.Name) and x.id in assignments for x in node.targets):selected.append(node)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(OFFICIAL),'exec'),namespace)
    for name in ('MotionLoader','compute_observation','pd_control','isaaclab_joint_names','mujoco_joint_names',
                 'isaaclab_to_mujoco_reindex','kps','kds','action_scale','default_angles'):
        if name not in namespace:raise ValueError('official controller part missing: '+name)
    return namespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args()
    request=pathlib.Path(a.request)
    if not __debug__ or os.getuid()!=1013 or request.resolve()!=request or request.parent!=ROOT/'receipts' or sha(request)!=a.trusted_request_sha256:
        raise ValueError('request identity')
    job=json.loads(request.read_text())
    if job['run']!=RUN or not job['authorized'] or sha(__file__)!=job['executor_sha256']:raise ValueError('wrong request')
    for name,digest in job['bound_sources'].items():
        path=ROOT/name
        if path.resolve()!=path or path.stat().st_uid!=1013 or sha(path)!=digest:raise ValueError('source changed: '+name)
    from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
    pid=os.getppid();found=False
    for _ in range(8):
        if pid<=1:break
        proc=pathlib.Path('/proc')/str(pid);argv=(proc/'cmdline').read_bytes().decode().split('\0')
        if proc.stat().st_uid==1013 and str(SUPERVISOR) in argv and str(request) in argv:found=True;break
        pid=int((proc/'stat').read_text().rsplit(')',1)[1].split()[1])
    st=(ROOT/'receipts/full_library_capacity.lock').stat()
    if not found or not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(),pid,os.major(st.st_dev),os.minor(st.st_dev),st.st_ino):
        raise ValueError('supervisor must hold common lock')
    if set(os.sched_getaffinity(0))!={0,1,2,3}:raise ValueError('CPU affinity')
    from mixed5000_dance_tracking_inputs_v1 import accept
    accepted=accept(job['input_binding']);rows=accepted['rows']
    if len(rows)!=16 or sha(ASSET)!=job['asset_sha256']:raise ValueError('inputs differ')
    for name,digest in job['mesh_hashes'].items():
        mesh=pathlib.Path(name)
        if mesh.resolve()!=mesh or ASSET.parent not in mesh.parents or sha(mesh)!=digest:raise ValueError('G1 mesh differs')
    policy=ROOT/'step34_20261029/user_mixed5000_onnx_export_v1/policy.onnx'
    if sha(policy)!=job['onnx_sha256']:raise ValueError('ONNX policy differs')
    out=ROOT/'step34_20261029'/RUN;out.mkdir(mode=0o700,exist_ok=False);start=time.monotonic()
    rec=dict(status='STARTING_MIXED5000_OFFICIAL_MUJOCO_DANCE',run=RUN,request_sha256=sha(request),
             actor_sha256=accepted['actor_sha256'],onnx_sha256=job['onnx_sha256'],reference_count=16,
             completed_trials=0,optimizer_updates=0,output_writers_closed=False,physical_tracking_executed=False,
             source_kind='original_val_dance',policy_hz=50,mujoco_timestep_seconds=.002,
             asset_kind='GMR 29-DoF G1 adaptation; official g1_act.xml is absent',
             paper_exact_asset_verified=False,official_controller_used=True)
    def save():rec['elapsed_seconds']=time.monotonic()-start;atomic(out/'receipt.json',rec)
    save()
    try:
        import numpy as np
        import mujoco
        import onnxruntime as ort
        import torch
        from textop_paper_tracking_metrics_v1 import measure
        torch.set_num_threads(4)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only required')
        official=official_functions(np,torch)
        model=mujoco.MjModel.from_xml_path(str(ASSET));model.opt.timestep=.002
        if (model.nq,model.nv,model.nu)!=(36,35,29):raise ValueError('G1 model dimensions')
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,model.njnt)]
        if names!=official['mujoco_joint_names']:raise ValueError('MuJoCo joint order differs')
        fkrec=json.loads((ROOT/'step34_20261029/user_mixed5000_dance_fk_library_v1/receipt.json').read_text())
        bodies=fkrec['body_names'];ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in bodies]
        if len(ids)!=30 or len(set(ids))!=30 or any(x<0 for x in ids) or bodies[0]!='pelvis':raise ValueError('body name mapping')
        if fkrec['joint_names']!=official['isaaclab_joint_names']:raise ValueError('reference joint ordering')
        opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
        session=ort.InferenceSession(str(policy),sess_options=opts,providers=['CPUExecutionProvider'])
        if session.get_providers()!=['CPUExecutionProvider'] or len(session.get_inputs())!=1:raise ValueError('ONNX provider')
        input_name=session.get_inputs()[0].name;results=[];manifest=[]
        for index,row in enumerate(rows):
            loader=official['MotionLoader'](row['file']);frames=loader.T
            if frames!=row['frames'] or frames<3 or int(np.asarray(loader.fps).reshape(-1)[0])!=50:raise ValueError('reference format')
            data=mujoco.MjData(model);action=np.zeros(29,dtype=np.float32);target=official['default_angles'].copy()
            data.qpos[7:]=loader.joint_pos[0][official['isaaclab_to_mujoco_reindex']]
            data.qpos[:3]=loader.body_pos[0][0];data.qpos[3:7]=loader.body_ori[0][0]
            mujoco.mj_step(model,data)
            poses=[];executed=[];reference=[]
            for t in range(frames):
                for _ in range(10):
                    tau=official['pd_control'](target,data.qpos[7:],official['kps'],np.zeros(29),data.qvel[6:],official['kds'])
                    data.ctrl[:]=tau;mujoco.mj_step(model,data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.xpos[ids]).all():break
                poses.append(data.qpos.copy());executed.append(data.xpos[ids].copy());reference.append(loader.body_pos[t].copy())
                obs=official['compute_observation'](data,loader,t,action)
                if obs.shape!=(431,) or not np.isfinite(obs).all():raise ValueError('official observation invalid')
                action=session.run(None,{input_name:np.asarray(obs,dtype=np.float32).reshape(1,-1)})[0].reshape(-1)
                if action.shape!=(29,) or not np.isfinite(action).all():raise ValueError('ONNX action invalid')
                target=action[official['isaaclab_to_mujoco_reindex']]*official['action_scale']+official['default_angles']
            if len(poses)<3:raise RuntimeError('fewer than three finite simulation frames')
            metrics=measure(executed,reference,frames)
            file=out/('dance_%02d.private.npz'%index)
            with file.open('xb') as f:
                os.chmod(file,0o600);np.savez_compressed(f,qpos=np.asarray(poses,dtype=np.float32),
                    executed_body_pos=np.asarray(executed,dtype=np.float32),reference_body_pos=np.asarray(reference,dtype=np.float32));f.flush();os.fsync(f.fileno())
            if file.stat().st_size>16*1024**2:raise ValueError('trace budget')
            results.append(dict(index=index,segment_id=row['segment_id'],mode=row['mode'],metrics=metrics))
            manifest.append(dict(index=index,file=str(file),sha256=sha(file),bytes=file.stat().st_size,frames=len(poses)))
            rec.update(status='RUNNING_MIXED5000_OFFICIAL_MUJOCO_DANCE',completed_trials=len(results),physical_tracking_executed=True);save()
        summary=dict(trials=len(results),success=sum(x['metrics']['success'] for x in results),
                     complete=sum(x['metrics']['complete'] for x in results),incomplete=sum(not x['metrics']['complete'] for x in results),
                     max_error_threshold_m=.3,body_count=30,
                     metric_scope='mean per-link norm over observed frames; incomplete trials count as non-success')
        atomic(out/'results.private.json',results);atomic(out/'trace_manifest.private.json',manifest)
        rec.update(status='MIXED5000_OFFICIAL_MUJOCO_DANCE_FINISHED',completed_trials=len(results),summary=summary,
                   results_sha256=sha(out/'results.private.json'),trace_manifest_sha256=sha(out/'trace_manifest.private.json'),output_writers_closed=True);save()
    except BaseException as error:
        rec.update(status='MIXED5000_OFFICIAL_MUJOCO_DANCE_FAILED',error=repr(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
