"""Exploratory CPU scene replay with PD, not a GRAB physical acceptance test.

Apple/table follow recorded poses kinematically. Their motion is not affected by
robot forces. The robot begins from a retargeted pose with zero velocity.
"""
import json,sys,zipfile
from collections import Counter
from pathlib import Path
import mujoco
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
import trimesh

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_mujoco_object_20261030'))
from grab_s3_mujoco_object_distance import sha,chosen,load,read_grab_field
sys.path.insert(0,str(ROOT/'grab_s3_scene_timeline_v2_20261030'))
from grab_s3_scene_timeline import load_table,mesh_for
sys.path.insert(0,str(ROOT/'pyroki_window_blend_v2_20261030'))
from resample_g1_27 import resample_27
sys.path.insert(0,str(ROOT/'grab_hand_independent_distance_20261030'))
from grab_hand_independent_distance import bracket

def object_hulls(obj,table):
    ap,ao,ah=mesh_for(obj['object_mesh']);tp,to,th=mesh_for(table['table_mesh'])
    v=np.asarray(ah.vertices);count=512;i=np.arange(count);z=1-2*(i+.5)/count;theta=i*np.pi*(3-np.sqrt(5))
    directions=np.column_stack((np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z))
    idx=np.unique([np.argmax(v@d) for d in directions]);small=trimesh.Trimesh(vertices=v[idx],process=False).convex_hull
    sv=np.asarray(small.vertices)
    if cKDTree(v).query(sv)[0].max()>1e-10:raise ValueError('apple hull not inscribed')
    bound=float(cKDTree(sv).query(v)[0].max())
    return {'apple':small,'table':th},{'apple':ap,'table':tp},bound

def prepare_scene(scene,hulls):
    spec=mujoco.MjSpec.from_file(str(scene))
    for name,h in hulls.items():
        mesh=spec.add_mesh(name='grab_'+name+'_hull')
        mesh.uservert=np.asarray(h.vertices,np.float64).reshape(-1).tolist()
        mesh.userface=np.asarray(h.faces,np.int32).reshape(-1).tolist()
        body=spec.worldbody.add_body(name='grab_'+name,mocap=True)
        body.add_geom(name='grab_'+name+'_geom',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='grab_'+name+'_hull',contype=1,conaffinity=1)
    return spec.compile()

def identifier(model,typ,name):
    n=mujoco.mj_name2id(model,typ,name)
    if n<0:raise ValueError('missing '+name)
    return n

def replay(scene,hulls,entries,frames,ref,names,cfg,mode):
    model=prepare_scene(scene,hulls);model.opt.timestep=.002
    objids={n:identifier(model,mujoco.mjtObj.mjOBJ_GEOM,'grab_'+n+'_geom') for n in hulls}
    mocaps={n:int(model.body_mocapid[identifier(model,mujoco.mjtObj.mjOBJ_BODY,'grab_'+n)]) for n in hulls}
    if min(mocaps.values())<0:raise ValueError('expected mocap objects')
    if mode=='objects_disabled':
        for g in objids.values():model.geom_contype[g]=0;model.geom_conaffinity[g]=0
    elif mode!='objects_enabled':raise ValueError('invalid mode')
    data=mujoco.MjData(model);controller=[n[:-5]+'_joint' if n.startswith(('left_hand_','right_hand_')) and n.endswith('_link') else n for n in cfg['joint_names']]
    kp=np.asarray(cfg['kps'],float);kd=np.asarray(cfg['kds'],float)
    if len(controller)!=len(kp) or len(controller)!=len(kd) or len(set(controller))!=len(controller) or not np.isfinite(kp).all() or not np.isfinite(kd).all():raise ValueError('invalid gains')
    joints=[identifier(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in controller]
    actuators=[identifier(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in controller]
    if len(controller)!=model.nu or len(set(actuators))!=model.nu or len(names)!=27 or not set(names)<=set(controller):raise ValueError('control layout mismatch')
    index={n:i for i,n in enumerate(names)}
    root=identifier(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');raddr=int(model.jnt_qposadr[root]);floor=identifier(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    data.qpos[raddr:raddr+3]=ref[0,4:7];data.qpos[raddr+3:raddr+7]=ref[0,:4]
    for n,j in zip(controller,joints):data.qpos[model.jnt_qposadr[j]]=ref[0,7+index[n]] if n in index else 0.
    data.qvel[:]=0
    times=np.arange(len(frames),dtype=float)/120.
    interpolators={}
    for n,entry in entries.items():
        trans=np.asarray(entry['params']['transl'],float)[frames];orient=np.asarray(entry['params']['global_orient'],float)[frames]
        if trans.shape!=(len(frames),3) or orient.shape!=trans.shape or not (np.isfinite(trans).all() and np.isfinite(orient).all()):raise ValueError('invalid '+n+' timeline')
        interpolators[n]=(trans,Slerp(times,Rotation.from_rotvec(orient).inv()))
    def set_poses(t):
        for n,(trans,slerp) in interpolators.items():
            pos=np.array([np.interp(t,times,trans[:,k]) for k in range(3)]);q=slerp(t).as_quat()
            data.mocap_pos[mocaps[n]]=pos;data.mocap_quat[mocaps[n]]=q[[3,0,1,2]]
    set_poses(0.);mujoco.mj_forward(model,data)
    transform_error={}
    for n,g in objids.items():
        m=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[m]);count=int(model.mesh_vertnum[m])
        actual=data.geom_xpos[g]+model.mesh_vert[a:a+count]@data.geom_xmat[g].reshape(3,3).T
        t=entries[n]['params']; expected=Rotation.from_rotvec(np.asarray(t['global_orient'])[frames[0]]).inv().apply(hulls[n].vertices)+np.asarray(t['transl'])[frames[0]]
        err=max(float(cKDTree(actual).query(expected)[0].max()),float(cKDTree(expected).query(actual)[0].max()))
        if err>1e-4:raise ValueError('object transform mismatch '+n)
        transform_error[n]=err
    dt=float(model.opt.timestep);steps=min(int(.9/dt),int((len(ref)-1)/20/dt));stats=Counter();first={};force_max=Counter();root_err=[];joint_err=[];min_z=float('inf');saturations=0;body_pairs=Counter();contact_audits=[]
    feet={'left_ankle_roll_link','right_ankle_roll_link'}
    for step in range(steps):
        t=step*dt;set_poses(t);k=min(int(t*20),len(ref)-2);a=t*20-k
        desired=(1-a)*ref[k,7:]+a*ref[k+1,7:];desired_root=(1-a)*ref[k,4:7]+a*ref[k+1,4:7];errors=[]
        for j,act,n,p,d in zip(joints,actuators,controller,kp,kd):
            target=float(desired[index[n]]) if n in index else 0.;position=float(data.qpos[model.jnt_qposadr[j]]);velocity=float(data.qvel[model.jnt_dofadr[j]])
            u=p*(target-position)-d*velocity
            if model.jnt_actfrclimited[j]:
                low,high=model.jnt_actfrcrange[j];clamped=float(np.clip(u,low,high));saturations+=int(clamped!=u);u=clamped
            data.ctrl[act]=u
            if n in index:errors.append(position-target)
        mujoco.mj_step(model,data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isclose(data.time,(step+1)*dt,atol=1e-9,rtol=0)):raise ValueError('nonfinite state or physics reset')
        root_err.append(float(np.linalg.norm(data.qpos[raddr:raddr+3]-desired_root)));joint_err.append(float(np.sqrt(np.mean(np.square(errors)))));min_z=min(min_z,float(data.qpos[raddr+2]))
        seen=set()
        for ci in range(data.ncon):
            c=data.contact[ci];g1,g2=int(c.geom1),int(c.geom2);pair={g1,g2}
            if floor in pair:
                g=g2 if g1==floor else g1;bn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
                cls='floor_foot' if bn in feet else 'floor_nonfoot'
            elif objids['apple'] in pair:
                g=g2 if g1==objids['apple'] else g1;bn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
                cls='apple_right_hand' if bn.startswith(('right_hand_','right_wrist_')) else 'apple_other'
            elif objids['table'] in pair:
                g=g2 if g1==objids['table'] else g1;bn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
                cls='table_other'
            else:continue
            seen.add(cls);body_pairs[(cls,bn)]+=1;f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f);force_max[cls]=max(force_max[cls],float(f[0]))
            if cls in ('apple_right_hand','apple_other','table_other'):
                observed=objids['table'] if cls=='table_other' else objids['apple']
                other=g2 if g1==observed else g1
                meshid=int(model.geom_dataid[observed]);adr=int(model.mesh_vertadr[meshid]);count=int(model.mesh_vertnum[meshid])
                world=data.geom_xpos[observed]+model.mesh_vert[adr:adr+count]@data.geom_xmat[observed].reshape(3,3).T
                distance=bracket(world,model,data,other)
                discrepancy=distance['lower_m']-max(0.,float(c.dist))
                contact_audits.append({'step':step,'time_s':float(data.time),'class':cls,'body':bn,
                                       'reported_distance_m':float(c.dist),'reported_normal_force_N':float(f[0]),
                                       'independent':distance,'inconsistent':bool(discrepancy>1e-4)})
        for cls in seen:
            stats[cls]+=1
            if cls not in first:first[cls]=float(data.time)
    return {'mode':mode,'steps':steps,'dt':dt,'duration_s':steps*dt,'gravity':model.opt.gravity.tolist(),
            'objects_transform_max_error_m':transform_error,'contact_step_fraction':{k:v/steps for k,v in stats.items()},
            'contact_first_s':first,'max_reported_normal_force_N':dict(force_max),
            'contact_pair_counts':{k[0]+' | '+k[1]:v for k,v in body_pairs.items()},
            'mean_root_error_m':float(np.mean(root_err)),'max_root_error_m':float(np.max(root_err)),
            'minimum_root_z_m':min_z,'mean_joint_rms_error_rad':float(np.mean(joint_err)),
            'torque_limit_fraction':saturations/(steps*len(controller)),'contact_audits':contact_audits}

def main():
    base=ROOT/'pyroki_grab_s3_windowed_20261030';early=ROOT/'pyroki_s3_overlap_anchor_20261030';tail=ROOT/'pyroki_s3_anchor_tail_20261030'
    docs=[json.loads((p/'report.json').read_text()) for p in (base,early,tail)]
    for p,d in zip((base,early,tail),docs):
        if sha(p/'windows.jsonl')!=d['rows_sha256']:raise ValueError('manifest changed')
    br=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    ar={x['window']:x for x in map(json.loads,(early/'windows.jsonl').read_text().splitlines())};tr={x['window']:x for x in map(json.loads,(tail/'windows.jsonl').read_text().splitlines())}
    if (len(br),len(ar),len(tr))!=(29,5,23):raise ValueError('missing windows')
    source=ROOT/'grab_s3_human_source_20261030/human_source.npz';sr=json.loads((source.parent/'report.json').read_text())
    if any(sha(source)!=d['source_sha256'] for d in docs):raise ValueError('source changed')
    anchored=[]
    for i,row in enumerate(br):
        first=load(base/row['artifact'],row['artifact_sha256'])
        second=first if i==0 else load((early if i<=5 else tail)/f'window_{i:02d}.npz',(ar if i<=5 else tr)[i]['artifact_sha256'])
        if not np.array_equal(first['frames'],second['frames']) or not np.array_equal(first['joint_names'],second['joint_names']):raise ValueError('window mismatch')
        anchored.append(second)
    frames,root,joints=chosen(anchored);names=anchored[0]['joint_names'].tolist()
    if not np.array_equal(frames,np.arange(264,382)):raise ValueError('frame mismatch')
    ref=resample_27(np.concatenate((root,joints),axis=1))
    if ref.shape!=(20,34):raise ValueError('wrong resampling result')
    refpath=Path(__file__).with_name('reference_20hz.npz')
    with refpath.open('xb') as f:np.savez_compressed(f,qpos=ref,joint_names=np.asarray(names),source_frames=np.arange(264,382,6))
    archive=ROOT/'data/GRAB/grab__s3.zip'
    if sha(archive)!=sr['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,sr['source_member'],'object')
    table=load_table(archive,sr['source_member']);hulls,paths,hbound=object_hulls(obj,table)
    project=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    scene=project/'Sim2Real/resources/robots/g1_description/scene_full.xml';gains=project/'Sim2Real/deploy/deploy_mujoco/configs/g1_full_body.yaml';cfg=yaml.safe_load(gains.read_text())
    results=[]
    for mode in ('objects_disabled','objects_enabled'):
        results.append(replay(scene,hulls,{'apple':obj,'table':table},frames,ref,names,cfg,mode))
    previous=json.loads((ROOT/'grab_s3_scene_pd_diagnostic_20261030/report.json').read_text())
    if sha(refpath)!=previous['reference_sha256']:raise ValueError('reference changed')
    for old,new in zip(previous['results'],results):
        if old['mode']!=new['mode'] or old['steps']!=new['steps']:raise ValueError('scenario mismatch')
        for key in ('mean_root_error_m','minimum_root_z_m','mean_joint_rms_error_rad','torque_limit_fraction'):
            if abs(old[key]-new[key])>1e-10:raise ValueError('replay mismatch: '+key)
        if old['contact_step_fraction']!=new['contact_step_fraction']:raise ValueError('contact replay mismatch')
    report={'source_sha256':sha(source),'archive_sha256':sha(archive),'mesh_sha256':{n:sha(p) for n,p in paths.items()},
            'scene_sha256':sha(scene),'gain_sha256':sha(gains),'script_sha256':sha(Path(__file__)),
            'reference_sha256':sha(refpath),'prior_report_sha256':sha(ROOT/'grab_s3_scene_pd_diagnostic_20261030/report.json'),'input_frames':[int(frames[0]),int(frames[-1])],
            'source_fps':120,'robot_fps':20,'apple_inner_hull_vertices':len(hulls['apple'].vertices),
            'apple_inner_hull_hausdorff_upper_bound_m':hbound,'results':results,
            'warning':'Single S3 clip, old MuJoCo scene, zero initial velocity and joint PD. Objects kinematically driven and nonresponsive. Independent distances check the same simulated convex geometry at contact stages; small lower bounds do not validate forces, nonconvex object geometry, dynamics, contact handling, grasp, Isaac Lab, training data or paper reproduction.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'scenario_count':len(results),'contact_audits':sum(len(x['contact_audits']) for x in results),'inconsistent':sum(y['inconsistent'] for x in results for y in x['contact_audits']),'nonbounded':sum(y['independent']['status']!='bounded' for x in results for y in x['contact_audits'])}),flush=True)
if __name__=='__main__':main()
