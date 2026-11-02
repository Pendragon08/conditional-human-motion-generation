"""Rebuild available licensed SMPL-X motions using GMR, paper 29DoF/69D.

Self-implemented orchestration, time alignment, contact extraction, mirroring
and paper Algorithms 1/2. GMR IK and SMPL-X remain their existing source.
This does NOT stand in for the paper's 40k-iteration dynamic tracking filter.
Outputs and labels are licensed/private; only aggregate receipts are public.
"""
import argparse,collections,contextlib,hashlib,importlib.metadata,io,json,math,pathlib,re,shutil,sys,time,types
import numpy as np,torch,smplx,mujoco
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation,Slerp

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def load_runtime(spec):
    actual={k:importlib.metadata.version(k) for k in spec['required_versions']}
    assert actual==spec['required_versions'],('Package drift',actual)
    sys.path.insert(0,spec['gmr_root'])
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
    return GeneralMotionRetargeting,get_smplx_data_offline_fast,actual

def human_forward(path,spec,models,max_frames=None):
    with np.load(path,allow_pickle=True) as f:raw={k:f[k] for k in f.files}
    n=len(raw['pose_body']);n=min(n,max_frames) if max_frames else n
    for k in ['pose_body','root_orient','trans']:raw[k]=raw[k][:n]
    gender=str(raw['gender']);betas=np.asarray(raw['betas']).reshape(-1);key=(gender,len(betas))
    if key not in models:models[key]=smplx.create(spec['body_models'],'smplx',gender=gender,use_pca=False,num_betas=len(betas))
    model=models[key];joints=[];pose=[];orient=[]
    with torch.no_grad():
        for a in range(0,n,64):
            b=min(a+64,n);z=lambda d:torch.zeros(b-a,d)
            out=model(betas=torch.from_numpy(betas.astype(np.float32))[None].expand(b-a,-1),expression=z(10),global_orient=torch.as_tensor(raw['root_orient'][a:b]).float(),body_pose=torch.as_tensor(raw['pose_body'][a:b]).float(),transl=torch.as_tensor(raw['trans'][a:b]).float(),left_hand_pose=z(45),right_hand_pose=z(45),jaw_pose=z(3),leye_pose=z(3),reye_pose=z(3),return_full_pose=True)
            joints.append(out.joints.detach());pose.append(out.full_pose.detach());orient.append(out.global_orient.detach())
    body=types.SimpleNamespace(joints=torch.cat(joints),full_pose=torch.cat(pose),global_orient=torch.cat(orient))
    height=float(1.66+.1*betas[0]);return raw,model,body,height

def fk(model,qpos):
    data=mujoco.MjData(model);out=[]
    for q in qpos:data.qpos[:]=q;mujoco.mj_forward(model,data);out.append(data.xpos.copy())
    return np.stack(out)

def mirror_map(model):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);S=np.diag([1,-1,1]);names=[];ids=[]
    for j in range(model.njnt):
        if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE:names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j));ids.append(j)
    assert len(names)==29
    swap=lambda s:s.replace('left_','TMP_').replace('right_','left_').replace('TMP_','right_')
    order=[];sign=[]
    for dst,j in zip(names,ids):
        src=names.index(swap(dst));axis_ref=-S@data.xaxis[ids[src]]
        similarity=float(axis_ref@data.xaxis[j]);assert abs(abs(similarity)-1)<1e-5
        order.append(src);sign.append(1 if similarity>0 else -1)
    # Validate the represented skeleton (root + actuated joint bodies). Fixed
    # IMU mount frames have a real 2.24mm lateral CAD offset and are not motion joints.
    bodies={i:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,swap(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i))) for i in range(1,model.nbody) if model.body_jntnum[i]>0}
    assert all(x>0 for x in bodies.values())
    return np.array(order),np.array(sign),bodies,names

def mirror_q(q,order,sign):
    out=q.copy();out[:,1]*=-1;out[:,[4,6]]*=-1
    out[:,7:]=q[:,7:][:,order]*sign
    return out

def contact(model,q):
    pos=fk(model,q);ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
    ankles=pos[:,ids];delta2=np.sum(np.diff(ankles,axis=0)**2,axis=-1)
    c=((delta2<.002)&(ankles[:-1,:,2]<.2)).astype(np.float32)
    return np.concatenate([c,c[-1:]],axis=0),pos

def encode(q,c):
    e=Rotation.from_quat(q[:,3:7][:,[1,2,3,0]]).as_euler('xyz');roll,pitch,yaw=e.T
    dy=(np.diff(yaw)+np.pi)%(2*np.pi)-np.pi;dp=np.diff(q[:,:3],axis=0)
    co,si=np.cos(yaw[:-1]),np.sin(yaw[:-1]);local=np.stack([co*dp[:,0]+si*dp[:,1],-si*dp[:,0]+co*dp[:,1],dp[:,2]],axis=-1)
    phi=np.stack([np.sin(roll[:-1]),np.cos(roll[:-1])-1,np.sin(pitch[:-1]),np.cos(pitch[:-1])-1],axis=-1)
    f=np.concatenate([phi,dy[:,None],c[:-1],local,q[:-1,2:3],q[:-1,7:],np.diff(q[:,7:],axis=0)],axis=-1).astype(np.float32);assert f.shape[1]==69
    # Independent inverse of the paper feature sequence, checked against raw trajectory.
    rp=np.arctan2(f[:,0],f[:,1]+1);pp=np.arctan2(f[:,2],f[:,3]+1)
    yy=yaw[0]+np.r_[0,np.cumsum(f[:-1,4].astype(np.float64))]
    local=f[:,7:10];cc,ss=np.cos(yy),np.sin(yy)
    world=np.stack([cc*local[:,0]-ss*local[:,1],ss*local[:,0]+cc*local[:,1],local[:,2]],axis=-1)
    p=q[0,:3]+np.vstack([np.zeros(3),np.cumsum(world[:-1],axis=0)]);p[:,2]=f[:,10]
    recR=Rotation.from_euler('xyz',np.stack([rp,pp,yy],axis=-1)).as_matrix();gtR=Rotation.from_quat(q[:-1,3:7][:,[1,2,3,0]]).as_matrix()
    err={'translation_m':float(np.max(np.abs(p-q[:-1,:3]))),'rotation_matrix':float(np.max(np.abs(recR-gtR))),'joint_rad':float(np.max(np.abs(f[:,11:40]-q[:-1,7:])))}
    assert max(err.values())<1e-3,err
    return f,err

def mirror_text(text):
    table={'left':'right','right':'left','clockwise':'counterclockwise','counterclockwise':'clockwise','counter-clockwise':'clockwise','counter clockwise':'clockwise','anticlockwise':'clockwise'}
    return re.sub(r'\b(counter[- ]?clockwise|anticlockwise|clockwise|left|right)\b',lambda m:table[m[0].lower()],text,flags=re.I)

def annotations(entry,times,mirror=False):
    good=[];rejected=[]
    for a,b,text,cats in entry['frame_ann']:
        if not(np.isfinite(a) and np.isfinite(b) and b>a and a>=0):rejected.append({'start':a,'end':b,'reason':'invalid_time'});continue
        start=max(0,float(a));end=min(float(b),float(times[-1]+.02))
        lo=max(0,int(np.ceil(start*50-1e-8)));hi=min(len(times),int(np.ceil(end*50-1e-8)))
        if hi<=lo:rejected.append({'start':a,'end':b,'reason':'no_sample_overlap'});continue
        good.append({'start_frame':lo,'end_frame_exclusive':hi,'start_s':lo/50,'end_s':hi/50,'text':mirror_text(text) if mirror else text,'action_categories':[mirror_text(c) for c in cats] if mirror else cats,'original_start_s':float(a),'original_end_s':float(b)})
    return good,rejected

def main(args):
    spec=json.loads(args.spec.read_text());args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    GMR,frames_fn,versions=load_runtime(spec);models={}
    pairs=json.loads(args.pairs.read_text());pairs=sorted(pairs,key=lambda x:(x['frames'],x['canonical_source']))
    if args.limit:pairs=pairs[:args.limit]
    receipt={'spec_sha256':sha(args.spec),'versions':versions,'target_dof':29,'feature_dim':69,'output_fps':50,'completed_sequences':0,'mirrored_sequences':0,'failed_sequences':0,'total_requested':len(pairs),'physical_filter_passed':False,'ready_for_generator_training':False,'status':'RUNNING','cpu_only':True,'implementation':'own paper-oriented preparation, GMR IK reused; not authors private final pipeline'}
    if args.witness:
        entry=pairs[0];raw,model,body,height=human_forward(entry['raw_stageii_candidates'][0],spec,models,max_frames=8)
        frames,_=frames_fn(raw,model,body,tgt_fps=float(raw['mocap_frame_rate']))
        with contextlib.redirect_stdout(io.StringIO()):retargeter=GMR('smplx','unitree_g1',actual_human_height=height,verbose=False);q=np.array(retargeter.retarget(frames[0]))
        assert q.shape==(36,) and np.isfinite(q).all()
        receipt.update(status='TEXTOP_GMR_BODY_WITNESS_PASSED',witness_qpos_shape=list(q.shape),witness_num_betas=int(model.num_betas),witness_frames_forwarded=8,source_motion_sha256=sha(entry['raw_stageii_candidates'][0]))
        receipt['source_hashes']={name:sha(pathlib.Path(spec['gmr_root'])/name) for name in ['general_motion_retargeting/motion_retarget.py','general_motion_retargeting/utils/smpl.py','general_motion_retargeting/ik_configs/smplx_to_g1.json','assets/unitree_g1/g1_mocap_29dof.xml']}
        gender=str(raw['gender']).upper();receipt['body_model_sha256']=sha(pathlib.Path(spec['body_models'])/'smplx'/('SMPLX_'+gender+'.npz'))
        (args.output/'witness.json').write_text(json.dumps(receipt,indent=2));print('TEXTOP_GMR_BODY_WITNESS_PASSED',json.dumps(receipt));return
    records=[]
    for index,entry in enumerate(pairs):
        try:
            if shutil.disk_usage(args.output).free<768*1024**2:raise RuntimeError('Disk reserve below768MiB; preserving all input data')
            source=entry['raw_stageii_candidates'][0];raw,model,body,height=human_forward(source,spec,models)
            frames,aligned=frames_fn(raw,model,body,tgt_fps=30)
            with contextlib.redirect_stdout(io.StringIO()):
                retargeter=GMR('smplx','unitree_g1',actual_human_height=height,verbose=False)
                q=np.stack([retargeter.retarget(frame).copy() for frame in frames])
            assert q.shape[1]==36 and np.isfinite(q).all()
            pos=fk(retargeter.model,q);q[:,2]-=float(pos[:,1:,2].min());q[:,:2]-=q[0,:2]
            duration=(len(raw['pose_body'])-1)/float(raw['mocap_frame_rate']);t30=np.linspace(0,duration,len(q));t50=np.arange(int(np.floor(duration*50))+1)/50
            pq=interp1d(t30,q[:,:3],axis=0)(t50);j=interp1d(t30,q[:,7:],axis=0)(t50)
            rot=Slerp(t30,Rotation.from_quat(q[:,3:7][:,[1,2,3,0]]))(t50).as_quat()[:,[3,0,1,2]]
            q50=np.concatenate([pq,rot,j],axis=-1)
            c,bodypos=contact(retargeter.model,q50);feature,err=encode(q50,c)
            order,sign,bodies,joint_names=mirror_map(retargeter.model);mq=mirror_q(q50,order,sign);mc,mpos=contact(retargeter.model,mq);mf,merr=encode(mq,mc)
            reflect=np.array([1,-1,1]);mirror_error=max(float(np.abs(mpos[:,dest]-bodypos[:,src]*reflect).max()) for src,dest in bodies.items())
            assert mirror_error<1e-3,('mirror_FK_error_m',mirror_error)
            ident=hashlib.sha256(entry['canonical_source'].encode()).hexdigest()[:16]
            file=args.output/(ident+'.private.npz')
            np.savez_compressed(file,qpos_29=q50,contact=c,features_69=feature,mirrored_qpos_29=mq,mirrored_contact=mc,mirrored_features_69=mf,times_s=t50)
            ann,invalid=annotations(entry,t50);mann,_=annotations(entry,t50,True)
            item={'canonical_source':entry['canonical_source'],'split':entry['split'],'raw_path':source,'raw_sha256':sha(source),'output_file':str(file),'output_sha256':sha(file),'raw_fps':float(raw['mocap_frame_rate']),'gmr_aligned_fps':float(aligned),'frames50':len(q50),'dof':29,'features':69,'annotations':ann,'mirrored_annotations':mann,'rejected_annotations':invalid,'roundtrip':err,'mirror_roundtrip':merr,'mirror_fk_error_m':mirror_error,'joint_names':joint_names,'physical_filter_passed':False}
            records.append(item);receipt['completed_sequences']+=1;receipt['mirrored_sequences']+=1
            with (args.output/'prepared_manifest.private.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        except Exception as e:
            receipt['failed_sequences']+=1
            with (args.output/'failures.private.jsonl').open('a') as f:f.write(json.dumps({'index':index,'source':entry['canonical_source'],'error':repr(e)})+'\n')
        receipt['processed']=index+1;receipt['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        (args.output/'progress.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
    receipt['status']='PREPROCESSING_FINISHED_FILTER_PENDING' if records else 'PREPROCESSING_FAILED';receipt['splits']=dict(collections.Counter(x['split'] for x in records));receipt['paired_segments']=sum(len(x['annotations'])+len(x['mirrored_annotations']) for x in records)
    (args.output/'progress.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=pathlib.Path,required=True);p.add_argument('--pairs',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--witness',action='store_true');main(p.parse_args())
