"""Independent convex distance brackets on selected hand-object poses.

Uses MuJoCo only for forward kinematics and geometry definitions. The distance
solver is independent. These are unsigned convex-set distances, not contact force.
"""
import json,sys,zipfile
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_mujoco_object_20261030'))
from grab_s3_mujoco_object_distance import sha,chosen,load,read_grab_field
sys.path.insert(0,str(ROOT/'grab_s3_scene_timeline_v2_20261030'))
from grab_s3_scene_timeline import mesh_for

def maximum_support(model,data,g,x):
    typ=int(model.geom_type[g]);p=data.geom_xpos[g];r=data.geom_xmat[g].reshape(3,3);s=model.geom_size[g]
    norm=np.linalg.norm(x)
    if norm<1e-15:return p.copy()
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        m=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[m]);n=int(model.mesh_vertnum[m]);v=model.mesh_vert[a:a+n]
        return p+r@v[int(np.argmax(v@(r.T@x)))]
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):return p+s[0]*x/norm
    if typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):return p+s[1]*np.sign(x@r[:,2])*r[:,2]+s[0]*x/norm
    if typ==int(mujoco.mjtGeom.mjGEOM_BOX):return p+r@(s*np.sign(r.T@x))
    if typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        axis=r[:,2];proj=x-axis*(x@axis);plen=np.linalg.norm(proj)
        return p+s[1]*np.sign(x@axis)*axis+(s[0]*proj/plen if plen>1e-15 else 0.)
    raise ValueError('unsupported geom type '+str(typ))

def bracket(a,model,data,g,tol=2e-5,max_iter=120):
    x=a[0]-data.geom_xpos[g];points=[x.copy()];weights=np.array([1.]);keys=set();last=None
    for i in range(max_iter):
        upper=float(np.linalg.norm(x))
        if upper<tol:return {'lower_m':0.,'upper_m':upper,'gap_m':upper,'status':'bounded','iterations':i}
        ia=int(np.argmin(a@x));w=a[ia]-maximum_support(model,data,g,x)
        lower=max(0.,float(x@w/upper));gap=max(0.,upper-lower)
        if lower>upper+1e-9:raise ValueError('invalid bounds')
        last={'lower_m':lower,'upper_m':upper,'gap_m':gap,'status':'bounded' if gap<=tol else 'running','iterations':i}
        if gap<=tol:return last
        key=tuple(np.round(w,12))
        if key in keys:last['status']='repeated_support';return last
        keys.add(key);points.append(w);weights=np.r_[weights,0.];mat=np.asarray(points);gram=mat@mat.T
        res=minimize(lambda q:float(q@gram@q),weights,jac=lambda q:2*gram@q,method='SLSQP',bounds=[(0.,1.)]*len(weights),
                     constraints={'type':'eq','fun':lambda q:float(q.sum()-1.),'jac':lambda q:np.ones_like(q)},
                     options={'ftol':1e-14,'maxiter':150,'disp':False})
        q=np.clip(res.x,0.,1.);q/=q.sum();keep=q>1e-10;q=q[keep];q/=q.sum();mat=mat[keep];points=list(mat);weights=q;x=q@mat
        if not np.isfinite(x).all():raise ValueError('nonfinite solve')
    last['status']='iteration_limit';return last

def main():
    # Confirm analytic sphere support and numerical brackets on an exact case.
    simple=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="s" type="sphere" size=".1" pos=".7 0 0"/></worldbody></mujoco>')
    sd=mujoco.MjData(simple);mujoco.mj_forward(simple,sd)
    box=np.array([[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.1,.1)])
    check=bracket(box,simple,sd,0)
    if not (check['lower_m']-1e-8<=.5<=check['upper_m']+1e-8 and check['gap_m']<=2e-5):raise ValueError('synthetic box sphere failed')
    base=ROOT/'pyroki_grab_s3_windowed_20261030';early=ROOT/'pyroki_s3_overlap_anchor_20261030';tail=ROOT/'pyroki_s3_anchor_tail_20261030'
    docs=[json.loads((p/'report.json').read_text()) for p in (base,early,tail)]
    for p,d in zip((base,early,tail),docs):
        if sha(p/'windows.jsonl')!=d['rows_sha256']:raise ValueError('manifest changed')
    br=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    ar={x['window']:x for x in map(json.loads,(early/'windows.jsonl').read_text().splitlines())}
    tr={x['window']:x for x in map(json.loads,(tail/'windows.jsonl').read_text().splitlines())}
    if (len(br),len(ar),len(tr))!=(29,5,23):raise ValueError('missing windows')
    src=ROOT/'grab_s3_human_source_20261030/human_source.npz';sr=json.loads((src.parent/'report.json').read_text())
    if any(sha(src)!=d['source_sha256'] for d in docs):raise ValueError('source changed')
    options=[[],[]]
    for i,row in enumerate(br):
        first=load(base/row['artifact'],row['artifact_sha256'])
        second=first if i==0 else load((early if i<=5 else tail)/f'window_{i:02d}.npz',(ar if i<=5 else tr)[i]['artifact_sha256'])
        if not np.array_equal(first['frames'],second['frames']) or not np.array_equal(first['joint_names'],second['joint_names']):raise ValueError('window mismatch')
        options[0].append(first);options[1].append(second)
    frames,r0,q0=chosen(options[0]);frames2,r1,q1=chosen(options[1]);names=options[0][0]['joint_names'].tolist()
    if not np.array_equal(frames,frames2) or not np.array_equal(frames,np.arange(264,382)):raise ValueError('time mismatch')
    archive=ROOT/'data/GRAB/grab__s3.zip'
    if sha(archive)!=sr['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,sr['source_member'],'object')
    meshfile,orig,hull=mesh_for(obj['object_mesh'])
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    def mid(kind,name):
        value=mujoco.mj_name2id(model,kind,name)
        if value<0:raise ValueError('missing '+name)
        return value
    rid=mid(mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');raddr=int(model.jnt_qposadr[rid]);jointids=[mid(mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    hand=[]
    for g in range(model.ngeom):
        bn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
        if bn.startswith(('right_hand_','right_wrist_')) and (model.geom_contype[g]!=0 or model.geom_conaffinity[g]!=0):hand.append(g)
    if len(hand)!=11:raise ValueError('unexpected hand geometry count')
    close={'right_hand_index_0_joint':np.pi/2,'right_hand_index_1_joint':np.pi/2,'right_hand_middle_0_joint':np.pi/2,'right_hand_middle_1_joint':np.pi/2,'right_hand_thumb_0_joint':0.,'right_hand_thumb_1_joint':-np.pi/3,'right_hand_thumb_2_joint':-np.pi/2}
    old={}
    for posture,dirname in (('open','grab_s3_mujoco_object_20261030'),('closed','grab_s3_mujoco_object_closed_20261030')):
        for row in map(json.loads,(ROOT/dirname/'frames.jsonl').read_text().splitlines()):old[(row['frame'],row['method'],posture)]=row['right_min_signed_distance_m']
    if sys.argv[1:] not in ([], ['--all']):raise ValueError('expected no argument or --all')
    selected=list(range(264,382)) if sys.argv[1:] else [264,287,302,310,330,380,381]
    rows=[]
    for frame in selected:
        i=int(frame-264); orient=np.asarray(obj['params']['global_orient'])[frame];trans=np.asarray(obj['params']['transl'])[frame]
        av=Rotation.from_rotvec(orient).inv().apply(hull.vertices)+trans
        for label,root,q in (('baseline',r0,q0),('anchored',r1,q1)):
            for posture in ('open','closed'):
                mujoco.mj_resetData(model,data);data.qpos[raddr:raddr+3]=root[i,4:];data.qpos[raddr+3:raddr+7]=root[i,:4]
                for j,val in zip(jointids,q[i]):data.qpos[model.jnt_qposadr[j]]=val
                if posture=='closed':
                    for name,val in close.items():data.qpos[model.jnt_qposadr[mid(mujoco.mjtObj.mjOBJ_JOINT,name)]]=val
                data.qvel[:]=0;mujoco.mj_forward(model,data)
                results=[bracket(av,model,data,g) for g in hand]
                lower=min(x['lower_m'] for x in results);upper=min(x['upper_m'] for x in results)
                if lower>upper+1e-8:raise ValueError('invalid hand bounds')
                prior=float(old[(frame,label,posture)])
                rows.append({'frame':frame,'method':label,'posture':posture,'lower_m':lower,'upper_m':upper,
                             'mujoco_prior_m':prior,'prior_outside_bracket':bool(prior<lower-2e-5 or prior>upper+2e-5),
                             'nonbounded_pairs':sum(x['status']!='bounded' for x in results),
                             'max_pair_gap_m':max(x['gap_m'] for x in results)})
        print(json.dumps({'completed_frame':frame,'rows':len(rows)}),flush=True)
    report={'archive_sha256':sha(archive),'source_sha256':sha(src),'mesh_sha256':sha(meshfile),'scene_sha256':sha(scene),
            'script_sha256':sha(Path(__file__)),'selected_frames':selected,'sphere_box_exact_m':.5,'sphere_box_check':check,
            'results':rows,'warning':'Independent unsigned convex-set distance brackets on source frames at 120Hz. MuJoCo provides robot forward kinematics, not distance calculations. Apple is a convex hull; no penetration depth, contact force, control, dynamics, grasp or dataset acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'rows':len(rows),'outside':sum(x['prior_outside_bracket'] for x in rows),
                      'nonbounded_pairs':sum(x['nonbounded_pairs'] for x in rows),'max_pair_gap_m':max(x['max_pair_gap_m'] for x in rows)}),flush=True)
if __name__=='__main__':main()
