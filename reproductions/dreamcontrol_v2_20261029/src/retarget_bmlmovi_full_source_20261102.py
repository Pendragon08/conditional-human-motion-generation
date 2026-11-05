"""GPU window retarget of twenty full-rate BMLmovi source previews.

Uses prior experimental v1 solver and assumed ground. Output is isolated,
unstitched, and cannot be used as training data without further acceptance.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import jax,jax.numpy as jnp,jaxlie,numpy as np,pyroki as pk,yourdfpy
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
ANCHOR=ROOT/'pyroki_s3_anchor_tail_20261030';VARIANT=ROOT/'pyroki_grab_temporal_20261030'
sys.path.insert(0,str(ANCHOR))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,create_conn_tree,solve_retargeting,foot_detect

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def guard():
    v=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    utilization,free=map(int,v)
    if utilization>30 or free<2048 or shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:
        raise RuntimeError(f'resource guard stopped: utilization={utilization}%, free={free}MiB')
def anchor(previous,indices):
    root=np.zeros((7,7),np.float32);root[:,0]=1.;q=np.zeros((7,27),np.float32);active=np.zeros(7,np.float32)
    if previous is not None:
        common=np.intersect1d(previous['indices'],indices);old=np.searchsorted(previous['indices'],common);new=np.searchsorted(indices,common)
        if len(common) not in (3,4,5,6) or not np.array_equal(new,np.arange(len(common))):raise ValueError('bad overlap')
        root[new]=previous['root'][old];q[new]=previous['q'][old];active[new]=1.
    return root,q,active

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--samples',nargs='+',required=True);a=ap.parse_args()
    if not a.samples or any(len(x)!=6 or not x.isdigit() for x in a.samples) or len(a.samples)!=len(set(a.samples)):raise ValueError('unexpected samples')
    guard()
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('one explicit GPU required')
    helper=ANCHOR/'retarget_helpers/_utils.py';receipt=json.loads((VARIANT/'report.json').read_text());anchor_manifest=json.loads((ANCHOR/'manifest.json').read_text())
    baseline=VARIANT/'retarget_helpers/_utils.py';urdf_path=PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if sha(helper)!=anchor_manifest['anchor_helper_sha256'] or sha(baseline)!=receipt['helper_sha256'] or sha(urdf_path)!=receipt['urdf_sha256']:raise ValueError('solver inputs changed')
    batch=ROOT/'bmlmovi_full_human_20261102';pilot_audit=batch/'audit.json'
    batch_report=json.loads((batch/'report.json').read_text());audit_report=json.loads(pilot_audit.read_text());source_rows=[json.loads(x) for x in (batch/'records.jsonl').read_text().splitlines()]
    if sha(batch/'records.jsonl')!=batch_report['records_sha256'] or sha(batch/'report.json')!=audit_report['source_report_sha256'] or len(source_rows)!=20:raise ValueError('source audit binding mismatch')
    audit_index={x['id']:x for x in audit_report['cases']};source_index={x['id']:x for x in source_rows}
    if len(audit_index)!=20 or len(source_index)!=20 or not set(a.samples).issubset(source_index):raise ValueError('source audit coverage mismatch')
    a.out.mkdir(exist_ok=False)
    robot=pk.Robot.from_urdf(yourdfpy.URDF.load(str(urdf_path),load_meshes=False,load_collision_meshes=True))
    collision=pk.collision.RobotCollision.from_urdf(yourdfpy.URDF.load(str(urdf_path),load_meshes=False,load_collision_meshes=True))
    hi,ri=get_humanoid_retarget_indices_g1_27dof();mask=create_conn_tree(robot,ri)
    plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),height_data=jnp.zeros((100,100),dtype=jnp.float32))
    names=list(robot.joints.actuated_names);weights=dict(receipt['weights']);weights['root_smoothness']=weights['foot_skating']=2.0
    with (a.out/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as progress:
      for sample in a.samples:
        record=source_index[sample];source=batch/record['artifact'];audit=audit_index.get(sample)
        if record['status']!='reconstructed_unreviewed' or audit is None or audit['status']!='technical_sample_checked' or sha(source)!=record['artifact_sha256'] or sha(source)!=audit['artifact_sha256']:raise ValueError('source/audit binding mismatch')
        with np.load(source,allow_pickle=False) as f:human=f['human_joints'].astype(np.float32);source_frames=f['source_frames'].copy();fps=float(f['declared_fps'])
        times=source_frames.astype(np.float64)/fps
        if fps!=120. or fps!=record['declared_fps'] or human.shape!=(len(times),22,3) or len(times)<8 or not np.isfinite(human).all() or not np.array_equal(source_frames,np.arange(len(human))):raise ValueError('bad declared-rate source timeline')
        starts=list(range(0,len(human)-7,4))
        if starts[-1]!=len(human)-8:starts.append(len(human)-8)
        out=a.out/sample;out.mkdir(exist_ok=False);previous=None;rows=[];began=time.monotonic()
        with (out/'windows.jsonl').open('x',encoding='utf-8',buffering=1) as log:
          for wi,start in enumerate(starts):
            if wi and wi%5==0:time.sleep(1.2);guard()
            w=human[start:start+8];indices=np.arange(start+1,start+8);left,right=foot_detect(w[None],thres=(.05/fps)**2);target=jnp.asarray(w[1:]);ar,aq,am=anchor(previous,indices)
            root,joints=solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,is_left_foot_contact=jnp.asarray(left[0]>0),is_right_foot_contact=jnp.asarray(right[0]>0),left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,weights=weights,pick_point=jnp.zeros(3),anchor_roots=jnp.asarray(ar),anchor_joints=jnp.asarray(aq),anchor_mask=jnp.asarray(am),anchor_weight=10.)
            q,rq=np.asarray(joints),np.asarray(root.wxyz_xyz)
            if q.shape!=(7,27) or rq.shape!=(7,7) or not np.isfinite(q).all() or not np.isfinite(rq).all() or not np.allclose(np.linalg.norm(rq[:,:4],axis=1),1.,atol=1e-4,rtol=0):raise ValueError('invalid solver output')
            overlap=None
            if previous is not None:
                common=np.intersect1d(previous['indices'],indices);old=np.searchsorted(previous['indices'],common);new=np.searchsorted(indices,common)
                x=Rotation.from_quat(previous['root'][old][:,[1,2,3,0]]);y=Rotation.from_quat(rq[new][:,[1,2,3,0]])
                overlap={'root_m':float(np.linalg.norm(previous['root'][old,4:]-rq[new,4:],axis=1).max()),'rotation_rad':float((x.inv()*y).magnitude().max()),'joint_rad':float(np.abs(previous['q'][old]-q[new]).max())}
            target_file=out/f'window_{wi:03d}.npz'
            with target_file.open('xb') as f:np.savez_compressed(f,sample_indices=indices,source_frames=source_frames[indices],source_times=times[indices],root_wxyz_xyz=rq,joint_pos=q,joint_names=np.asarray(names))
            row={'window':wi,'artifact':target_file.name,'sha256':sha(target_file),'sample_indices':indices.tolist(),'overlap':overlap}
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno());previous={'indices':indices,'root':rq,'q':q};rows.append(row)
            print('WINDOW',sample,wi+1,'of',len(starts),flush=True)
        if sha(source)!=audit['artifact_sha256']:raise RuntimeError('human source changed')
        report={'id':sample,'human_sha256':sha(source),'human_report_sha256':sha(batch/'report.json'),'human_audit_sha256':sha(pilot_audit),'script_sha256':sha(Path(__file__)),'helper_sha256':sha(helper),'urdf_sha256':sha(urdf_path),'rows_sha256':sha(out/'windows.jsonl'),'windows':len(rows),'elapsed_s':time.monotonic()-began,'source_declared_fps':fps,'retarget_sample_rate':'full_declared_source_rate','training_ready':False,'limitations':['Unstitched windows, assumed z=0, low-speed contact proxy and old solver are approximations.','Source FPS is declared 120 Hz, not independently measured; solver weights inherited from old 20 Hz experiments.','Source identity, true time, text, collision and dynamics not accepted.','No downsampling yet; source captions remain unaccepted.']}
        with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
        progress.write(json.dumps({'id':sample,'report_sha256':sha(out/'report.json'),'windows':len(rows)})+'\n');print('COMPLETE',sample,flush=True)
if __name__=='__main__':main()
