"""Compare all existing 258 real SDK-exported poses, CPU only; no new simulation."""
import argparse,hashlib,json,os,pathlib,time,traceback
import numpy as np
from tracker_analytic_fk import TrackerFK,reference_arrays
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
sp=ROOT/'configs/tracker-analytic-fk-spec.json';s=json.loads(sp.read_text());assert a.run.isidentifier()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={0,1,2,3}
for n,h in s['bindings'].items():assert sha(ROOT/n)==h,n
out=ROOT/'analytic_fk_validation'/a.run;out.mkdir(parents=True,exist_ok=False)
r={'status':'STARTING','spec_sha256':sha(sp),'gpu_calls':0,'new_sdk_calls':0,'optimizer_updates':0,'physics_steps':0}
try:
    er=json.loads((ROOT/s['export_receipt']).read_text());ge=json.loads((ROOT/s['generation_receipt']).read_text())
    meta=json.loads((pathlib.Path(ge['artifact']['file']).parent/'generation.private.json').read_text())
    urdf=s['urdf'];assert sha(urdf)==s['urdf_sha256']==er['urdf_sha256']
    assert sha(ge['artifact']['file'])==ge['artifact']['sha256'] and sha(er['reference']['file'])==er['reference']['sha256']
    with np.load(ge['artifact']['file'],allow_pickle=False) as z:q=z['qpos_29'].copy()
    with np.load(er['reference']['file'],allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
    fk=TrackerFK(urdf,er['joint_names'],er['body_names']);t=time.perf_counter();candidate=reference_arrays(fk,q,meta['joint_names']);elapsed=time.perf_counter()-t
    pos=float(np.linalg.norm(candidate['body_pos_w'].astype(float)-ref['body_pos_w'],axis=-1).max())
    qa=candidate['body_quat_w'].astype(float);qb=ref['body_quat_w'].astype(float)
    qa/=np.linalg.norm(qa,axis=-1,keepdims=True);qb/=np.linalg.norm(qb,axis=-1,keepdims=True)
    dq=np.minimum(np.linalg.norm(qa-qb,axis=-1),np.linalg.norm(qa+qb,axis=-1));ang=float((4*np.arcsin(np.clip(dq/2,0,1))).max())
    errs={k:float(np.abs(candidate[k].astype(float)-ref[k]).max()) for k in ['joint_pos','joint_vel','body_lin_vel_w','body_ang_vel_w']}
    # The exported world angular velocity uses q_next * inverse(q_prev), as does this implementation.
    r.update(frames=len(q),body_count=len(er['body_names']),joint_count=len(er['joint_names']),max_position_error_m=pos,max_rotation_error_rad=ang,
        other_max_abs_errors=errs,all_frames_cpu_seconds=elapsed,input_sha256=ge['artifact']['sha256'],reference_sha256=er['reference']['sha256'])
    assert len(q)==258 and pos<=s['max_position_error_m'] and ang<=s['max_rotation_error_rad'],r
    assert errs['joint_pos']==0 and errs['joint_vel']==0,r
    assert errs['body_lin_vel_w']<=s['max_linear_velocity_error_m_s'] and errs['body_ang_vel_w']<=s['max_angular_velocity_error_rad_s'],r
    r.update(status='TRACKER_URDF_ANALYTIC_FK_MATCHED_EXISTING_SDK_EXPORT',passed=True,online_execution_verified=False)
except BaseException as e:
    r.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
finally:
    (out/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r),flush=True)
