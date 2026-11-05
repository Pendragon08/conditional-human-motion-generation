"""Independent provenance, timing, and finite-result checks for scene replay."""
import json,sys,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_mujoco_object_20261030'))
from grab_s3_mujoco_object_distance import sha,chosen,load,read_grab_field
sys.path.insert(0,str(ROOT/'grab_s3_scene_timeline_v2_20261030'))
from grab_s3_scene_timeline import load_table

def main():
    directory=ROOT/'grab_s3_scene_pd_diagnostic_20261030';report=json.loads((directory/'report.json').read_text())
    if sha(directory/'grab_s3_scene_pd_diagnostic.py')!=report['script_sha256']:raise ValueError('script mismatch')
    reference=directory/'reference_20hz.npz'
    if sha(reference)!=report['reference_sha256']:raise ValueError('reference mismatch')
    with np.load(reference,allow_pickle=False) as z:
        q=z['qpos'];names=z['joint_names'].tolist();sampled=z['source_frames']
    if q.shape!=(20,34) or not np.isfinite(q).all() or not np.array_equal(sampled,np.arange(264,382,6)) or len(names)!=27:raise ValueError('bad reference')
    if not np.allclose(np.linalg.norm(q[:,:4],axis=1),1.,atol=1e-6,rtol=0):raise ValueError('reference quaternion invalid')
    base=ROOT/'pyroki_grab_s3_windowed_20261030';early=ROOT/'pyroki_s3_overlap_anchor_20261030';tail=ROOT/'pyroki_s3_anchor_tail_20261030'
    docs=[json.loads((p/'report.json').read_text()) for p in (base,early,tail)]
    for p,d in zip((base,early,tail),docs):
        if sha(p/'windows.jsonl')!=d['rows_sha256'] or d['source_sha256']!=report['source_sha256']:raise ValueError('manifest mismatch')
    br=[json.loads(x) for x in (base/'windows.jsonl').read_text().splitlines()]
    ar={x['window']:x for x in map(json.loads,(early/'windows.jsonl').read_text().splitlines())};tr={x['window']:x for x in map(json.loads,(tail/'windows.jsonl').read_text().splitlines())}
    windows=[]
    for i,row in enumerate(br):
        item=load(base/row['artifact'],row['artifact_sha256']) if i==0 else load((early if i<=5 else tail)/f'window_{i:02d}.npz',(ar if i<=5 else tr)[i]['artifact_sha256'])
        windows.append(item)
    frames,roots,joints=chosen(windows)
    if not np.array_equal(frames,np.arange(264,382)) or windows[0]['joint_names'].tolist()!=names:raise ValueError('source frame or joint mapping mismatch')
    original=np.asarray(np.r_[roots[0],joints[0]],dtype=np.float64)
    original[:4]/=np.linalg.norm(original[:4])
    first_diff=float(np.max(np.abs(original-q[0])))
    if first_diff>1e-10:raise ValueError('reference start does not match source')
    archive=ROOT/'data/GRAB/grab__s3.zip';meta=json.loads((ROOT/'grab_s3_human_source_20261030/report.json').read_text())
    if sha(archive)!=report['archive_sha256'] or sha(archive)!=meta['archive_sha256']:raise ValueError('archive mismatch')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,meta['source_member'],'object')
    table=load_table(archive,meta['source_member']);t=np.arange(len(frames))/120.;time_error={}
    for name,entry in (('apple',obj),('table',table)):
        ori=np.asarray(entry['params']['global_orient'])[frames];pos=np.asarray(entry['params']['transl'])[frames]
        rotation=Rotation.from_rotvec(ori).inv();slerp=Slerp(t,rotation)
        rot_err=float(np.max((slerp(t).inv()*rotation).magnitude()))
        pinterp=np.column_stack([np.interp(t,t,pos[:,k]) for k in range(3)])
        pos_err=float(np.max(np.abs(pinterp-pos)))
        if rot_err>1e-10 or pos_err>1e-10:raise ValueError('source interpolation mismatch')
        time_error[name]={'position_max_m':pos_err,'angle_max_rad':rot_err}
    if len(report['results'])!=2 or {x['mode'] for x in report['results']}!={'objects_disabled','objects_enabled'}:raise ValueError('missing scenarios')
    for x in report['results']:
        if x['steps']!=450 or abs(x['dt']-.002)>1e-12 or abs(x['duration_s']-.9)>1e-12:raise ValueError('time mismatch')
        if any(v<0 or v>1 for v in x['contact_step_fraction'].values()):raise ValueError('bad contact fraction')
        for k in ('mean_root_error_m','max_root_error_m','minimum_root_z_m','mean_joint_rms_error_rad','torque_limit_fraction'):
            if not np.isfinite(x[k]):raise ValueError('nonfinite result')
    result={'report_sha256':sha(directory/'report.json'),'reference_sha256':sha(reference),'script_sha256':sha(Path(__file__)),
            'initial_reference_max_difference':first_diff,'source_frames':len(frames),'reference_frames':len(q),
            'interpolation_source_time_error':time_error,'scenario_count':len(report['results']),
            'warning':'Checks provenance, time mapping, source-frame interpolation and finite statistics only. Does not validate physical model, object reactivity, contact accuracy, control, full trajectory or training readiness.'}
    with Path(__file__).with_name('audit_report.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
