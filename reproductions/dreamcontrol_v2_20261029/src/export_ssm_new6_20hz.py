"""Isolated 20 Hz/263D previews from six audited soft-anchor PyRoki runs.

No declared text is automatically accepted, and no preview is a training set.
Individual encoding failures remain visible and do not erase other outputs.
"""
import hashlib,json,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from blend_robot_windows import blend_windows
from caption_interval_alignment import associate_caption
from feature263 import canonicalize_z_up,encode
from g1_fk import URDFFK
from resample_robot_60_to_20 import resample_robot_60_to_20
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN=ROOT/'ssm_new6_anchor_gpu_v2_20261101';HUMAN=ROOT/'ssm_new6_human_20261101'
HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
IDS=('000162','004028','006749','007129','008894','009466')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(1)
    if (RUN/'queue_status.txt').read_text().strip()!='audited':raise ValueError('human GPU queue is not audited')
    batch=json.loads((RUN/'batch_audit.json').read_text()); audited={x['id']:x for x in batch['cases']}
    if set(audited)!=set(IDS):raise ValueError('incomplete audit coverage')
    limits={}
    for j in ET.parse(URDF).getroot().findall('joint'):
        if j.get('type')=='continuous':limits[j.get('name')]=(-np.inf,np.inf)
        elif j.get('type')=='revolute':
            x=j.find('limit');limits[j.get('name')]=(float(x.get('lower')),float(x.get('upper')))
    results=[]
    with (HERE/'records.jsonl').open('x',buffering=1) as log:
        for sid in IDS:
            try:
                source=HUMAN/sid;run=RUN/sid
                hr=json.loads((source/'human_report.json').read_text());ha=json.loads((source/'human_audit.json').read_text())
                rr=json.loads((run/'report.json').read_text());ra=json.loads((run/'audit_report.json').read_text())
                if (sha(source/'human_report.json')!=ha['human_report_sha256'] or
                    sha(source/'human_rawfps.npz')!=ha['artifact_sha256'] or ha['artifact_sha256']!=ra['source_sha256'] or
                    sha(run/'report.json')!=ra['report_sha256'] or ra['report_sha256']!=audited[sid]['report_sha256'] or
                    sha(run/'windows.jsonl')!=rr['rows_sha256'] or sha(URDF)!=rr['urdf_sha256']):
                    raise ValueError('input or audit changed')
                with np.load(source/'human_rawfps.npz',allow_pickle=False) as f:
                    raw_frames=f['source_frames'].copy();raw_times=f['source_times'].copy()
                rows=[json.loads(s) for s in (run/'windows.jsonl').read_text().splitlines()];windows=[]
                for row in rows:
                    p=run/row['artifact']
                    if sha(p)!=row['artifact_sha256']:raise ValueError('window hash changed')
                    with np.load(p,allow_pickle=False) as f:
                        windows.append({'frames':f['sample_indices'].copy(),'root_wxyz_xyz':f['root_wxyz_xyz'].copy(),
                            'joint_pos':f['joint_pos'].copy(),'joint_names':f['joint_names'].copy()})
                merged=blend_windows(windows);indices=merged['frames']
                np.testing.assert_array_equal(indices,np.arange(1,len(raw_frames)))
                sampled=resample_robot_60_to_20(merged['blended'],raw_times[indices]);q=sampled['states'];t=sampled['times'];names=list(merged['joint_names'])
                if set(names)!=set(limits):raise ValueError('joint names mismatch')
                low=np.array([limits[n][0] for n in names]);high=np.array([limits[n][1] for n in names])
                violation=float(np.maximum(np.maximum(low-q[:,7:],q[:,7:]-high),0.).max())
                points=URDFFK(URDF).forward(q[:,4:7],q[:,[1,2,3,0]],q[:,7:],names)
                features=encode(points,foot_velocity_squared_threshold=.002)
                with torch.no_grad():decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
                error=float(np.linalg.norm(decoded-canonicalize_z_up(points)[:-1],axis=-1).max())
                if features.shape!=(len(t)-1,263) or error>1e-4 or not np.isfinite(features).all():raise ValueError(f'encoding self-check failed: {error} m')
                captions=[]
                for caption in hr['original_caption_records']:
                    try:
                        assoc=associate_caption(clip_sample_times=raw_times[::hr['official_stride']],feature_sample_times=t,
                            start_tag=caption['tags'][0],end_tag=caption['tags'][1],timed_tag_convention='unknown')
                    except ValueError as e:assoc={'status':'invalid_or_unknown_tags','reason':str(e),'caption_accepted':False}
                    captions.append({'caption':caption['text'],'tags':caption['tags'],'association':assoc})
                out=HERE/(sid+'.npz')
                with out.open('xb') as f:
                    np.savez_compressed(f,features=features,root_wxyz_xyz=q[:,:7],joint_pos=q[:,7:],sample_times=t,
                        joint_names=np.asarray(names),points_z_up_world=points,raw_sample_indices=indices,
                        raw_source_frames=raw_frames[indices],raw_blended=merged['blended'])
                result={'id':sid,'status':'technical_preview','artifact':out.name,'artifact_sha256':sha(out),
                    'human_report_sha256':sha(source/'human_report.json'),'solver_report_sha256':sha(run/'report.json'),
                    'solver_audit_sha256':sha(run/'audit_report.json'),'source_group':hr['source_parent'],
                    'source_fps_declared':hr['reference_fps'],'sampled_points':len(t),'feature_rows':len(features),
                    'time_range_s':[float(t[0]),float(t[-1])],'tail_after_last_target_s':sampled['discarded_tail_s'],
                    'max_joint_limit_excess_rad':violation,'decode_max_error_m':error,
                    'overlap_max':{k:ra['max_overlap_'+k] for k in ('root_m','rotation_rad','joint_rad')},
                    'captions':captions,'caption_accepted':False,'training_ready':False}
            except Exception as e:
                result={'id':sid,'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            results.append(result);log.write(json.dumps(result,allow_nan=False)+'\n');log.flush();print(sid,result['status'],flush=True)
    report={'script_sha256':sha(Path(__file__)),'source_batch_audit_sha256':sha(RUN/'batch_audit.json'),
        'records_sha256':sha(HERE/'records.jsonl'),'cases':len(results),
        'successful':sum(x['status']=='technical_preview' for x in results),
        'failed':[x['id'] for x in results if x['status']=='failed'],'training_ready':False,
        'limitations':['Approximate soft-overlap, offline FIR and encoder; not author-v2.',
        'Declared time and unreviewed captions are not time or semantic acceptance.',
        'No source independence, collision, contact or dynamics acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
