"""Create unaccepted 20 Hz/263D candidates from audited anchored SSM windows."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from blend_robot_windows import blend_windows
from caption_interval_alignment import associate_caption
from feature263 import canonicalize_z_up, encode
from g1_fk import URDFFK
from resample_robot_60_to_20 import resample_robot_60_to_20

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN = ROOT/'ssm_rawfps_anchor_gpu_20261031'
HUMAN = ROOT/'ssm_rawfps_human_batch_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    if 'compared' not in (RUN/'queue_status.txt').read_text().splitlines():
        raise ValueError('anchored queue is not audited and compared')
    comparison = json.loads((RUN/'comparison.json').read_text())
    summaries = {x['id']:x for x in comparison['cases']}
    if sha(URDF) != comparison['urdf_sha256']:
        raise ValueError('URDF changed')
    limits={}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        if joint.get('type') == 'continuous': limits[joint.get('name')] = (-np.inf,np.inf)
        elif joint.get('type') == 'revolute':
            lim=joint.find('limit'); limits[joint.get('name')] = (float(lim.get('lower')),float(lim.get('upper')))
    results=[]
    # Walking has large baseline-vs-anchor branch differences and remains isolated.
    for sample in ('007410','011120'):
        source = HUMAN/sample; run = RUN/sample
        human_report = json.loads((source/'human_report.json').read_text())
        human_audit = json.loads((source/'human_audit.json').read_text())
        report = json.loads((run/'report.json').read_text()); checked = json.loads((run/'audit_report.json').read_text())
        if (sha(source/'human_report.json') != human_audit['human_report_sha256'] or
            sha(source/'human_rawfps.npz') != checked['source_sha256'] or
            sha(run/'report.json') != checked['report_sha256'] or sha(run/'windows.jsonl') != report['rows_sha256'] or
            sha(run/'audit_report.json') != summaries[sample]['anchored_audit_sha256']):
            raise ValueError('input or audit changed')
        with np.load(source/'human_rawfps.npz',allow_pickle=False) as f:
            raw_frames,raw_times = f['source_frames'].copy(),f['source_times'].copy()
        rows=[json.loads(x) for x in (run/'windows.jsonl').read_text().splitlines()]
        windows=[]
        for row in rows:
            p=run/row['artifact']
            if sha(p)!=row['artifact_sha256']: raise ValueError('window changed')
            with np.load(p,allow_pickle=False) as f:
                windows.append({'frames':f['sample_indices'].copy(),'root_wxyz_xyz':f['root_wxyz_xyz'].copy(),
                                'joint_pos':f['joint_pos'].copy(),'joint_names':f['joint_names'].copy()})
        merged=blend_windows(windows); indices=merged['frames']
        np.testing.assert_array_equal(indices,np.arange(1,len(raw_frames)))
        raw_grid=raw_times[indices]
        sampled=resample_robot_60_to_20(merged['blended'],raw_grid)
        filtered=sampled['states']; times=sampled['times']; names=list(merged['joint_names'])
        points=URDFFK(URDF).forward(filtered[:,4:7],filtered[:,[1,2,3,0]],filtered[:,7:],names)
        features=encode(points,foot_velocity_squared_threshold=.002)
        with torch.no_grad(): decoded=recover_from_ric(torch.from_numpy(features),22).numpy()
        error=float(np.linalg.norm(decoded-canonicalize_z_up(points)[:-1],axis=-1).max())
        if features.shape!=(len(times)-1,263) or error>1e-4 or not np.isfinite(features).all():
            raise ValueError('encoding self-check failed')
        if set(names)!=set(limits): raise ValueError('joint names mismatch')
        lower=np.array([limits[x][0] for x in names]); upper=np.array([limits[x][1] for x in names])
        violation=lambda x:float(np.maximum(np.maximum(lower-x,x-upper),0).max())
        clip_times=raw_times[::human_report['official_stride']]
        caption_records=[]
        for record in human_report['original_caption_records']:
            try:
                assoc=associate_caption(clip_sample_times=clip_times,feature_sample_times=times,
                    start_tag=record['tags'][0],end_tag=record['tags'][1],timed_tag_convention='unknown')
            except ValueError as exc:
                assoc={'status':'invalid_or_unknown_tags','reason':str(exc),
                       'matching_feature_intervals':[],'caption_accepted':False}
            caption_records.append({'caption':record['text'],'raw_tags':record['tags'],'association':assoc})
        out=HERE/(sample+'.npz')
        with out.open('xb') as f:
            np.savez_compressed(f,features=features,root_wxyz_xyz=filtered[:,:7],joint_pos=filtered[:,7:],
                sample_times=times,joint_names=np.asarray(names),points_z_up_world=points,decoded_points=decoded,
                raw_sample_indices=indices,raw_source_frames=raw_frames[indices],raw_blended=merged['blended'])
        result={'id':sample,'output_sha256':sha(out),'human_report_sha256':sha(source/'human_report.json'),
                'human_audit_sha256':sha(source/'human_audit.json'),'anchored_report_sha256':sha(run/'report.json'),
                'anchored_audit_sha256':sha(run/'audit_report.json'),'comparison_sha256':sha(RUN/'comparison.json'),
                'script_sha256':sha(Path(__file__)),
                'module_sha256':{x:sha(HERE/x) for x in ('blend_robot_windows.py','caption_interval_alignment.py',
                                  'feature263.py','g1_fk.py','resample_robot_60_to_20.py')},
                'source_sha256':human_report['source_sha256'],'caption_member_sha256':human_report['caption_member_sha256'],
                'source_group_candidate':human_report['source_parent'],'official_split':human_report['official_split'],
                'reference_fps':human_report['reference_fps'],'target_fps':20.,'raw_sample_indices':[int(indices[0]),int(indices[-1])],
                'raw_frames':[int(raw_frames[indices[0]]),int(raw_frames[indices[-1]])],
                'sample_time_range':[float(times[0]),float(times[-1])],'samples':len(times),'feature_shape':list(features.shape),
                'tail_after_last_target_s':sampled['discarded_tail_s'],'max_joint_limit_before_filter_rad':violation(merged['blended'][:,7:]),
                'max_joint_limit_after_filter_rad':violation(filtered[:,7:]),'decode_max_error_m':error,
                'anchored_overlap_max':{k:summaries[sample]['anchored_overlap'][k]['max'] for k in ('root_m','rotation_rad','joint_rad')},
                'caption_records':caption_records,'caption_accepted':False,'training_ready':False,
                'limitations':['Soft anchor, overlap blend and 60-to-20 Hz FIR are approximations, not author v2.',
                'Quaternion component filtering is approximate and uses offline context.',
                'Source FPS, body model, caption semantics, contact/collision and physical quality remain unaccepted.']}
        with (HERE/(sample+'_report.json')).open('x',encoding='utf8') as f:
            json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
        results.append(result); print(json.dumps({'id':sample,'samples':len(times),'training_ready':False}),flush=True)
    with (HERE/'batch_report.json').open('x',encoding='utf8') as f:
        json.dump({'cases':results,'training_ready':False},f,indent=2,allow_nan=False); f.write('\n')


if __name__ == '__main__': main()
