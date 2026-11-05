"""Independently reread the complete nominal-publisher60 paired dataset profile.

File/source/caption/time/grid/statistics checks; no added semantic adjudication,
physics screening, source-data edit or claimed actual-clock verification.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path[:0]=[str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(R/'antialias_v3_20261030/src'),
    '/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen',str(R/'stage3_pick29_20261103/src')]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
from stage3_pick29_core import BODY_JOINTS


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    began=time.monotonic();d=a.dataset;r=json.loads((d/'report.json').read_text())
    assert r['status']=='paired_profile_built' and r['converted_trajectories']==11424 and not r['failed_ids']
    assert sha(d/'manifest.json')==r['manifest_sha256'] and sha(d/'stats.npz')==r['stats_sha256']
    assert sha(d/'records.jsonl')==r['records_sha256']
    m=json.loads((d/'manifest.json').read_text());rows=m['records'];inputs=m['source_inputs']
    for v in inputs.values():assert sha(v['path'])==v['sha256']
    with Path(inputs['corpus']['path']).open('rb') as f:raw=pickle.load(f)['data_dict']
    assert len(rows)==len(raw)==11424 and {x['sample_id'] for x in rows}==set(raw)
    fk=URDFFK(inputs['target_urdf']['path'])
    selected=set(range(0,len(rows),max(1,len(rows)//30)))|{0,len(rows)-1}
    s=np.zeros(263);sq=np.zeros(263);n=0;groups={};duplicates={};split=Counter();captions=0
    resample_error=fk_error=feature_error=0.
    for i,row in enumerate(rows):
        sid=row['sample_id'];source=raw[sid];motion=np.asarray(source['motion']);file=(d/row['feature_path']).resolve()
        assert file.is_relative_to(d.resolve()) and sha(file)==row['feature_sha256']
        assert hashlib.sha256(motion.tobytes()).hexdigest()==row['source_motion_sha256']
        assert source['text']==row['captions'];captions+=len(row['captions'])
        with np.load(file,allow_pickle=False) as z:
            q=z['qpos29_wxyz_xyz'];points=z['points_z_up'];features=z['features'];times=z['time_seconds']
            assert z['joint_names'].tolist()==BODY_JOINTS
            packed=json.loads(z['metadata_utf8'].tobytes());assert packed['sample_id']==sid and packed['captions']==row['captions']
        assert q.shape==(row['states'],36) and features.shape==(len(q)-1,263) and points.shape==(len(q),22,3)
        assert all(np.isfinite(v).all() for v in [q,points,features,times])
        assert len(q)==min(200,(len(motion)-1)//3+1) and np.array_equal(times,np.arange(len(q))/20.)
        assert np.allclose(np.linalg.norm(q[:,:4],axis=-1),1.,atol=1e-5,rtol=0)
        assert row['source_fps_nominal']==60. and row['target_fps']==20.
        group=row['exact_indexed_source_group'];groups.setdefault(group,set()).add(row['split']);split[row['split']]+=1
        fingerprint=hashlib.sha256(features.tobytes()).hexdigest();duplicates.setdefault(fingerprint,[]).append((sid,row['split']))
        if row['split']=='train':
            n+=len(features);s+=features.astype(float).sum(0);sq+=np.square(features.astype(float)).sum(0)
        if i in selected:
            source_q=np.column_stack((motion[:,[6,3,4,5]],motion[:,:3],motion[:,7:]))
            recomputed=resample_antialias(source_q,60.,20.,max_seconds=10.)
            resample_error=max(resample_error,float(np.max(np.abs(q-recomputed))))
            expected=fk.forward(q[:,4:7],q[:,[1,2,3,0]],q[:,7:36],BODY_JOINTS)
            fk_error=max(fk_error,float(np.max(np.abs(expected-points))))
            expected_feature=encode_signfixed(expected,foot_velocity_squared_threshold=.002)
            feature_error=max(feature_error,float(np.max(np.abs(expected_feature-features))))
    assert all(len(x)==1 for x in groups.values()),'indexed source group crosses splits'
    assert resample_error<1e-10 and fk_error<1e-10 and feature_error<1e-6
    with np.load(d/'stats.npz',allow_pickle=False) as z:
        mean=s/n;std=np.maximum(np.sqrt(np.maximum(sq/n-mean**2,0.)),1e-6)
        assert np.allclose(z['feature_mean'],mean,atol=1e-6,rtol=1e-6)
        assert np.allclose(z['feature_std'],std,atol=1e-6,rtol=1e-6)
        assert int(z['train_feature_frames'])==n
    cross=[v for v in duplicates.values() if len({x[1] for x in v})>1]
    out={'status':'paired_profile_file_audit_passed','trajectories_checked':len(rows),'captions_bound':captions,
        'split_counts':dict(split),'exact_indexed_source_groups':len(groups),'source_group_split_overlap':0,
        'stratified_recompute_records':len(selected),'resample_max_error':resample_error,'FK_max_error':fk_error,
        'encoder_max_error':feature_error,'cross_split_exact_feature_duplicate_groups':cross,
        'train_stats_recomputed_frames':n,'profile':'publisher_nominal60_to20Hz_29DoF_weak_text',
        'actual_recording_clock_verified':False,'semantic_relabeling_or_physics_filter_added':False,
        'manifest_sha256':r['manifest_sha256'],'report_sha256':sha(d/'report.json'),
        'elapsed_seconds':time.monotonic()-began,'new_training_started':False}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
