"""Reuse published AMASS-HumanML3D G1 motion/text, with nominal60->20Hz.

CPU-only paper-format conversion, not human-to-robot retargeting. Every29DoF
state and published caption is retained; actual physical clock/recording identity
is not inferred from matching IDs. Output is a separately named temporal profile.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import time
import zipfile
import numpy as np

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
P=Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
H=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public')
sys.path[:0]=[str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(R/'antialias_v3_20261030/src'),str(T),str(R/'stage3_pick29_20261103/src')]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
from stage3_pick29_core import BODY_JOINTS
import torch

FLOOR=8*1024**3+456*1024**2
EXPECTED_SOURCE='53a3fff8c2928d3e36b06b1b8bce1148675854ff3c5ef5bad0e56f295abe302d'
EXPECTED_URDF='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--publisher-contract',type=Path,required=True)
    a=p.parse_args();out=a.output_dir;out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    began=time.monotonic();records=out/'records.jsonl';records.open('x').close()
    status={'status':'preparing','pid':os.getpid(),'profile':'publisher_nominal60_to20Hz_29DoF_weak_text',
            'completed':0,'converted':0,'failed':0,'gpu_used':False,'retargeting_performed':False}
    def save(**values):
        status.update(values);tmp=out/'status.tmp';tmp.write_text(json.dumps(status,indent=2,allow_nan=False));tmp.replace(out/'status.json')
    try:
        save()
        assert sha(P)==EXPECTED_SOURCE and sha(U)==EXPECTED_URDF
        contract=a.publisher_contract.read_bytes()
        assert b'60 Hz' in contract and b'29-DoF order' in contract
        assert hashlib.sha1(b'blob '+str(len(contract)).encode()+b'\0'+contract).hexdigest()=='04ebf854bf9c6deaac9623509c4b49c6794c7041','publisher contract blob differs'
        with P.open('rb') as f:raw=pickle.load(f)
        data=raw['data_dict'];assert len(data)==11424 and set(raw['name_list'])==set(data)
        assert sum(len(v['motion']) for v in data.values())==1156021
        official={s:set((H/'HumanML3D'/f'{s}.txt').read_text().splitlines()) for s in ['train','val','test']}
        provider={s:set((P.parent/f'{s}.txt').read_text().splitlines()) for s in ['train','val','test']}
        assert set.union(*provider.values())==set(data)
        with (H/'index.csv').open(newline='') as f:index={Path(r['new_name']).stem:r for r in csv.DictReader(f)}
        def source_group(sid):
            return index[sid.removeprefix('M')]['source_path'] if sid.removeprefix('M') in index else 'unresolved_id:'+sid.removeprefix('M')
        priority={'train':0,'val':1,'test':2}
        official_id={sid:s for s,ids in official.items() for sid in ids}
        provider_id={sid:s for s,ids in provider.items() for sid in ids}
        group_split={}
        for sid,s in official_id.items():
            group=source_group(sid)
            if group not in group_split or priority[s]>priority[group_split[group]]:group_split[group]=s
        predicted=sum(min(200,(len(v['motion'])-1)//3+1) for v in data.values())
        estimate=predicted*(36*8+66*8+263*4)+len(data)*8192
        assert shutil.disk_usage('/data').free>FLOOR+estimate+128*1024**2,'full-output raw budget plus floor'
        inputs={'corpus':{'path':str(P),'sha256':EXPECTED_SOURCE},'target_urdf':{'path':str(U),'sha256':EXPECTED_URDF},
            'publisher_contract':{'path':str(a.publisher_contract),'sha256':sha(a.publisher_contract),'git_blob':'04ebf854bf9c6deaac9623509c4b49c6794c7041'},
            'index':{'path':str(H/'index.csv'),'sha256':sha(H/'index.csv')},
            'source_caption_zip':{'path':str(H/'HumanML3D/texts.zip'),'sha256':sha(H/'HumanML3D/texts.zip')}}
        with (out/'inputs.json').open('x') as f:json.dump(inputs,f,indent=2)
        (out/'motions').mkdir()
        fk=URDFFK(U);assert set(fk.actuated)==set(BODY_JOINTS)
        stats_n=0;stats_sum=np.zeros(263);stats_square=np.zeros(263)
        split_counts=Counter();caption_counts=Counter();failed=[];output_bytes=0;all_rows=[]
        with zipfile.ZipFile(H/'HumanML3D/texts.zip') as texts:
            members={Path(x.filename).stem:x.filename for x in texts.infolist() if x.filename.endswith('.txt') and not x.is_dir()}
            for ordinal,sid in enumerate(sorted(data)):
                assert re.fullmatch(r'\d{6}|M\d{6}',sid),'unsafe/noncanonical motion ID'
                assert shutil.disk_usage('/data').free>FLOOR+16*1024**2,'per-record disk guard'
                entry=data[sid];motion=np.asarray(entry['motion'])
                captions=entry['text'];assert isinstance(captions,list) and captions
                row={'sample_id':sid,'dataset_source':'ReactiveBFM/AMASS-HumanML3D',
                    'source_corpus_sha256':EXPECTED_SOURCE,'source_motion_sha256':hashlib.sha256(motion.tobytes()).hexdigest(),
                    'source_frames':len(motion),'source_layout':'xyz_xyzw_29joint','source_fps_nominal':60.,'target_fps':20.,
                    'time_basis':'publisher60Hz declaration, no per-record clock field; not independently verified real recording clock',
                    'captions':captions,'caption_policy':'publisher captions verbatim weak supervision, no semantic relabeling',
                    'provider_split':provider_id[sid],'official_humanml_split':official_id.get(sid),
                    'exact_indexed_source_group':source_group(sid),'split':group_split.get(source_group(sid),provider_id[sid]),
                    'retargeting_performed':False,'joint_names':BODY_JOINTS}
                assert sid in members
                text_bytes=texts.read(members[sid]);original=[x.split('#')[0].strip() for x in text_bytes.decode('utf8').splitlines() if x.strip()]
                row.update(humanml_caption_member=members[sid],humanml_caption_member_sha256=hashlib.sha256(text_bytes).hexdigest(),
                    exact_caption_matches=sum(c['caption'] in original for c in captions),source_captions=original)
                caption_counts['published_sentences']+=len(captions)
                caption_counts['exact_original_sentences']+=row['exact_caption_matches']
                caption_counts['records_with_nonexact_caption']+=row['exact_caption_matches']!=len(captions)
                try:
                    assert motion.ndim==2 and motion.shape[1]==36 and len(motion)==entry['length'] and np.isfinite(motion).all()
                    qpos=np.column_stack((motion[:,[6,3,4,5]],motion[:,:3],motion[:,7:]))
                    q=resample_antialias(qpos,60.,20.,max_seconds=10.)
                    points=fk.forward(q[:,4:7],q[:,[1,2,3,0]],q[:,7:36],BODY_JOINTS)
                    features=encode_signfixed(points,foot_velocity_squared_threshold=.002)
                    assert features.shape==(len(q)-1,263) and np.isfinite(features).all()
                    assert len(q)<=200 and (len(q)-1)/20 <= (len(motion)-1)/60+1e-10
                    rel='motions/'+sid+'.npz';dest=out/rel;partial=dest.with_suffix('.partial')
                    meta={'sample_id':sid,'source_corpus_sha256':EXPECTED_SOURCE,'source_motion_sha256':row['source_motion_sha256'],
                        'source_fps_nominal':60.,'target_fps':20.,'time_basis':row['time_basis'],'captions':captions}
                    with partial.open('xb') as f:np.savez_compressed(f,qpos29_wxyz_xyz=q,points_z_up=points,features=features,
                        time_seconds=np.arange(len(q))/20.,joint_names=np.array(BODY_JOINTS),
                        metadata_utf8=np.frombuffer(json.dumps(meta,ensure_ascii=False).encode(),dtype=np.uint8))
                    with np.load(partial,allow_pickle=False) as z:
                        assert np.array_equal(z['qpos29_wxyz_xyz'],q) and np.array_equal(z['features'],features)
                    assert not dest.exists();partial.rename(dest)
                    row.update(status='converted',feature_path=rel,feature_sha256=sha(dest),states=len(q),feature_frames=len(features),
                        source_nominal_duration_s=(len(motion)-1)/60.,output_duration_s=(len(q)-1)/20.,
                        truncated=(len(motion)-1)/60.>10.,bytes=dest.stat().st_size)
                    output_bytes+=dest.stat().st_size;split_counts[row['split']]+=1
                    if row['split']=='train':
                        stats_n+=len(features);stats_sum+=features.astype(float).sum(0);stats_square+=np.square(features.astype(float)).sum(0)
                except Exception as e:
                    row.update(status='conversion_failed_preserved_source',error=repr(e));failed.append(sid)
                with records.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                all_rows.append(row)
                assert output_bytes<1024**3,'per-build output budget1GiB exceeded'
                if ordinal%100==0 or ordinal==len(data)-1:
                    save(status='converting',completed=ordinal+1,converted=ordinal+1-len(failed),failed=len(failed),
                        output_bytes=output_bytes,elapsed_seconds=time.monotonic()-began,free_bytes=shutil.disk_usage('/data').free)
        assert sha(P)==EXPECTED_SOURCE,'read-only source changed during conversion'
        good=[r for r in all_rows if r['status']=='converted']
        mean=stats_sum/max(1,stats_n);std=np.sqrt(np.maximum(stats_square/max(1,stats_n)-mean**2,0.));std=np.maximum(std,1e-6)
        with (out/'stats.npz').open('xb') as f:np.savez(f,feature_mean=mean.astype(np.float32),feature_std=std.astype(np.float32),train_feature_frames=stats_n)
        manifest={'records':good,'source_inputs':inputs,'temporal_profile':'publisher_declared60_to20Hz',
            'split_policy':'retain all converted IDs; assign each exact indexed AMASS source path to highest-priority official test>val>train',
            'limitations':['Exact index path and ID are metadata associations, not original recording identity proof.',
                'Publisher60Hz used as nominal time; absent per-record clock metadata, prior20Hz sensitivity findings retained.',
                'Same-ID source text is weak supervision; exact text match does not prove semantics.',
                'Antialias quaternion component filtering and263D sign alignment are project adaptations.',
                'Target URDF FK adopted;263D22points cannot reconstruct all29DoF losslessly.'],
            'caption_policy':'all publisher captions retained verbatim; differences against HumanML recorded without automatic rejection'}
        with (out/'manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False)
        result={'status':'paired_profile_built' if not failed else 'paired_profile_partial_failures_preserved',
            'source_trajectories':11424,'converted_trajectories':len(good),'failed_ids':failed,
            'source_frames':1156021,'fps':20,'body_dof':29,'features':263,'maximum_states':200,
            'caption_counts':dict(caption_counts),'split_counts':dict(split_counts),'train_feature_frames':stats_n,
            'manifest_sha256':sha(out/'manifest.json'),'records_sha256':sha(records),'stats_sha256':sha(out/'stats.npz'),
            'output_bytes':output_bytes,'elapsed_seconds':time.monotonic()-began,
            'nominal_temporal_basis':'publisher_declared60Hz','actual_recording_clock_verified':False,
            'source_and_models_unchanged':True,'new_training_started':False,'independent_file_audit':'pending'}
        with (out/'report.json').open('x') as f:json.dump(result,f,indent=2)
        save(**result)
        print(json.dumps(result),flush=True)
    except BaseException as e:
        save(status='failed_preserved_prefix',error=repr(e),elapsed_seconds=time.monotonic()-began)
        raise


if __name__=='__main__':main()
