"""Crop 15 remaining long originals before GPU conversion, with provenance.

The first 300 source samples occupy 9.9667 s at the declared 30 Hz. The source
caption describes the full recording and is not accepted for this crop. Output
is isolated and is not training-qualified.
"""
import argparse,collections,hashlib,io,json,os,shutil,subprocess,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from omniretarget_bridge import convert_layout
from waist_lock_root_ik import fit_root_and_waist
from resample_antialias_omniretarget import resample_antialias
from feature263 import encode
from g1_fk import URDFFK
from g1_joint_limit_gate import check_joint_limits,read_joint_limits

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
PAIRS=ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl'
PROBE=ROOT/'omomo_object_caption_probe_20261101/frames_summary.jsonl'
ROBOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
EXCLUDE={'sub10_largebox_081','sub16_largebox_044'}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def choose(rows,used,per_group):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['sample_id'] not in used and r['frames']>=60 and r['nominal_last_sample_s']<=9.95:
            groups[r['subject_group']].append(r)
    chosen=[]
    for group,items in sorted(groups.items()):
        # Spread over distinct textual templates before filling the group.
        buckets=collections.defaultdict(list)
        for item in sorted(items,key=lambda x:x['sample_id']):buckets[item['caption']].append(item)
        picked=[]
        while (per_group is None or len(picked)<per_group) and any(buckets.values()):
            for caption in sorted(buckets):
                if buckets[caption] and (per_group is None or len(picked)<per_group):picked.append(buckets[caption].pop(0))
        if per_group is not None and len(picked)!=per_group:raise ValueError('insufficient nominal group coverage')
        chosen.extend(picked)
    if len(groups)!=6 or (per_group is not None and len(chosen)!=6*per_group):raise ValueError('unexpected group count')
    return chosen
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remaining-from',type=Path,help='existing selection manifest to exclude')
    args=parser.parse_args()
    start=time.perf_counter(); visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('0','1') or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('explicit single GPU required')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free,memory.total',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    gpu={int(v[0]):tuple(map(int,v[1:])) for line in lines if (v:=[x.strip() for x in line.split(',')])}
    util,free,total=gpu[int(visible)]
    if util>10 or free<total*.05+2048:raise RuntimeError('GPU headroom not available')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('less than 8 GiB disk headroom')
    if sha(ARCHIVE)!='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3':raise ValueError('archive hash changed')
    if sha(PAIRS)!='96f30c547ad4b7a6fe8a0d24f71ac9b6ae016ce3cb14eafa7a67dd40d50fb69f':raise ValueError('pair hash changed')
    rows=[json.loads(s) for s in PROBE.read_text().splitlines()]
    excluded=set(EXCLUDE); previous_hash=None
    if args.remaining_from:
        previous=args.remaining_from.resolve(strict=True)
        previous_hash=sha(previous)
        prior=json.loads(previous.read_text())['members']
        if not isinstance(prior,list) or len(prior)!=len(set(prior)):raise ValueError('invalid prior selection')
        excluded.update(Path(m).name.removesuffix('_original.npz') for m in prior)
    if args.remaining_from:
        raise ValueError('long-original selection is fixed; no prior selection argument')
    selected=sorted([r for r in rows if r['sample_id'] not in excluded and r['nominal_last_sample_s']>9.95],key=lambda r:r['sample_id'])
    if len(selected)!=15 or len({r['sample_id'] for r in selected})!=15:
        raise ValueError('expected 15 unique remaining long original tracks')
    pairing={r['motion_member']:r for r in (json.loads(s) for s in PAIRS.read_text().splitlines())}
    selection={'source_sha256':sha(PROBE),'prior_selection_sha256':previous_hash,
               'selection_method':'all 15 long originals excluded from previous full-track batch; first 300 source frames only; source caption not accepted for crop',
               'members':[r['motion_member'] for r in selected]}
    with (HERE/'selection.json').open('x') as f:json.dump(selection,f,indent=2);f.write('\n')
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.05,0)
    urdf29,urdf27=ROBOT/'g1_29dof.urdf',ROBOT/'g1_27dof.urdf'; fk=URDFFK(urdf27); limits=read_joint_limits(urdf27)
    successful=0; timing=[]; failures=[]; stopped_reason=None
    with zipfile.ZipFile(ARCHIVE) as z,(HERE/'records.jsonl').open('x',buffering=1) as record:
        for index,row in enumerate(selected):
            if index and index%5==0:
                # Wait out the preceding GPU work so this reading mostly reflects
                # other tasks; defer rather than compete with newly busy jobs.
                torch.cuda.synchronize();time.sleep(1.2)
                rows_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
                snap={int(v[0]):tuple(map(int,v[1:])) for line in rows_gpu if (v:=[x.strip() for x in line.split(',')])}
                current_util,current_free=snap[int(visible)]
                disk_free=shutil.disk_usage(ROOT).free
                if current_util>30 or current_free<2048 or disk_free<8*1024**3:
                    stopped_reason=f'GPU became busy or free resources fell: util={current_util}%, GPU free={current_free} MiB, disk free={disk_free} B'
                    print('stopped:',stopped_reason,flush=True);break
            item_start=time.perf_counter(); sid=row['sample_id']; print('start',sid,flush=True)
            try:
                pair=pairing[row['motion_member']]
                raw=z.read(row['motion_member'])
                if hashlib.sha256(raw).hexdigest()!=row['motion_member_sha256'] or pair['motion_member_sha256']!=row['motion_member_sha256'] or pair['caption']!=row['caption']:
                    raise ValueError('source/caption manifest mismatch')
                with np.load(io.BytesIO(raw),allow_pickle=False) as d:
                    fps=float(d['fps']); motion,obj=convert_layout(d['qpos'])
                if fps!=30 or obj is None or len(motion)!=row['frames']:raise ValueError('unexpected source frames/rate')
                source_total_frames=len(motion)
                motion=motion[:300];obj=obj[:300]
                fit_start=time.perf_counter()
                fit=fit_root_and_waist(motion,urdf29,urdf27,steps=120,learning_rate=.015,fps=fps,
                    foot_position_weight=2.,contact_velocity_weight=10.,device='cuda:0')
                torch.cuda.synchronize(); fit_sec=time.perf_counter()-fit_start
                root=np.concatenate([fit['root_quat_xyzw'][:,[3,0,1,2]],fit['root_pos']],axis=1)
                full=HERE/(sid+'_30hz.npz')
                with full.open('xb') as f:np.savez_compressed(f,root_wxyz_xyz=root,joint_pos=fit['joint_pos'],
                    joint_names=np.asarray(fit['joint_names']),object_wxyz_xyz=obj,sample_times=np.arange(len(root))/fps)
                q=np.concatenate([root,fit['joint_pos'],obj],axis=1); sampled=resample_antialias(q,30.,20.)
                angles=sampled[:,7:34]; names=list(fit['joint_names']); points=fk.forward(sampled[:,4:7],sampled[:,[1,2,3,0]],angles,names)
                features=encode(points,foot_velocity_squared_threshold=.002)
                check=check_joint_limits(angles,names,limits,tolerance_rad=1e-10)
                state=HERE/(sid+'_20hz.npz'); feature=HERE/(sid+'_263.npz')
                with state.open('xb') as f:np.savez_compressed(f,root_wxyz_xyz=sampled[:,:7],joint_pos=angles,
                    joint_names=np.asarray(names),object_wxyz_xyz=sampled[:,34:41],points_z_up=points,
                    sample_times=np.arange(len(sampled))/20.,source_frame_positions=np.arange(len(sampled))*1.5)
                with feature.open('xb') as f:np.savez_compressed(f,features=features)
                elapsed=time.perf_counter()-item_start; timing.append(elapsed); successful+=1
                result={'sample_id':sid,'source_member':row['motion_member'],'source_sha256':row['motion_member_sha256'],
                    'source_group':row['subject_group'],'caption':pair['caption'],'annotation_member':pair['annotation_member'],
                    'annotation_sha256':pair['annotation_member_sha256'],'text_status':'full_recording_caption_not_accepted_for_crop',
                    'license_status':'unverified','split':'unassigned','declared_fps':fps,'source_frames':len(root),
                    'source_total_frames':source_total_frames,'crop_source_start_frame':0,'crop_source_end_frame_exclusive':len(root),
                    'outside_crop_source_frames':source_total_frames-len(root),
                    'feature_frames':len(features),'state30':full.name,'state30_sha256':sha(full),
                    'state20':state.name,'state20_sha256':sha(state),'feature':feature.name,'feature_sha256':sha(feature),
                    'fit_seconds':fit_sec,'total_seconds':elapsed,'joint_limits':check,
                    'fit_mean_error_m':fit['final_mean_keypoint_error_m'],'fit_max_error_m':fit['final_max_keypoint_error_m'],
                    'source_tail_seconds':float((len(root)-1)/30.-(len(sampled)-1)/20.),
                    'technical_processing':'success','training_ready':False}
            except Exception as e:
                result={'sample_id':sid,'technical_processing':'failed','error':repr(e),'traceback':traceback.format_exc(),
                        'elapsed_seconds':time.perf_counter()-item_start,'training_ready':False}
                failures.append(sid)
            record.write(json.dumps(result,allow_nan=False)+'\n');record.flush()
            print('finish',sid,result['technical_processing'],flush=True)
    report={'script_sha256':sha(Path(__file__)),'fitter_sha256':sha(HERE/'waist_lock_root_ik.py'),
        'source_archive_sha256':sha(ARCHIVE),'pair_manifest_sha256':sha(PAIRS),'probe_sha256':sha(PROBE),
        'urdf29_sha256':sha(urdf29),'urdf27_sha256':sha(urdf27),'selection_sha256':sha(HERE/'selection.json'),
        'records_sha256':sha(HERE/'records.jsonl'),'selected':len(selected),'attempted':successful+len(failures),
        'stopped_reason':stopped_reason,'successful':successful,'failed':failures,
        'total_elapsed_seconds_including_setup':time.perf_counter()-start,'successful_total_seconds':sum(timing),
        'median_seconds_per_success':float(np.median(timing)) if timing else None,'gpu':visible,
        'gpu_precheck_util_percent':util,'gpu_precheck_free_mb':free,'pytorch_memory_fraction':.05,
        'pytorch_max_reserved_bytes':int(torch.cuda.max_memory_reserved()),'training_ready':False,
        'limitations':['The full-recording caption is not accepted for this crop; no semantic, source identity, license or physical acceptance.',
          '30Hz source declared; foot low-speed is a proxy; fitting and offline quaternion filtering approximate.',
          'Per-case timing excludes setup; throughput is not a training-start ETA.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
