"""Fit two pinned robot-to-robot candidates at source rate in an isolated run.

The source is already a 29-DoF OmniRetarget robot trajectory; this is not the
paper's human-to-G1 PyRoki optimizer. Loss weights and contact proxies are
experimental. Neither these outputs nor their paired text are training-ready.
"""
import hashlib,io,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
import zipfile
from omniretarget_bridge import convert_layout
from waist_lock_root_ik import fit_root_and_waist

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip';PAIRS=ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl'
ROBOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
WANTED=('sub10_largebox_081','sub16_largebox_044')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('0','1') or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('one explicit physical GPU required')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free,memory.total',
                                   '--format=csv,noheader,nounits'],text=True).splitlines()
    gpus={int(x[0]):tuple(map(int,x[1:])) for line in lines if (x:=[v.strip() for v in line.split(',')])}
    utilization,free,total=gpus[int(visible)]
    if utilization>10 or free < total*.05+2048:raise RuntimeError('GPU lacks idle time or memory headroom')
    if sha(ARCHIVE)!='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3' or sha(PAIRS)!='96f30c547ad4b7a6fe8a0d24f71ac9b6ae016ce3cb14eafa7a67dd40d50fb69f':
        raise ValueError('pinned source or caption manifest changed')
    rows=[json.loads(line) for line in PAIRS.read_text().splitlines()]
    selected=[x for x in rows if x['nominal_action_key'] in WANTED and x['variant']=='original']
    if len(selected)!=2 or {x['nominal_action_key'] for x in selected}!=set(WANTED):raise ValueError('candidate coverage mismatch')
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.05,0)
    u29,u27=ROBOT/'g1_29dof.urdf',ROBOT/'g1_27dof.urdf';results=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for row in selected:
            sid=row['nominal_action_key'];print('fitting '+sid,flush=True)
            source=z.read(row['motion_member'])
            if hashlib.sha256(source).hexdigest()!=row['motion_member_sha256']:raise ValueError('member hash mismatch')
            with np.load(io.BytesIO(source),allow_pickle=False) as f:
                fps=float(f['fps']);motion,obj=convert_layout(f['qpos'])
            if fps!=30 or obj is None:raise ValueError('unexpected source format')
            fit=fit_root_and_waist(motion,u29,u27,steps=120,learning_rate=.015,fps=fps,
                    foot_position_weight=2.,contact_velocity_weight=10.,device='cuda:0')
            torch.cuda.synchronize()
            path=HERE/(sid+'_27dof_30hz.npz')
            with path.open('xb') as f:
                np.savez_compressed(f,root_wxyz_xyz=np.concatenate([fit['root_quat_xyzw'][:,[3,0,1,2]],fit['root_pos']],axis=1),
                    joint_pos=fit['joint_pos'],joint_names=np.asarray(fit['joint_names']),
                    sample_times=np.arange(len(motion),dtype=float)/fps,object_wxyz_xyz=obj)
            results.append({'sample_id':sid,'source_member':row['motion_member'],'source_sha256':row['motion_member_sha256'],
                'source_caption':row['caption'],'annotation_sha256':row['annotation_member_sha256'],
                'output':path.name,'output_sha256':sha(path),'frames':len(motion),'declared_fps':fps,
                'initial_mean_keypoint_error_m':fit['initial_mean_keypoint_error_m'],
                'final_mean_keypoint_error_m':fit['final_mean_keypoint_error_m'],
                'final_max_keypoint_error_m':fit['final_max_keypoint_error_m'],
                'max_root_translation_m':fit['max_root_translation_m'],'max_root_rotation_rad':fit['max_root_rotation_rad'],
                'text_semantic_accepted':False,'training_ready':False})
            print('finished '+sid,flush=True)
    if sha(ARCHIVE)!='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3' or sha(PAIRS)!='96f30c547ad4b7a6fe8a0d24f71ac9b6ae016ce3cb14eafa7a67dd40d50fb69f':raise RuntimeError('inputs changed')
    report={'source_archive_sha256':sha(ARCHIVE),'pair_manifest_sha256':sha(PAIRS),'urdf29_sha256':sha(u29),
        'urdf27_sha256':sha(u27),'script_sha256':sha(Path(__file__)),
        'fitter_sha256':sha(Path(__file__).with_name('waist_lock_root_ik.py')),
        'physical_cuda_device':visible,'pytorch_memory_fraction':.05,'gpu_precheck_util_percent':utilization,
        'gpu_precheck_free_mb':free,'max_reserved_bytes':int(torch.cuda.max_memory_reserved()),
        'parameters':{'steps':120,'learning_rate':.015,'foot_position_weight':2.,'contact_velocity_weight':10.,
            'source_rate_hz':30.,'contact_speed_proxy_m_s':.03},'cases':results,'training_ready':False,
        'limitations':['Near-sample foot contact is a velocity proxy, not measured force.',
            'Source fps, original caption association, geometry, license chain and physics are not accepted.',
            'The fitted outputs remain at 30 Hz; no 20 Hz/263D export or training is claimed.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
