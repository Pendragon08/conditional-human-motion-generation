"""Small CPU/GPU numerical comparison for an isolated robot-to-robot fitter.

This is neither human-source retargeting nor data/physics/training acceptance.
CUDA_VISIBLE_DEVICES must explicitly name one physical GPU. The PyTorch
allocator is limited to 5% before any model is initialized.
"""
import hashlib,io,json,os
from pathlib import Path
import numpy as np
import torch
import zipfile
from omniretarget_bridge import convert_layout
from waist_lock_root_ik import fit_root_and_waist

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
PAIRS=ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl'
ROBOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('explicit single physical CUDA device required')
    if sha(ARCHIVE)!='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3' or sha(PAIRS)!='96f30c547ad4b7a6fe8a0d24f71ac9b6ae016ce3cb14eafa7a67dd40d50fb69f':
        raise ValueError('pinned archive or pair inventory changed')
    p=[json.loads(line) for line in PAIRS.read_text().splitlines()]
    p=[x for x in p if x['nominal_action_key']=='sub10_largebox_081' and x['variant']=='original']
    if len(p)!=1:raise ValueError('candidate not unique')
    with zipfile.ZipFile(ARCHIVE) as z:raw=z.read(p[0]['motion_member'])
    if hashlib.sha256(raw).hexdigest()!=p[0]['motion_member_sha256']:raise ValueError('member hash mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        fps=float(f['fps']);motion,_=convert_layout(f['qpos'][:12])
    if fps!=30:raise ValueError('unexpected declared FPS')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.05,0)
    u29,u27=ROBOT/'g1_29dof.urdf',ROBOT/'g1_27dof.urdf'
    cpu=fit_root_and_waist(motion,u29,u27,steps=5,fps=fps,foot_position_weight=2.,contact_velocity_weight=10.,device='cpu')
    gpu=fit_root_and_waist(motion,u29,u27,steps=5,fps=fps,foot_position_weight=2.,contact_velocity_weight=10.,device='cuda:0')
    torch.cuda.synchronize()
    diffs={key:float(np.max(np.abs(cpu[key]-gpu[key]))) for key in ('joint_pos','root_pos','root_quat_xyzw')}
    if cpu['joint_names']!=gpu['joint_names'] or max(diffs.values())>1e-5:
        raise RuntimeError('CPU/GPU fit mismatch '+repr(diffs))
    report={'source_archive_sha256':sha(ARCHIVE),'member':p[0]['motion_member'],
            'member_sha256':p[0]['motion_member_sha256'],'urdf29_sha256':sha(u29),'urdf27_sha256':sha(u27),
            'fitter_sha256':sha(Path(__file__).with_name('waist_lock_root_ik.py')),
            'physical_cuda_visible_device':os.environ['CUDA_VISIBLE_DEVICES'],'pytorch_memory_fraction':.05,
            'max_reserved_bytes':int(torch.cuda.max_memory_reserved()),'frames_compared':len(motion),'steps':5,
            'foot_position_weight':2.,'contact_velocity_weight':10.,
            'cpu_gpu_max_abs_differences':diffs,'training_ready':False,
            'limitations':['Short numerical comparison only, not a full optimization or model training.',
                'Fitter is an approximation; source subject, semantics, contacts and dynamics remain unaccepted.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
