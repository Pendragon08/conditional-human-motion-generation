"""CPU comparison of estimated SMPL-X foot positions after two resamplers."""
import contextlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import smplx
from humanml_time_mapping import aligned_grid
from humanml_rotation_resample import apply_grid
from humanml_fir_resample import fir_resample

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ROOT=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
OUT=BASE/'ssm_fir_smplx_foot_comparison_20261030'

def main():
    torch.set_num_threads(1)
    ids=list(map(json.loads,(BASE/'ssm_id_alignment_20261030/ids.jsonl').open()))
    receipts={r['source_group']:r for r in map(json.loads,(BASE/'ssm_receipt_audit_20261030/rows.jsonl').open())}
    selected=[r for r in ids if r['status']=='prior_marker_aligned_non20_requires_reprocessing' and not r['mirrored']]
    if len(selected)!=13:raise ValueError('unexpected source count')
    models={};rows=[]
    for record in selected:
        receipt=json.loads(Path(receipts[record['source_path']]['receipt']).read_text())
        with np.load(receipt['smplx_file'],allow_pickle=False) as x:
            pos=np.asarray(x['trans'],dtype=np.float64);r=np.asarray(x['root_orient']);body=np.asarray(x['pose_body'])
            betas=np.asarray(x['betas'],dtype=np.float32).reshape(-1)[:16];gender=str(x['gender'].item())
        grid=aligned_grid(record['reference_raw_frame_count'],record['reference_fps'],record['start'],record['stop'])
        pose=np.concatenate([r,body],axis=1); selected_idx=np.asarray(grid['source_indices'])
        p0,r0=apply_grid(pos,pose,grid)
        p1,r1,_=fir_resample(pos[selected_idx],pose[selected_idx],grid['actual_reference_sample_fps'])
        if p0.shape!=p1.shape:raise ValueError('output grid mismatch')
        if gender not in ('female','male','neutral') or betas.shape!=(16,):raise ValueError('model metadata invalid')
        if gender not in models:
            with contextlib.redirect_stdout(sys.stderr):
                models[gender]=smplx.create(str(ROOT),model_type='smplx',gender=gender,num_betas=16,use_pca=False,ext='npz').cpu()
        def compute(p,r):
            count=len(p); tensor=lambda a:torch.as_tensor(a,dtype=torch.float32)
            zero=lambda n:torch.zeros((count,n),dtype=torch.float32)
            with torch.no_grad():
                out=models[gender](betas=tensor(np.broadcast_to(betas,(count,16)).copy()),
                    global_orient=tensor(r[:,:3]),body_pose=tensor(r[:,3:]),transl=tensor(p),
                    left_hand_pose=zero(45),right_hand_pose=zero(45),jaw_pose=zero(3),
                    leye_pose=zero(3),reye_pose=zero(3),expression=zero(models[gender].num_expression_coeffs),return_verts=False)
            # Standard SMPL-X joint ordering: ankles 7/8 and feet 10/11.
            return out.joints[:,[7,8,10,11],:].detach().cpu().numpy().astype(np.float64)
        foot0,foot1=compute(p0,r0),compute(p1,r1)
        diff=np.linalg.norm(foot1-foot0,axis=-1)
        speed0=np.linalg.norm(np.diff(foot0,axis=0)*20,axis=-1)
        speed1=np.linalg.norm(np.diff(foot1,axis=0)*20,axis=-1)
        rows.append({'motion_id':record['motion_id'],'frames':len(p0),'source':record['source_path'],
                     'foot_joint_position_diff_mean_m':float(diff.mean()),'foot_joint_position_diff_max_m':float(diff.max()),
                     'unfiltered_foot_joint_speed_mean_m_s':float(speed0.mean()),
                     'fir_foot_joint_speed_mean_m_s':float(speed1.mean()),
                     'foot_speed_absolute_difference_mean_m_s':float(np.abs(speed1-speed0).mean())})
    OUT.mkdir(mode=0o700)
    with (OUT/'rows.jsonl').open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    summary={'recordings':len(rows),'per_clip_max_foot_difference_worst_m':max(r['foot_joint_position_diff_max_m'] for r in rows),
             'mean_of_clip_foot_difference_mean_m':float(np.mean([r['foot_joint_position_diff_mean_m'] for r in rows])),
             'mean_foot_speed_unfiltered_m_s':float(np.mean([r['unfiltered_foot_joint_speed_mean_m_s'] for r in rows])),
             'mean_foot_speed_fir_m_s':float(np.mean([r['fir_foot_joint_speed_mean_m_s'] for r in rows])),
             'limitations':['Human SMPL-X foot joints, not robot collision geometry or contact forces.',
                            'No ground plane, environment, task object or dynamics evaluated.',
                            'Locked-head model, zero hand/face pose, FIR quaternion component filtering are approximations.',
                            'Comparisons are method differences, not ground-truth errors; source hashes from previous audit not recomputed here.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
