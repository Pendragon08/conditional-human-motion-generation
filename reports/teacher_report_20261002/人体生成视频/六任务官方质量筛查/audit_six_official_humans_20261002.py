"""Read-only CPU screening of released HumanML skeletons; not dynamics certification.

Source arrays are not changed, discarded, smoothed or reused for policy training.
All thresholds are exploratory reporting thresholds, not DreamControl rules.
"""
from pathlib import Path
import csv, datetime, hashlib, json, argparse
import numpy as np

REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
TEACHER=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
SOURCE=TEACHER/'人体生成视频/六任务官方原始样本'
OUT=REPO/'results/human_generation_quality_20261002'
CHAINS=((0,2,5,8,11),(0,1,4,7,10),(0,3,6,9,12,15),(9,14,17,19,21),(9,13,16,18,20))
EDGES=tuple((a,b) for c in CHAINS for a,b in zip(c[:-1],c[1:]))
CONFIG={
    'fps':20,'vertical_axis':'y','ground_reference_m':0.0,'foot_joint_indices':[10,11],
    'foot_penetration_m':-0.03,'penetration_min_consecutive_frames':3,
    'near_ground_band_m':[-0.03,0.05],'slide_planar_speed_m_s':0.2,
    'slide_min_consecutive_intervals':3,
    'bone_deviation_relative_to_own_median':0.10,'bone_min_consecutive_frames':3,
    'large_joint_displacement_m_per_interval':0.15,
    'both_feet_airborne_height_m':0.1,'airborne_min_consecutive_frames':6,
    'sensitivity_slide_speed_m_s':0.4,'sensitivity_bone_relative_deviation':0.2,
    'edges':[list(e) for e in EDGES],
    'status':'exploratory_proxy_thresholds_not_paper_filters',
    'run_rule':'Consecutive events must persist on the same foot or same bone; do not OR across entities first.',
    'duration_rule':'k sampled frames span (k-1)/fps seconds; k velocity intervals span k/fps seconds.',
    'classification':'self_supervised_proxy',
    'not_verified':['inverse_dynamics','torque_limits','friction_cones','balance','mesh_self_collision','scene_collision','contact_forces','text_semantics','human_judgments'],
    'action':'report_only_no_sample_rejection_no_policy_change'
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def longest(mask):
    best=cur=0
    for active in mask:
        cur=cur+1 if active else 0
        best=max(best,cur)
    return int(best)

def per_entity_runs(mask):
    assert mask.ndim==2
    return [longest(mask[:,i]) for i in range(mask.shape[1])]

def metrics(m):
    assert m.ndim==3 and m.shape[1:]==(22,3) and len(m)>=2
    assert np.isfinite(m).all()
    feet=m[:,CONFIG['foot_joint_indices'],:]; y=feet[:,:,1]
    penetration=per_entity_runs(y<CONFIG['foot_penetration_m'])
    planar_speed=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0),axis=2)*CONFIG['fps']
    lo,hi=CONFIG['near_ground_band_m']
    near=(y[:-1]>=lo)&(y[:-1]<=hi)&(y[1:]>=lo)&(y[1:]<=hi)
    slide=near&(planar_speed>CONFIG['slide_planar_speed_m_s'])
    slide_runs=per_entity_runs(slide)
    fast_slide=per_entity_runs(near&(planar_speed>CONFIG['sensitivity_slide_speed_m_s']))
    lengths=np.stack([np.linalg.norm(m[:,a]-m[:,b],axis=1) for a,b in EDGES],axis=1)
    medians=np.median(lengths,axis=0)
    # This is an explicitly self-referenced geometric proxy, not anatomical GT.
    degenerate=medians<=1e-8
    dev=np.abs(lengths-medians)/np.maximum(medians,1e-8)
    bone_runs=per_entity_runs(dev>CONFIG['bone_deviation_relative_to_own_median'])
    bone_runs20=per_entity_runs(dev>CONFIG['sensitivity_bone_relative_deviation'])
    displacement=np.linalg.norm(np.diff(m,axis=0),axis=2)
    air=longest((y>CONFIG['both_feet_airborne_height_m']).all(axis=1))
    flags={
        'foot_below_nominal_ground':max(penetration)>=CONFIG['penetration_min_consecutive_frames'],
        'any_foot_y_below_zero':bool((y<0).any()),
        'near_ground_slide_proxy':max(slide_runs)>=CONFIG['slide_min_consecutive_intervals'],
        'bone_length_variation_proxy':max(bone_runs)>=CONFIG['bone_min_consecutive_frames'],
        'large_joint_step_descriptive':float(displacement.max())>CONFIG['large_joint_displacement_m_per_interval'],
        'airborne_descriptive':air>=CONFIG['airborne_min_consecutive_frames']
    }
    return {
        **flags,
        'geometry_warning_any':any(flags[k] for k in ['foot_below_nominal_ground','near_ground_slide_proxy','bone_length_variation_proxy']),
        'foot_min_y_m':float(y.min()),'penetration_runs_per_foot_frames':penetration,
        'slide_runs_per_foot_intervals':slide_runs,'slide_active_intervals_per_foot':slide.sum(axis=0).astype(int).tolist(),
        'max_foot_planar_speed_m_s':float(planar_speed.max()),
        'bone_runs_per_edge_frames':bone_runs,'bone_median_lengths_m':medians.tolist(),
        'max_bone_absolute_deviation_m':float(np.abs(lengths-medians).max()),
        'max_bone_relative_deviation':float(dev.max()),'degenerate_median_bone_count':int(degenerate.sum()),
        'bone20_warning':max(bone_runs20)>=CONFIG['bone_min_consecutive_frames'],
        'slide04_warning':max(fast_slide)>=CONFIG['slide_min_consecutive_intervals'],
        'max_joint_step_m':float(displacement.max()),'p95_joint_step_m':float(np.percentile(displacement,95)),
        'airborne_longest_frames':air,'airborne_sample_span_s':max(0,air-1)/CONFIG['fps']
    }

def self_checks():
    # A sustained event cannot be fabricated by alternating left/right feet.
    assert per_entity_runs(np.array([[1,0],[0,1],[1,0],[0,1]],dtype=bool))==[1,1]
    alternating_bones=np.zeros((6,len(EDGES)),dtype=bool)
    alternating_bones[::2,0]=True;alternating_bones[1::2,1]=True
    assert max(per_entity_runs(alternating_bones))==1 and longest(alternating_bones.any(axis=1))==6
    assert longest([False,True,True,False,True])==2
    base=np.zeros((10,22,3));base[:,:,1]=1
    for j in range(22):base[:,j,0]=j*.02
    base[:,[10,11],1]=0
    steady=metrics(base);assert not steady['geometry_warning_any']
    slide=base.copy();slide[:,:,0]+=np.arange(10)[:,None]*.02
    slid=metrics(slide);assert slid['near_ground_slide_proxy'] and not slid['bone_length_variation_proxy']
    air=base.copy();air[:,:,1]+=np.r_[0,np.full(8,.3),0][:,None]
    assert metrics(air)['airborne_descriptive'] and not metrics(air)['foot_below_nominal_ground']
    below=base.copy();below[2:5,10,1]=-.04
    assert metrics(below)['foot_below_nominal_ground']
    return {'alternating_feet_not_pooled':True,'alternating_bones_not_pooled':True,'static_rigid_skeleton':True,'rigid_translation_slide':True,'airborne_not_physics_failure':True,'three_frame_below_ground':True}

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((SOURCE/'source_manifest.json').read_text(encoding='utf-8'))
    rows=[];tasks=[]
    for entry in manifest:
        file=SOURCE/entry['file'];assert sha(file)==entry['sha256']
        d=np.load(file,allow_pickle=True).item();motions=d['motion']
        assert motions.shape==tuple(entry['motion_shape'])
        assert len(d['lengths'])==len(motions)
        task_rows=[]
        for i,m in enumerate(motions):
            n=int(d['lengths'][i]);valid=(2<=n<=m.shape[-1] and np.isfinite(m[:,:,:n]).all())
            row={'task':entry['task'],'sample_id':i,'source_file':str(file),'source_sha256':entry['sha256'],
                 'frames':n,'fps':20,'nominal_video_duration_seconds':n/20,'sample_time_span_seconds':(n-1)/20,
                 'text':str(d['text'][i]),'structure_valid':bool(valid),'evaluation_type':'self_supervised_proxy'}
            if valid:row.update(metrics(m[:,:,:n].transpose(2,0,1)))
            task_rows.append(row);rows.append(row)
        keys=['foot_below_nominal_ground','any_foot_y_below_zero','near_ground_slide_proxy','bone_length_variation_proxy','large_joint_step_descriptive','airborne_descriptive','geometry_warning_any','bone20_warning','slide04_warning']
        counts={k:sum(bool(r.get(k,False)) for r in task_rows) for k in keys}
        tasks.append({'task':entry['task'],'count':len(task_rows),'structural_invalid_count':sum(not r['structure_valid'] for r in task_rows),
                      'counts_overlapping':counts,'source_sha256':entry['sha256'],'source_url':entry['url'],
                      'max_joint_step_m':max(r.get('max_joint_step_m',0) for r in task_rows),
                      'min_foot_y_m':min(r.get('foot_min_y_m',0) for r in task_rows),
                      'physics_failed_count':None,'physics_validity_certified_count':None})
        assert sha(file)==entry['sha256']
    assert len(rows)==342 and [t['count'] for t in tasks]==[100,100,13,20,100,9]
    summary={'status':'COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'scope':'All six released official human skeleton arrays; not historical own-generated Kick batches; not G1 filtering rerun.',
             'sequences':342,'tasks':tasks,'thresholds':CONFIG,'metric_self_checks':self_checks(),
             'structural_invalid_count':sum(t['structural_invalid_count'] for t in tasks),
             'physics_failed_count':None,'not_dynamically_certified_count':342,
             'semantic_pass_rate':None,'physical_pass_rate':None,'sample_rejection_performed':False,
             'warnings_are_not_failures':True,'rows_file':'per_sample.json','csv_file':'per_sample.csv',
             'script_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(SOURCE/'source_manifest.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'thresholds.json').write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'per_sample.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'per_sample.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    print(json.dumps({'tasks':tasks,'self_checks':summary['metric_self_checks']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:print(json.dumps(self_checks()))
    else:run()
