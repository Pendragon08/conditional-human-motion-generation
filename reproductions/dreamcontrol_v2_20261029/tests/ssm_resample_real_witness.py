"""Real SSM pose resampling witness using previous measured-marker receipts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from humanml_time_mapping import aligned_grid
from humanml_rotation_resample import apply_grid

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
IDS=BASE/'ssm_id_alignment_20261030/ids.jsonl'
RECEIPTS=BASE/'ssm_receipt_audit_20261030/rows.jsonl'
OUT=BASE/'ssm_resample_real_witness_20261030'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    entries=list(map(json.loads,IDS.open(encoding='utf8')))
    receipts={r['source_group']:r for r in map(json.loads,RECEIPTS.open(encoding='utf8'))}
    targets=('prior_marker_aligned_near20_crops_in_bounds','prior_marker_aligned_non20_requires_reprocessing')
    selected=[next(e for e in entries if e['status']==s and not e['mirrored']) for s in targets]
    results=[]
    for item in selected:
        old=receipts[item['source_path']]; recpath=Path(old['receipt']); rec=json.loads(recpath.read_text())
        ref=Path(rec['reference_file']); src=Path(rec['smplx_file'])
        if sha(recpath)!=old['receipt_sha256'] or sha(ref)!=rec['reference_sha256'] or sha(src)!=rec['smplx_sha256']:
            raise ValueError('receipt or source hash changed')
        grid=aligned_grid(item['reference_raw_frame_count'],item['reference_fps'],item['start'],item['stop'])
        with np.load(src,allow_pickle=False) as data:
            trans=np.asarray(data['trans'],dtype=np.float64)
            root=np.asarray(data['root_orient'],dtype=np.float64)
            body=np.asarray(data['pose_body'],dtype=np.float64)
        if len(trans)!=item['reference_raw_frame_count'] or root.shape!=(len(trans),3) or body.shape!=(len(trans),63):
            raise ValueError('source frame/pose dimensions differ')
        pos,rot=apply_grid(trans,np.concatenate([root,body],axis=1),grid)
        source=np.asarray(grid['source_indices']); target=np.array([w['time'] for w in grid['weights']])
        source_t=(source-source[0])/item['reference_fps']
        expected_pos=np.column_stack([np.interp(target,source_t,trans[source,j]) for j in range(3)])
        original=np.concatenate([root,body],axis=1); max_angle=0.0
        for joint in range(22):
            desired=Slerp(source_t,Rotation.from_rotvec(original[source,joint*3:joint*3+3]))(target)
            got=Rotation.from_rotvec(rot[:,joint*3:joint*3+3])
            max_angle=max(max_angle,float(np.max((desired.inv()*got).magnitude())))
        position_error=float(np.max(np.abs(pos-expected_pos)))
        if position_error>1e-8 or max_angle>1e-6:raise ValueError('independent oracle differs')
        results.append({'motion_id':item['motion_id'],'source':item['source_path'],
            'reference_fps':item['reference_fps'],'target_header_fps':item['smplx_header_fps'],
            'legacy_sample_fps':grid['actual_reference_sample_fps'],'old_samples':len(source),
            'output_fps':grid['output_fps'],'new_samples':len(target),'duration_s':grid['source_duration_s'],
            'translation_oracle_max_abs_m':position_error,'rotation_oracle_max_angle_rad':max_angle,
            'source_sha256':sha(src),'reference_sha256':sha(ref)})
    OUT.mkdir(mode=0o700)
    report={'samples':results,'limitations':['Two non-mirrored source motions; numerical interpolation witness only.',
        'Reference frame identity depends on prior measured-marker audit; marker comparisons not rerun.',
        'No caption interval alignment, anti-aliasing validation, reconstruction ground truth, G1 IK, contact, dynamics or training.']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
