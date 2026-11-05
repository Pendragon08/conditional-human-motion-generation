"""Read small source metadata/NPY headers for only184 uncovered named raw candidates.

No full pose decompression, source edit, retargeting or source-identity/physical
clock acceptance. HumanML nominal indexing follows the pinned public notebook.
"""
from collections import Counter
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
W=R/'reactive29_pairing_work_20261104'
RAW_ROOT=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass').resolve()


def inspect(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(RAW_ROOT):raise ValueError('candidate outside named read-only raw root')
    headers={};scalars={}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if not info.filename.endswith('.npy'):continue
            key=Path(info.filename).stem
            with z.open(info) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError('unsupported NPY header version')
            headers[key]={'shape':list(shape),'dtype':str(dtype),'member_CRC32':info.CRC,'uncompressed_bytes':info.file_size}
            if key in ['mocap_frame_rate','mocap_framerate','fps','gender','surface_model_type'] and info.file_size<=4096 and not dtype.hasobject:
                value=np.load(io.BytesIO(z.read(info.filename)),allow_pickle=False).tolist()
                if isinstance(value,bytes):value=value.decode('utf8')
                scalars[key]=value
    rates=[float(scalars[k]) for k in ['mocap_frame_rate','mocap_framerate','fps'] if k in scalars and np.isscalar(scalars[k])]
    rate=rates[0] if rates and np.isfinite(rates).all() and min(rates)>0 and max(rates)-min(rates)<1e-8 else None
    poses=headers.get('poses',{}).get('shape',[]);trans=headers.get('trans',{}).get('shape',[])
    return {'path':str(path),'file_bytes':path.stat().st_size,'fields':headers,'small_declared_metadata':scalars,
            'declared_fps_unambiguous':rate,'frames':poses[0] if len(poses)==2 else None,
            'pose_columns':poses[1] if len(poses)==2 else None,'pose_trans_shape_consistent':len(poses)==2 and trans==[poses[0],3],
            'model_kind_by_pose_shape':'SMPL-X-compatible165' if len(poses)==2 and poses[1]==165 else 'SMPL-H-compatible156' if len(poses)==2 and poses[1]==156 else 'requires_parameter_mapping'}


def main():
    out=W/'uncovered_source_header_probe_v1.json';assert not out.exists()
    triage=json.loads((W/'uncovered_source_triage.json').read_text())
    candidates=[x for x in triage['records'] if x['named_candidate_exists_on_server']]
    cache={};errors={};began=time.monotonic();rows=[]
    for path in sorted({x['named_candidate_path'] for x in candidates}):
        try:cache[path]=inspect(path)
        except Exception as e:errors[path]=repr(e)
    for row in candidates:
        path=row['named_candidate_path'];source=cache.get(path)
        result={'motion_id':row['motion_id'],'indexed_dataset':row['indexed_dataset'],
            'indexed_source_path':row['indexed_source_path'],'candidate_path':path,
            'source_name_match_only':True,'source_identity_verified':False,'actual_clock_verified':False}
        if source is None:result.update(status='header_probe_error',error=errors[path]);rows.append(result);continue
        fps=source['declared_fps_unambiguous'];dataset=row['indexed_dataset']
        trim=60 if dataset in ['Eyes_Japan_Dataset','MPI_HDM05'] else 20 if dataset in ['TotalCapture','MPI_Limits'] else 10 if dataset=='Transitions_mocap' else 0
        result.update(declared_fps=fps,pose_columns=source['pose_columns'],model_kind=source['model_kind_by_pose_shape'],
            humanml_nominal20_pretrim_frames=trim)
        start=int(row['index_window_nominal20Hz']['start_frame']);end=int(row['index_window_nominal20Hz']['end_frame'])
        if fps is not None and fps>=20 and source['frames'] is not None:
            stride=int(fps/20);available=(source['frames']+stride-1)//stride-trim
            bounded=0<=start<end<=available
            result.update(humanml_integer_decimation_stride=stride,humanml_nominal_available_frames=available,
                index_start20=start,index_end20=end,index_window_in_bounds=bounded,
                declared_elapsed_per_nominal20_step_s=stride/fps,
                declared_grid_is_exact20Hz=abs(stride/fps-.05)<1e-8,
                raw_frame_start_if_same_recording=(trim+start)*stride,
                raw_last_selected_frame_if_same_recording=(trim+end-1)*stride,
                status='in_bounds_under_declared_metadata_and_index_association' if bounded else 'index_outside_declared_metadata_range')
        else:result['status']='requires_fps_or_pose_shape_resolution'
        rows.append(result)
    result={'scope':'read-only small declared metadata/header probe of missing nominal IDs, not retargeting or identity/time acceptance',
        'humanml_notebook_commit':'9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23',
        'humanml_processing_rules':{'ex_fps':20,'stride':'int(source_declared_fps/20)',
            'pretrim_nominal20_frames':{'Eyes_Japan_Dataset':60,'MPI_HDM05':60,'TotalCapture':20,'MPI_Limits':20,'Transitions_mocap':10},
            'index_interval':'start inclusive/end exclusive after pretrim'},
        'unique_sources':len(cache),'source_errors':errors,'nominal_ids':len(rows),
        'status_counts':dict(Counter(x['status'] for x in rows)),
        'declared_fps_counts':dict(Counter(str(x['declared_fps_unambiguous']) for x in cache.values())),
        'model_kind_counts':dict(Counter(x['model_kind_by_pose_shape'] for x in cache.values())),
        'sources':cache,'records':rows,'elapsed_seconds':time.monotonic()-began,'new_retargeting_or_training_started':False}
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['unique_sources','nominal_ids','status_counts','declared_fps_counts','model_kind_counts','elapsed_seconds']}))


if __name__=='__main__':main()
