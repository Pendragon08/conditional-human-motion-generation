"""Evaluate proposed time grid for SSM without generating or retiming poses."""
import collections
import json
from pathlib import Path
from humanml_time_mapping import aligned_grid

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=BASE/'ssm_id_alignment_20261030/ids.jsonl'
OUT=BASE/'ssm_20hz_timegrid_audit_20261030'

def main():
    rows=[]
    for row in map(json.loads,INPUT.open(encoding='utf8')):
        if not row['status'].startswith('prior_marker_aligned_'):
            raise ValueError('a source lacks marker evidence')
        grid=aligned_grid(row['reference_raw_frame_count'],row['reference_fps'],row['start'],row['stop'])
        if grid['reference_stride'] != row['recorded_stride']:
            raise ValueError('stride differs from previous proof')
        weights=grid['weights']
        if any(not (0 <= w['alpha'] <= 1 and w['left_frame'] <= w['right_frame']) for w in weights):
            raise ValueError('interpolation outside original bounds')
        rows.append({'motion_id':row['motion_id'],'source':row['source_path'],'prior_status':row['status'],
                     'legacy_fps':grid['actual_reference_sample_fps'],'legacy_sample_count':len(grid['source_indices']),
                     'duration_s':grid['source_duration_s'],'output_fps':20.0,'output_samples':len(weights),
                     'last_target_time':weights[-1]['time'],
                     'last_source_time':grid['source_duration_s']})
    OUT.mkdir(mode=0o700)
    with (OUT/'rows.jsonl').open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    grouped={}
    for status in set(r['prior_status'] for r in rows):
        subset=[r for r in rows if r['prior_status']==status]
        grouped[status]={'ids':len(subset),'unique_recordings':len({r['source'] for r in subset}),
                         'legacy_samples':sum(r['legacy_sample_count'] for r in subset),
                         'proposed_output_samples':sum(r['output_samples'] for r in subset)}
    summary={'status':'TIME_GRID_ONLY','ids':len(rows),'groups':grouped,
             'max_tail_remainder_s':max(r['last_source_time']-r['last_target_time'] for r in rows),
             'limitations':['Source timing is inherited from old marker evidence and rechecked source hashes.',
                            'No motion interpolation, annotation re-alignment, mirror, robot retargeting or physics was performed.',
                            'Applying these weights requires verified rotation interpolation and handling caption interval time bases.',
                            'The mapping is not a training-ready dataset.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
