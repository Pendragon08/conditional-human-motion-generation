"""Correlate HumanML ID crops with previously verified SSM marker receipts."""
import collections
import hashlib
import json
from pathlib import Path

BASE = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MAP = BASE/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
AUDIT = BASE/'ssm_receipt_audit_20261030/rows.jsonl'
OUT = BASE/'ssm_id_alignment_20261030'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    receipts = {r['source_group']:r for r in map(json.loads,AUDIT.open())}
    rows=[]
    for row in map(json.loads,MAP.open()):
        if '/SSM_synced/' not in row['source_path']:
            continue
        r=receipts.get(row['source_path'])
        item={'motion_id':row['motion_id'],'source_path':row['source_path'],
              'mirrored':row['mirrored'], 'reference_raw_frame_count':None}
        if r is None:
            item['status']='missing_receipt'
        elif not (r['ref_hash_matches'] and r['target_hash_matches']):
            item['status']='changed_or_missing_file'
        else:
            receipt=json.loads(Path(r['receipt']).read_text())
            n=int(receipt['timeline']['reference_frames'])
            start,end=int(row['index_start_20']),int(row['index_end_20'])
            stride=r['stride']; length=len(range(0,n,stride)); stop=length-1 if end==-1 else end
            in_bounds=0 <= start < stop <= length
            item.update(reference_raw_frame_count=n, reference_fps=r['reference_fps'],
                smplx_header_fps=r['target_header_fps'], recorded_stride=stride,
                recorded_effective_fps=r['effective_fps'], nominal_crop_in_bounds=in_bounds,
                start=start, stop=stop, source_receipt=r['receipt'],
                status=('prior_marker_aligned_near20_crops_in_bounds' if in_bounds and r['within_0_01_hz_of_20']
                        else ('prior_marker_aligned_non20_requires_reprocessing' if in_bounds else 'prior_crop_out_of_bounds')))
        rows.append(item)
    OUT.mkdir(mode=0o700)
    with (OUT/'ids.jsonl').open('x',encoding='utf8') as f:
        for x in rows:f.write(json.dumps(x)+'\n')
    summary={'ids':len(rows),'status_counts':dict(collections.Counter(x['status'] for x in rows)),
             'unique_recordings_by_status':{status:len({x['source_path'] for x in rows if x['status']==status})
                                             for status in set(x['status'] for x in rows)},
             'input_mapping_sha256':sha(MAP),'receipt_recheck_sha256':sha(AUDIT),'ids_sha256':sha(OUT/'ids.jsonl'),
             'limitations':['Prior marker evidence and current hashes used; markers not recomputed in this pass.',
                            'Near-20Hz and in-bounds crop alone do not prove reconstructed G1 motion quality or v2 compatibility.',
                            'Non-20Hz case needs verified, explicit temporal resampling before training.',
                            'Mirrored IDs require a verified motion transform; annotations and physical quality not revalidated here.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
