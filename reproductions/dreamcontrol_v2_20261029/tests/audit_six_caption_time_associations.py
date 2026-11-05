"""Recompute time containment for six saved full-clip label associations."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=json.loads((HERE/'report.json').read_text())
    manifest=ROOT/'six_candidate_technical_preview_20261031/experimental_manifest_not_for_training.json'
    assert sha(manifest)==report['manifest_sha256']
    records={x['sample_id']:x for x in json.loads(manifest.read_text())['records']}
    counts=[]
    for case in report['cases']:
        rec=records[case['sample_id']];p=manifest.parent/rec['feature_path'];assert sha(p)==rec['feature_sha256']
        with np.load(p,allow_pickle=False) as f:
            times=f['source_times'].copy() if case['sample_id'].startswith('BML_') else f['time_grid'].copy()
            assert len(times)==len(f['features'])+1
        lo,hi=case['clip_first_last_s'];association=case['association']
        assert case['raw_tags']==['0.0','0.0'] and association['tag_interpretation']=='whole_clip_convention'
        expected=[i for i in range(len(times)-1) if times[i]>=lo-1e-9 and times[i+1]<=hi+1e-9]
        assert association['matching_feature_intervals']==expected
        assert association['feature_covers_full_annotation_sample_extent'] is False
        assert association['caption_accepted'] is False
        counts.append({'sample_id':case['sample_id'],'contained_intervals':len(expected),
                       'head_crop_s':float(times[0]-lo),'tail_crop_s':float(hi-times[-1])})
    assert len(counts)==6
    result={'report_sha256':sha(HERE/'report.json'),'cases':counts,'training_ready':False,
            'limitations':['Uses reported clip bounds, not independently measured source time or semantic matching.']}
    with (HERE/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
