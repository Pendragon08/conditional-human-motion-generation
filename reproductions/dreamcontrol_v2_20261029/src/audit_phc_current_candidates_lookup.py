"""Check candidate literals without deserializing, independently of parser."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report=json.loads((HERE/'report.json').read_text())
    source=ROOT/'phc_filter_parser_v2_20261031/source.pkl'
    manifest=ROOT/'four_semantic_subset_20261101/manifest.json'
    raw=source.read_bytes()
    assert sha(source)==report['source_sha256'] and sha(manifest)==report['manifest_sha256']
    original={r['sample_id']:r['source_id'] for r in json.loads(manifest.read_text())['records']}
    counts={}
    for row in report['cases']:
        assert original[row['sample_id']]==row['source_id']
        assert row['status']=='not_listed_under_these_keys' and row['matches']==[]
        counts[row['sample_id']]={key:raw.count(key.encode('utf8')) for key in row['conditional_keys']}
        assert all(n==0 for n in counts[row['sample_id']].values())
    controls=['0-BioMotionLab_NTroje_rub020_0002_treadmill_slow_poses','0-KIT_3_kneel_down_with_stool_right_hand08_poses']
    positive={key:raw.count(key.encode('utf8')) for key in controls}
    assert all(n==1 for n in positive.values())
    out={'report_sha256':sha(HERE/'report.json'),'source_sha256':sha(source),'candidate_literal_counts':counts,
         'positive_controls':positive,'training_ready':False,
         'limitations':'Raw bytes are not independent full parser validation, and absent names do not prove feasibility or matching source identity.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
