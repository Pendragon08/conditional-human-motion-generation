"""Verify all adapted schema records and their nominal source/time bindings.

This audit does not review semantic content, true capture timing, provenance
identity, source rights or physical feasibility and never authorizes training.
"""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=ROOT/'omomo_training_schema_preview_20261101'
PREVIEW=ROOT/'omomo_mixed_encoding_preview_20261101'
GROUP=ROOT/'omomo_group_split_20261101/candidates.jsonl'
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    report=json.loads((INPUT/'report.json').read_text()); feature_audit=json.loads((INPUT/'features_audit.json').read_text())
    if (sha(INPUT/'manifest.json')!=report['manifest_sha256'] or sha(INPUT/'state_manifest.json')!=report['state_manifest_sha256'] or
        sha(PREVIEW/'manifest.json')!=report['source_manifest_sha256'] or sha(PREVIEW/'stats.npz')!=report['source_stats_sha256'] or
        sha(GROUP)!=report['group_manifest_sha256'] or feature_audit['manifest_sha256']!=report['manifest_sha256'] or
        feature_audit['stats_sha256']!=report['source_stats_sha256'] or not feature_audit['technical_checks_passed'] or
        feature_audit['training_ready'] is not False):raise ValueError('input audit/hash mismatch')
    orig=json.loads((PREVIEW/'manifest.json').read_text())['records']; out=json.loads((INPUT/'manifest.json').read_text())['records']
    state=json.loads((INPUT/'state_manifest.json').read_text())['records']
    groups={r['motion_member']:r for r in map(json.loads,GROUP.read_text().splitlines())}
    if len(out)!=1717 or len(state)!=len(out) or len(orig)!=len(out):raise ValueError('bad output coverage')
    sources={r['sample_id']:r for r in orig}; states={r['sample_id']:r for r in state}
    if len(sources)!=len(orig) or len(states)!=len(state) or {x['sample_id'] for x in out}!=set(sources) or len({x['sample_id'] for x in out})!=len(out):
        raise ValueError('duplicate or missing sample')
    counts={'train':0,'val':0,'test':0}
    for r in out:
        src=sources[r['sample_id']];st=states[r['sample_id']];g=groups[src['source_member']]
        subject='OMOMO_nominal_subject:'+src['nominal_subject'];action='OMOMO_nominal_action:'+src['nominal_action_key']
        expected={'group_id':subject,'source_id':action,'split':src['auxiliary_split'],'caption':src['caption'],
                  'caption_accepted':False,'caption_origin':'source_text',
                  'caption_source':'OMOMO author archive: '+g['annotation_member'],
                  'annotation_sha256':g['annotation_sha256'],'fps':20.,'training_ready':False,
                  'feature_path':src['feature_path'],'feature_sha256':src['feature_sha256'],
                  'source_member':src['source_member'],'source_sha256':src['source_sha256'],
                  'encoding_variant':src['encoding_variant'],
                  'provenance_keys':{'nominal_subject':[subject],'nominal_action':[action],'source_member_sha256':[src['source_sha256']]}}
        if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('adapted record mismatch')
        if st!={'sample_id':r['sample_id'],'feature_sha256':src['feature_sha256'],'state_path':src['state_path'],
                'state_sha256':src['state_sha256'],'foot_velocity_squared_threshold':.002,'encoding_variant':src['encoding_variant']}:
            raise ValueError('state binding mismatch')
        p=(ROOT/st['state_path']).resolve(strict=True)
        if not p.is_relative_to(ROOT.resolve()) or sha(p)!=st['state_sha256']:raise ValueError('state bytes changed')
        with np.load(p,allow_pickle=False) as f:t=np.asarray(f['sample_times'],dtype=float)
        if len(t)!=src['feature_frames']+1 or not np.all(np.abs(np.diff(t)-.05)<1e-8) or r['nominal_frame_range_s']!=[float(t[0]),float(t[-1])]:
            raise ValueError('state timing mismatch')
        counts[r['split']]+=1
    if counts!=report['counts_by_split']:raise ValueError('split counts mismatch')
    result={'input_report_sha256':sha(INPUT/'report.json'),'feature_audit_sha256':sha(INPUT/'features_audit.json'),
            'manifest_sha256':report['manifest_sha256'],'state_manifest_sha256':report['state_manifest_sha256'],
            'source_manifest_sha256':report['source_manifest_sha256'],'script_sha256':sha(Path(__file__)),
            'records_checked':len(out),'counts_by_split':counts,'captions_accepted':0,'training_ready':False,
            'limitations':['Checks schema and pinned input consistency, not truth of input labels or clocks.',
             'Feature-audit spatial statistics share the existing decoder; no source license or physical acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))

if __name__=='__main__':main()
