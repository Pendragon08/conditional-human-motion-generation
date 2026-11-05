"""Adapt the audited OMOMO technical preview to the loader's schema.

This produces an isolated, explicitly unaccepted preview and state manifest.
It does not start training, validate caption meaning or certify licensing,
capture time, source identities or physical quality. Group keys are nominal.
"""
import hashlib,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'omomo_mixed_encoding_preview_20261101'
GROUP=ROOT/'omomo_group_split_20261101'
HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def checked_path(relative,expected):
    if not isinstance(relative,str) or Path(relative).is_absolute():raise ValueError('relative path required')
    p=(ROOT/relative).resolve(strict=True)
    if not p.is_relative_to(ROOT.resolve()) or digest(p)!=expected:raise ValueError('file path or hash mismatch')
    return p

def main():
    report=json.loads((PREVIEW/'report.json').read_text()); audit=json.loads((PREVIEW/'audit.json').read_text())
    manifest_sha=digest(PREVIEW/'manifest.json'); stats_sha=digest(PREVIEW/'stats.npz')
    if report['manifest_sha256']!=manifest_sha or report['stats_sha256']!=stats_sha or audit['manifest_sha256']!=manifest_sha:
        raise ValueError('preview manifest, statistics or audit mismatch')
    gr=json.loads((GROUP/'report.json').read_text()); candidate_sha=digest(GROUP/'candidates.jsonl')
    if gr['candidate_manifest_sha256']!=candidate_sha:raise ValueError('group manifest mismatch')
    groups={x['motion_member']:x for x in map(json.loads,(GROUP/'candidates.jsonl').read_text().splitlines())}
    original=json.loads((PREVIEW/'manifest.json').read_text())['records']
    if len(groups)!=1826 or len(original)!=1717:raise ValueError('unexpected input coverage')
    records=[];states=[];seen=set();split_frames=Counter();split_counts=Counter()
    for i,r in enumerate(original):
        if i%25==0 and shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('disk stop line')
        sid=r['sample_id'];g=groups[r['source_member']]
        if sid in seen:raise ValueError('duplicate sample ID')
        seen.add(sid)
        for a,b in [('source_sha256','motion_member_sha256'),('nominal_subject','nominal_subject'),
                    ('nominal_action_key','nominal_action_key'),('auxiliary_split','auxiliary_split'),('caption','caption')]:
            if r[a]!=g[b]:raise ValueError('group binding mismatch')
        if r['caption_status']!='unreviewed_filename_match' or r['training_ready'] is not False:raise ValueError('unexpected acceptance')
        feature=checked_path(r['feature_path'],r['feature_sha256']);state=checked_path(r['state_path'],r['state_sha256'])
        with np.load(state,allow_pickle=False) as f:t=np.asarray(f['sample_times'],dtype=float)
        with np.load(feature,allow_pickle=False) as f:x=np.asarray(f['features'])
        if t.ndim!=1 or len(t)!=r['feature_frames']+1 or not np.isfinite(t).all() or not np.allclose(np.diff(t),.05,atol=1e-8,rtol=0):
            raise ValueError('nominal 20 Hz sample support mismatch')
        if x.shape!=(r['feature_frames'],263) or not np.isfinite(x).all():raise ValueError('feature shape mismatch')
        origin='OMOMO_nominal_subject:'+r['nominal_subject']; action='OMOMO_nominal_action:'+r['nominal_action_key']
        records.append({'sample_id':sid,'group_id':origin,'source_id':action,'split':r['auxiliary_split'],
            'provenance_keys':{'nominal_subject':[origin],'nominal_action':[action],
                               'source_member_sha256':[r['source_sha256']]},
            'caption':r['caption'],'caption_source':'OMOMO author archive: '+g['annotation_member'],
            'caption_origin':'source_text','caption_accepted':False,'annotation_sha256':g['annotation_sha256'],
            'feature_path':r['feature_path'],'feature_sha256':r['feature_sha256'],'fps':20.,
            'fps_evidence':'resampled grid; source capture clock not independently verified',
            'source_member':r['source_member'],'source_sha256':r['source_sha256'],
            'nominal_frame_range_s':[float(t[0]),float(t[-1])],
            'encoding_variant':r['encoding_variant'],'training_ready':False})
        states.append({'sample_id':sid,'feature_sha256':r['feature_sha256'],'state_path':r['state_path'],
            'state_sha256':r['state_sha256'],'foot_velocity_squared_threshold':.002,
            'encoding_variant':r['encoding_variant']})
        split_counts[r['auxiliary_split']]+=1;split_frames[r['auxiliary_split']]+=len(x)
        if digest(feature)!=r['feature_sha256'] or digest(state)!=r['state_sha256']:raise RuntimeError('input changed')
    if digest(PREVIEW/'manifest.json')!=manifest_sha or digest(PREVIEW/'stats.npz')!=stats_sha or digest(GROUP/'candidates.jsonl')!=candidate_sha:
        raise RuntimeError('input changed during adaptation')
    for name,value in [('manifest.json',{'records':records,'training_ready':False}),('state_manifest.json',{'records':states,'training_ready':False})]:
        with (HERE/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
    result={'source_manifest_sha256':manifest_sha,'source_stats_sha256':stats_sha,'group_manifest_sha256':candidate_sha,
            'manifest_sha256':digest(HERE/'manifest.json'),'state_manifest_sha256':digest(HERE/'state_manifest.json'),
            'script_sha256':digest(Path(__file__)),'counts_by_split':dict(split_counts),'frames_by_split':dict(split_frames),
            'captions_accepted':0,'training_ready':False,
            'limitations':['Schema preview, not a training manifest or author split.',
               'All caption semantics, original source rights, true time and physical quality remain unaccepted.',
               'Nominal subject and action keys do not prove actual identity or absence of aliasing.',
               'Foot displacement threshold is an existing encoding parameter, not contact evidence.']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))

if __name__=='__main__':main()
