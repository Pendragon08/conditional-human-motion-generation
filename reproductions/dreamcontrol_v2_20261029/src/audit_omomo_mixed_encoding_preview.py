"""Independently recompute the processed-preview index and train statistics."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
GROUP=ROOT/'omomo_group_split_20261101'; BASE=ROOT/'omomo_processed_full_preview_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def checked(relative, wanted):
    p=(ROOT/relative).resolve(strict=True)
    if Path(relative).is_absolute() or not p.is_relative_to(ROOT.resolve()) or sha(p)!=wanted:
        raise ValueError('path or hash mismatch')
    return p


def main():
    torch.set_num_threads(1)
    report=json.loads((HERE/'report.json').read_text());manifest=HERE/'manifest.json';stats=HERE/'stats.npz'
    if sha(manifest)!=report['manifest_sha256'] or sha(stats)!=report['stats_sha256'] or sha(GROUP/'candidates.jsonl')!=report['group_manifest_sha256']:
        raise ValueError('preview input hash mismatch')
    if sha(BASE/'manifest.json')!=report['base_manifest_sha256'] or sha(BASE/'audit.json')!=report['base_audit_sha256']:
        raise ValueError('base evidence changed')
    expected_repairs={}
    for folder,pins in report['repair_pins'].items():
        rp,ap=ROOT/folder,ROOT/pins['audit_folder']
        if sha(rp/'report.json')!=pins['repair_report_sha256'] or sha(ap/'report.json')!=pins['audit_report_sha256']:
            raise ValueError('repair report changed')
        repair=json.loads((rp/'report.json').read_text());audit=json.loads((ap/'report.json').read_text())
        if sha(rp/'records.jsonl')!=repair['rows_sha256'] or sha(ap/'records.jsonl')!=audit['rows_sha256'] or audit['failed']:
            raise ValueError('repair records changed')
        good={r['sample_id'] for r in map(json.loads,(ap/'records.jsonl').read_text().splitlines()) if r['status']=='checked'}
        for r in map(json.loads,(rp/'records.jsonl').read_text().splitlines()):
            if r['status']=='technical_preview':
                if r['sample_id'] not in good or r['sample_id'] in expected_repairs:raise ValueError('repair audit coverage')
                expected_repairs[r['sample_id']]=(folder,r)
        if len(good)!=repair['technical_previews']:raise ValueError('repair count mismatch')
    candidates={r['motion_member']:r for r in map(json.loads,(GROUP/'candidates.jsonl').read_text().splitlines())}
    rows=json.loads(manifest.read_text())['records'];counts=collections.Counter();frames=collections.Counter()
    base={r['sample_id']:r for r in json.loads((BASE/'manifest.json').read_text())['records']}
    if len(rows)!=len(base)+len(expected_repairs):raise ValueError('mixed preview count mismatch')
    seen=set();subjects={};actions={};feature_content={};encodings=collections.Counter()
    fsum=np.zeros(263);fsq=np.zeros(263);psum=np.zeros(66);psq=np.zeros(66);n=0
    for row in rows:
        member=row['source_member'];g=candidates[member];split=row['auxiliary_split']
        if member in seen or g['motion_member_sha256']!=row['source_sha256'] or g['caption']!=row['caption'] or g['auxiliary_split']!=split:
            raise ValueError('source binding mismatch')
        seen.add(member)
        sid=row['sample_id'];variant=row.get('encoding_variant');encodings[variant]+=1
        if sid in base:
            if variant!='legacy' or {k:v for k,v in row.items() if k!='encoding_variant'}!=base[sid]:raise ValueError('base record changed')
        else:
            if sid not in expected_repairs or variant!='sign_continuous_v1':raise ValueError('unknown repair')
            folder,rep=expected_repairs[sid]
            if row['feature_path']!=folder+'/'+rep['feature'] or row['feature_sha256']!=rep['feature_sha256'] or row['state_sha256']!=rep['parent_state_sha256'] or row['parent_feature_sha256']!=rep['parent_feature_sha256']:
                raise ValueError('repair binding mismatch')
        if subjects.setdefault(row['nominal_subject'],split)!=split or actions.setdefault(row['nominal_action_key'],split)!=split:
            raise ValueError('nominal group crosses split')
        if row['caption_status']!='unreviewed_filename_match' or row['training_ready'] is not False:
            raise ValueError('unexpected acceptance status')
        p=checked(row['feature_path'],row['feature_sha256']);checked(row['state_path'],row['state_sha256'])
        with np.load(p,allow_pickle=False) as f:x=np.asarray(f['features'],dtype=np.float32)
        if x.shape!=(row['feature_frames'],263) or not np.isfinite(x).all():raise ValueError('invalid feature array')
        digest=hashlib.sha256(np.asarray(x.shape,dtype='<i8').tobytes()+np.ascontiguousarray(x,dtype='<f4').tobytes()).hexdigest()
        if feature_content.setdefault(digest,split)!=split:raise ValueError('exact feature array crosses split')
        counts[split]+=1;frames[split]+=len(x)
        if split=='train':
            f64=x.astype(np.float64)
            with torch.no_grad():points=recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64)
            if not np.isfinite(points).all():raise ValueError('invalid decoded points')
            fsum+=f64.sum(0);fsq+=(f64*f64).sum(0);psum+=points.sum(0);psq+=(points*points).sum(0);n+=len(x)
    if dict(counts)!=report['counts_by_split'] or dict(frames)!=report['feature_frames_by_split'] or n!=frames['train']:
        raise ValueError('split count mismatch')
    if dict(encodings)!=report['encoding_counts']:raise ValueError('encoding counts mismatch')
    floor=report['min_std'];fm=fsum/n;pm=psum/n
    expected={'feature_mean':fm,'feature_std':np.maximum(np.sqrt(np.maximum(fsq/n-fm*fm,0)),floor),
              'spatial_mean':pm,'spatial_std':np.maximum(np.sqrt(np.maximum(psq/n-pm*pm,0)),floor)}
    with np.load(stats,allow_pickle=False) as data:
        errors={k:float(np.max(np.abs(data[k]-v))) for k,v in expected.items()}
    if max(errors.values())>1e-6:raise ValueError('training statistic mismatch')
    if sha(manifest)!=report['manifest_sha256'] or sha(stats)!=report['stats_sha256']:
        raise RuntimeError('preview changed')
    result={'preview_report_sha256':sha(HERE/'report.json'),'manifest_sha256':report['manifest_sha256'],
            'stats_sha256':report['stats_sha256'],'script_sha256':sha(Path(__file__)),'records':len(rows),
            'counts_by_split':dict(counts),'feature_frames_by_split':dict(frames),'stat_max_abs_errors':errors,
            'exact_feature_arrays_cross_split':0,'training_ready':False,
            'limitations':['Same experimental decoder used for spatial stats; no independent FK or encoder.',
                           'Exact hashes cannot exclude overlap after encoding or source aliases.',
                           'Caption, source rights, time and physics remain unaccepted; this is not a training manifest.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
