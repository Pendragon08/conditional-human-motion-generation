"""Link audited Omni outputs to an unaccepted split and train-only statistics.

Nothing is copied from source outputs, and this is not a training manifest.
Captions, identity, source license and physical quality are not accepted.
"""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
GROUP=ROOT/'omomo_group_split_20261101'
HERE=Path(__file__).resolve().parent
FOLDERS=('omni_gpu_batch24_20261101','omni_gpu_remaining_20261101','omni_long15_gpu_20261101','omni_variants_smoke_20261101','omni_variants_snapshot_20261101')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    group_report=json.loads((GROUP/'report.json').read_text());group_audit=json.loads((GROUP/'audit.json').read_text())
    if sha(GROUP/'candidates.jsonl')!=group_report['candidate_manifest_sha256'] or group_audit['manifest_sha256']!=group_report['candidate_manifest_sha256'] or group_audit['report_sha256']!=sha(GROUP/'report.json'):
        raise ValueError('group evidence mismatch')
    groups={r['motion_member']:r for r in map(json.loads,(GROUP/'candidates.jsonl').read_text().splitlines())}
    records=[];folder_pins={};members=set();feature_arrays=[];spatial_arrays=[];counts=collections.Counter();frame_counts=collections.Counter()
    for folder in FOLDERS:
        path=ROOT/folder;report=json.loads((path/'report.json').read_text());audit=json.loads((path/'audit_v2.json').read_text())
        if sha(path/'report.json')!=audit['report_sha256'] or sha(path/'records.jsonl')!=report['records_sha256'] or sha(path/'audit_rows_v2.jsonl')!=audit['audit_rows_sha256']:
            raise ValueError('batch evidence mismatch')
        raw={r['sample_id']:r for r in map(json.loads,(path/'records.jsonl').read_text().splitlines())}
        checks=[json.loads(line) for line in (path/'audit_rows_v2.jsonl').read_text().splitlines()]
        if len(raw)!=len(checks) or sum(x['status']=='technical_checks_passed' for x in checks)!=audit['technical_passed']:
            raise ValueError('batch audit coverage mismatch')
        folder_pins[folder]={'report_sha256':sha(path/'report.json'),'audit_sha256':sha(path/'audit_v2.json')}
        for check in checks:
            if check['status']!='technical_checks_passed':continue
            r=raw[check['sample_id']];member=r['source_member'];g=groups[member]
            if member in members or r['source_sha256']!=g['motion_member_sha256'] or r['annotation_sha256']!=g['annotation_sha256'] or r['caption']!=g['caption']:
                raise ValueError('source, text or duplicate mismatch')
            members.add(member)
            artifact_root=ROOT/'omni_variants_gpu_20261101' if folder=='omni_variants_snapshot_20261101' else path
            feature=artifact_root/r['feature'];state=artifact_root/r['state20']
            if sha(feature)!=r['feature_sha256'] or sha(state)!=r['state20_sha256']:
                raise ValueError('processed output changed')
            with np.load(feature,allow_pickle=False) as f:x=np.asarray(f['features'],dtype=np.float32)
            if x.ndim!=2 or x.shape[1]!=263 or not 2<=len(x)<=199 or not np.isfinite(x).all() or len(x)!=r['feature_frames']:
                raise ValueError('invalid processed feature')
            split=g['auxiliary_split'];counts[split]+=1;frame_counts[split]+=len(x)
            if split=='train':
                feature_arrays.append(x.astype(np.float64))
                with torch.no_grad():points=recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66)
                if not np.isfinite(points).all():raise ValueError('nonfinite decoded point')
                spatial_arrays.append(points.astype(np.float64))
            records.append({'sample_id':r['sample_id'],'source_member':member,'source_sha256':r['source_sha256'],
                            'nominal_subject':g['nominal_subject'],'nominal_action_key':g['nominal_action_key'],
                            'auxiliary_split':split,'caption':r['caption'],'caption_status':'unreviewed_filename_match',
                            'feature_path':str(feature.relative_to(ROOT)),'feature_sha256':r['feature_sha256'],
                            'state_path':str(state.relative_to(ROOT)),'state_sha256':r['state20_sha256'],
                            'feature_frames':len(x),'processing_evidence_folder':folder,'training_ready':False})
    if set(counts)!={'train','val','test'} or not feature_arrays:raise ValueError('missing auxiliary split')
    x=np.concatenate(feature_arrays);p=np.concatenate(spatial_arrays);min_std=.001
    with (HERE/'stats.npz').open('xb') as f:
        np.savez_compressed(f,feature_mean=x.mean(0),feature_std=np.maximum(x.std(0),min_std),
                            spatial_mean=p.mean(0),spatial_std=np.maximum(p.std(0),min_std))
    with (HERE/'manifest.json').open('x') as f:json.dump({'records':records,'training_ready':False},f,indent=2,allow_nan=False);f.write('\n')
    result={'group_manifest_sha256':group_report['candidate_manifest_sha256'],'folder_pins':folder_pins,
            'manifest_sha256':sha(HERE/'manifest.json'),'stats_sha256':sha(HERE/'stats.npz'),'script_sha256':sha(Path(__file__)),
            'counts_by_split':dict(counts),'feature_frames_by_split':dict(frame_counts),'min_std':min_std,'training_ready':False,
            'limitations':['Only technical-pass outputs from completed originals, six smoke variants and audited queue snapshot; not full batch.',
                           'Nominal subject split is non-author and not proof of real identity independence.',
                           'Caption semantics, source license, actual timing and physical quality are not accepted.',
                           'Decoded spatial stats share the experimental HumanML decoder.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
