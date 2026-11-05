"""Combine only compatible, completed fit audits without duplicate samples."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def combine(directories: list[Path]) -> dict:
    if not directories: raise ValueError('at least one completed audit is required')
    config_keys=('methods','steps','learning_rate','urdf29_sha256','urdf27_sha256',
                 'source_archives_sha256','dependency_sha256','torch_version','numpy_version')
    reference=None; records=[]; seen=set(); failures=[]
    for directory in directories:
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        complete=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
        if complete.get('status')!='finished' or complete.get('record_count')!=len(manifest['samples']):
            raise ValueError(f'incomplete or inconsistent audit: {directory}')
        fingerprint={k:manifest['config'][k] for k in config_keys}
        if reference is None: reference=fingerprint
        elif reference!=fingerprint: raise ValueError('audit configurations or dependencies differ')
        for sample in manifest['samples']:
            pair=(sample['archive'],sample['member'])
            if pair in seen: raise ValueError(f'duplicate source sample: {pair}')
            seen.add(pair)
            key=hashlib.sha256((pair[0]+'\0'+pair[1]).encode()).hexdigest()
            record=json.loads((directory/(key+'.json')).read_text(encoding='utf-8'))
            if (record.get('archive'),record.get('member'))!=pair:
                raise ValueError('record does not match manifest')
            if record.get('status')=='ok': records.append(record)
            elif record.get('status')=='error': failures.append({'archive':pair[0],'member':pair[1],'error':record.get('error')})
            else: raise ValueError('unknown record status')
    metrics=('mean_keypoint_error_m','max_keypoint_error_m','mean_foot_error_m','max_foot_error_m',
             'mean_wrist_error_m','contact_proxy_velocity_error_mps','max_root_translation_m','max_root_rotation_rad')
    summary={}
    for method in reference['methods']:
        summary[method]={}
        for key in metrics:
            vals=np.asarray([r['methods'][method][key] for r in records if r['methods'][method][key] is not None],dtype=np.float64)
            if vals.size and (not np.isfinite(vals).all() or (vals<0).any()):
                raise ValueError(f'invalid metric {method}/{key}')
            summary[method][key]={'n':int(vals.size),'mean':float(vals.mean()) if vals.size else None,
                                  'p95':float(np.quantile(vals,.95)) if vals.size else None,
                                  'max':float(vals.max()) if vals.size else None}
    return {'expected':len(seen),'successful':len(records),'failed':len(failures),'errors':failures,
            'sources':dict(Counter(x['archive'] for x in records)),'methods':summary,
            'note':'Sample groups are filename-derived, source independence is unproven; no collision or dynamics validation.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit_dirs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); result=combine(args.audit_dirs)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    print(json.dumps({'expected':result['expected'],'successful':result['successful'],'failed':result['failed']}))


if __name__=='__main__': main()
