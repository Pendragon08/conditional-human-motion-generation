"""Select existing pairs with consistent export and sample clocks; never alter inputs.

This is a prerequisite, not a physical timestamp certificate or tracking test.
Do not automatically repair conflicting clocks by moving captions or stretching poses.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock_check(timeline, rtol=1e-5):
    reasons=[]
    if timeline.get('status') != 'FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS':
        reasons.append('unverified_cross_export_frame_identity')
    try:
        h,x,f = [float(timeline[k]) for k in ['reference_fps','smplx_header_fps','nominal_output_fps']]
        stride=timeline['reference_stride']
        if (not all(math.isfinite(v) and v>0 for v in [h,x,f])
                or isinstance(stride,bool) or not isinstance(stride,int) or stride<1):
            raise ValueError('Invalid clock values')
    except (KeyError,TypeError,ValueError):
        return {'eligible':False,'reasons':reasons+['invalid_clock_metadata']}
    close=lambda a,b: abs(a-b)<=rtol*max(abs(a),abs(b))
    if not close(h,x):reasons.append('conflicting_export_fps_headers')
    if not close(h/stride,f):reasons.append('reference_stride_clock_differs_from_feature_clock')
    if not close(x/stride,f):reasons.append('target_stride_clock_differs_from_feature_clock')
    return {'eligible':not reasons,'reasons':reasons,'reference_fps':h,'target_fps':x,
            'raw_frame_stride':stride,'reference_sample_fps':h/stride,
            'target_sample_fps':x/stride,'feature_fps':f,
            'relative_tolerance':rtol,'external_physical_clock_certified':False}


def save(path,obj):
    path.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))


def run(args):
    source=args.dataset.resolve(strict=True)
    if args.output.exists():raise FileExistsError('Previous results retained: '+str(args.output))
    if not 0<args.rtol<1e-3:raise ValueError('Only narrow numeric header tolerance is allowed')
    manifest_file=source/'manifest_source_disjoint.json'
    manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest['protocol']!='source_disjoint':raise ValueError('Expected recording-disjoint input')
    observed={str(manifest_file):digest(manifest_file)}
    receipts={}
    for f in sorted((source/'source_receipts').glob('*.json')):
        receipt=json.loads(f.read_text(encoding='utf-8'))
        if not receipt['success']:continue
        group=receipt['source_group']
        if group in receipts:raise ValueError('Duplicate source receipt')
        receipts[group]=clock_check(receipt['timeline'],args.rtol)
        observed[str(f)]=digest(f)
    selected=copy.deepcopy(manifest);selected['splits']={s:[] for s in manifest['splits']}
    excluded=[];train=[];included_groups=set();excluded_groups=set()
    for split,rows in manifest['splits'].items():
        for row in rows:
            group=row['source_group']
            if group not in receipts:raise ValueError('Missing source receipt: '+group)
            clock=receipts[group]
            entry=copy.deepcopy(row)
            for key,hashkey in [('features','sha256'),('native','native_sha256'),('codec_metadata',None)]:
                f=(source/entry[key]).resolve(strict=True)
                if not f.is_relative_to(source):raise ValueError('Input asset escaped dataset directory')
                actual=digest(f);observed[str(f)]=actual
                if hashkey and actual!=entry[hashkey]:raise ValueError('Input hash mismatch: '+entry[key])
                entry[key]=str(f)
            metadata=json.loads(Path(entry['codec_metadata']).read_text(encoding='utf-8'))
            if metadata['fps']!=clock.get('feature_fps'):raise ValueError('Codec and receipt FPS differ')
            with np.load(entry['native'],allow_pickle=False) as native:
                raw=native['reference_raw_frames']
                if len(raw)!=row['frames'] or not np.all(np.diff(raw)==clock['raw_frame_stride']):
                    raise ValueError('Actual recorded raw-frame strides differ')
            if clock['eligible']:
                entry['sample_clock_gate']=clock
                selected['splits'][split].append(entry);included_groups.add(group)
                if split=='train':train.append(np.load(entry['features'],allow_pickle=False))
            else:
                excluded_groups.add(group)
                excluded.append({'motion_id':row['motion_id'],'interval':row['interval'],
                                 'source_group':group,'split':split,'clock':clock})
    if not train:raise ValueError('No eligible paired training data; refusing an empty training manifest')
    sys.path.insert(0,str(args.phygile_code.resolve()/'reimplementation'))
    from features import BlockNormalizer
    normalizer=BlockNormalizer.fit(train)
    selected['sample_clock_gate']={'status':'NUMERIC_CLOCK_CONSISTENCY_ONLY','source_manifest_sha256':observed[str(manifest_file)],
       'rtol':args.rtol,'external_physical_clock_certified':False,'script_sha256':digest(__file__),
       'normalization':'Refit on retained training entries only','excluded_sequence_count':len(excluded)}
    selected['physical_tracking_validated']=False
    selected['feature_semantics']+='; only measured-marker frame-identical sources whose export and sample clocks agree within the declared tolerance'
    summary={'status':'CLOCK_GATE_COMPLETED','script_sha256':digest(__file__),
       'source_manifest_sha256':observed[str(manifest_file)],'input_sequences':sum(map(len,manifest['splits'].values())),
       'retained_sequences':sum(map(len,selected['splits'].values())),'excluded_sequences':len(excluded),
       'retained_sources':len(included_groups),'excluded_sources':len(excluded_groups),
       'retained_splits':{s:{'sequences':len(rows),'frames':sum(x['frames'] for x in rows),
          'sources':len({x['source_group'] for x in rows})} for s,rows in selected['splits'].items()},
       'exclusion_reasons':dict(Counter(v for e in excluded for v in e['clock']['reasons'])),
       'source_clocks':receipts,'rtol':args.rtol,'normalizer_training_frames':sum(map(len,train)),
       'motion_arrays_changed':False,'captions_shifted':False,'data_deleted':False,
       'physical_tracking_validated':False,'author_data_reproduced':False,'generator_updates':0}
    if any(digest(p)!=expected for p,expected in observed.items()):raise ValueError('Input changed during processing')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'manifest_clock_consistent.json',selected)
    save(args.output/'excluded_entries.json',excluded)
    np.savez(args.output/'normalization_train_only.npz',mean=normalizer.mean,std=normalizer.std)
    save(args.output/'input_integrity.json',{'before_after_unchanged':True,'files':observed})
    summary['output_manifest_sha256']=digest(args.output/'manifest_clock_consistent.json')
    summary['normalization_sha256']=digest(args.output/'normalization_train_only.npz')
    save(args.output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_clocks'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--phygile-code',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rtol',type=float,default=1e-5)
    run(p.parse_args())
