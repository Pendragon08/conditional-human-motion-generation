"""Validate downloaded numerical corpora with a NumPy-only restricted unpickler."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


class NumericPickle(pickle.Unpickler):
    def find_class(self, module, name):
        permitted = {'numpy': {'ndarray','dtype'},
            'numpy.core.multiarray': {'_reconstruct','scalar'},
            'numpy._core.multiarray': {'_reconstruct','scalar'},
            'numpy.core.numeric': {'_frombuffer'}, 'numpy._core.numeric': {'_frombuffer'}}
        if name not in permitted.get(module, set()):
            raise pickle.UnpicklingError('Unexpected pickle global: '+module+'.'+name)
        return super().find_class(module, name)


def check(root):
    with (root/'full_train.pkl').open('rb') as f: data = NumericPickle(f).load()
    names=data['name_list']; motions=data['data_dict']; splits={}
    if len(names)!=len(set(names)):raise ValueError('Duplicate corpus keys')
    for split in ('train','val','test'):
        allowed=set((root/(split+'.txt')).read_text().splitlines())
        if not allowed:raise ValueError('Empty official split')
        splits[split]={n for n in names if n in allowed or n.rsplit('__ep',1)[0] in allowed}
    intersections={a+'_'+b:len(splits[a]&splits[b]) for a in splits for b in splits if a<b}
    if any(intersections.values()):raise ValueError('Official key overlap across splits')
    frames=0; captions=0; quat_error=0.; sums=np.zeros(36);squares=np.zeros(36);train_frames=0
    length_mismatches=[]; hashes={s:set() for s in splits}
    membership={n:s for s,ns in splits.items() for n in ns}
    for name in names:
        item=motions[name];x=np.asarray(item['motion'])
        if x.ndim!=2 or x.shape[1]!=36 or not np.isfinite(x).all():raise ValueError('Invalid motion array')
        if len(x)!=item['length']:length_mismatches.append(name)
        frames+=len(x);captions+=len(item['text'])
        if not item['text'] or any(not t.get('caption','').strip() for t in item['text']):raise ValueError('Missing real caption')
        quat_error=max(quat_error,float(np.abs(np.linalg.norm(x[:,3:7],axis=1)-1).max()))
        split=membership.get(name)
        if split:hashes[split].add(hashlib.sha256(x.tobytes()).hexdigest())
        if split=='train':
            d=x.astype(float);sums+=d.sum(0);squares+=np.square(d).sum(0);train_frames+=len(x)
    mean=sums/train_frames;std=np.sqrt(np.maximum(squares/train_frames-mean**2,0))
    supplied_mean=np.load(root/'Mean.npy',allow_pickle=False);supplied_std=np.load(root/'Std.npy',allow_pickle=False)
    return {'dataset':root.name,'motion_records':len(names),'frames':frames,'caption_entries':captions,
        'split_records':{s:len(n) for s,n in splits.items()},'split_key_intersections':intersections,
        'split_exact_motion_hash_intersections':{a+'_'+b:len(hashes[a]&hashes[b]) for a in splits for b in splits if a<b},
        'all_motion_values_finite':True,'frame_dim':36,'quaternion_norm_max_abs_error':quat_error,
        'length_mismatch_count':len(length_mismatches),'train_frames_for_stats':train_frames,
        'provided_mean_vs_train_mean_max_abs':float(np.abs(mean-supplied_mean).max()),
        'provided_std_vs_train_std_max_abs':float(np.abs(std-supplied_std).max()),
        'normalization_preserved_from_official_release':True,'physical_feasibility_evaluated':False,
        'official_split_summary':json.loads((root/'split_summary.json').read_text()),
        'sample_record_fields':list(motions[names[0]])}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    results=[check(path) for path in sorted(a.data.iterdir()) if (path/'full_train.pkl').exists()]
    a.output.write_text(json.dumps({'status':'DATA_NUMERIC_AUDIT_COMPLETED','datasets':results},indent=2)+'\n')
    print(json.dumps({'datasets':[{'dataset':r['dataset'],'records':r['motion_records'],'frames':r['frames'],'split_records':r['split_records'],'quat_error':r['quaternion_norm_max_abs_error']} for r in results]}))
