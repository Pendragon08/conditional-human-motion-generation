"""Check two candidate BMLhandball recordings against split index and bytes."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PUBLIC=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public')
DATA=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/BMLhandball')
OUT=Path(__file__).resolve().parent
TARGETS=(('S04_Expert/Trial_upper_right_left_227_poses.npz','002323','008550'),
         ('S08_Novice/Trial_upper_left_right_043_poses.npz','001705','014519'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    search=ROOT/'humanml_cross_split_reference_search_20261030/report.json'
    if sha(search)!='71d4e6f86f36b061556c54869f34de385a6150aba80e932aea53cc35356a73e2':raise ValueError('search changed')
    if sha(PUBLIC/'index.csv')!='80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065':raise ValueError('index changed')
    with (PUBLIC/'index.csv').open(newline='',encoding='utf8') as f:
        index={r['new_name'].removesuffix('.npy'):r for r in csv.DictReader(f)}
    splits={s:set((PUBLIC/'HumanML3D'/f'{s}.txt').read_text().splitlines()) for s in ('train','test')}
    history_path=DATA/'server_verification.json';history=json.loads(history_path.read_text());history_rows={r['path']:r for r in history['files']}
    hits={h['path']:h for h in json.loads(search.read_text())['hits'] if h['kind']=='file'}
    out=[]
    for rel,test_id,train_id in TARGETS:
        path=DATA/'extracted/BMLhandball'/rel;current=sha(path);prior=history_rows['BMLhandball/'+rel]
        if current!=prior['sha256'] or current!=hits[str(path)]['sha256'] or path.stat().st_size!=prior['bytes']:
            raise ValueError('file hash or provenance mismatch')
        a,b=index[test_id],index[train_id]
        if a['source_path']!=b['source_path'] or a['source_path'] not in hits[str(path)]['matching_index_sources']:
            raise ValueError('source index mismatch')
        if test_id not in splits['test'] or train_id not in splits['train'] or ('M'+test_id) not in splits['test'] or ('M'+train_id) not in splits['train']:
            raise ValueError('split or mirror mismatch')
        a0,a1,b0,b1=(int(a['start_frame']),int(a['end_frame']),int(b['start_frame']),int(b['end_frame']))
        with np.load(path,allow_pickle=False) as f:
            keys=sorted(f.files);fps=float(np.asarray(f['mocap_framerate']).item());pose=np.asarray(f['poses']);trans=np.asarray(f['trans'])
        if not np.isfinite(fps) or fps<20 or pose.ndim!=2 or trans.shape!=(len(pose),3) or not np.isfinite(pose).all() or not np.isfinite(trans).all():
            raise ValueError('invalid numeric arrays')
        stride=int(fps/20);sampled=np.arange(0,len(pose),stride)
        if max(a1,b1)>len(sampled) or min(a0,b0)<0:raise ValueError('candidate slice truncated')
        left=sampled[a0:a1];right=sampled[b0:b1]
        common=np.intersect1d(left,right,assume_unique=True)
        nominal_first=max(a0,b0);nominal_end=min(a1,b1)
        expected=sampled[nominal_first:nominal_end]
        if not np.array_equal(common,expected):raise ValueError('numeric source overlap mismatch')
        for arr in (pose,trans):
            if not np.array_equal(arr[left[nominal_first-a0:nominal_end-a0]],arr[right[nominal_first-b0:nominal_end-b0]]):
                raise ValueError('overlap arrays differ')
        out.append({'source_path':a['source_path'],'candidate_path':str(path),'candidate_sha256':current,
                    'archive_historical_sha256':history['archive_sha256'],'keys':keys,'pose_shape':list(pose.shape),
                    'declared_fps':fps,'stride_from_declared_fps':stride,'candidate_samples':len(sampled),
                    'ids':{test_id:'test',train_id:'train'},'index_slices':{test_id:[a0,a1],train_id:[b0,b1]},
                    'shared_sampled_indices':[nominal_first,nominal_end],'shared_sample_count':len(common),
                    'shared_first_raw_frame':int(common[0]),'shared_last_raw_frame':int(common[-1]),
                    'pose_and_translation_overlap_byte_equal':True})
    report={'script_sha256':sha(__file__),'search_sha256':sha(search),'history_sha256':sha(history_path),
            'official_index_sha256':sha(PUBLIC/'index.csv'),'witnesses':out,
            'limitations':'Same-candidate sampled arrays are equal in overlapping slices, conditional on its declared FPS and source mapping. A 156-coordinate pose shape and historical receipt do not prove authoritative SMPL+H release, capture FPS, per-frame identity to the official intermediate pose, mirror equivalence or physical/training validity. No split changed.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':main()
