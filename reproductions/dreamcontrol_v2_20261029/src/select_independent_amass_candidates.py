"""Bounded selection from an existing, hash-verified HumanML source inventory.

Source path matches are hypotheses; source identity, time, model equivalence,
caption semantics and physical quality must be checked separately. Official
splits are unchanged. This script does not create training data.
"""
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE=Path(__file__).resolve().parent
DATASETS=('MPI_HDM05','EKUT','Transitions_mocap','MPI_mosh','ACCAD','DFaust_67','SFU','HumanEva')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    mapping=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
    summary=json.loads((mapping.parent/'summary.json').read_text())
    if sha(mapping)!=summary['mapping_sha256']:raise ValueError('inventory changed')
    rows=[json.loads(line) for line in mapping.read_text().splitlines()]
    groups=collections.defaultdict(list)
    for r in rows:
        path=PurePosixPath(r['source_path']);dataset=path.parts[1]
        if dataset not in DATASETS:continue
        groups[(dataset,str(path.parent))].append(r)
    group_counts=collections.Counter();select=[]
    for (dataset,parent),items in sorted(groups.items()):
        splits={x['official_split'] for x in items}
        group_counts[(dataset,','.join(sorted(splits)))]+=1
        if splits!={'train'}:continue
        choices=sorted((r for r in items if not r['mirrored'] and r['candidate_local_path'] is not None
                        and r['status'] in ('candidate_path_match','candidate_normalized_name_match')),
                       key=lambda r:r['motion_id'])
        if not choices:continue
        if sum(r['dataset']==dataset for r in select)>=2:continue
        chosen=choices[0];source=Path(chosen['candidate_local_path'])
        if not source.is_file():continue
        with np.load(source,allow_pickle=False) as f:
            if not {'poses','trans','betas'}.issubset(f.files):continue
            poses=f['poses'];trans=f['trans'];betas=f['betas']
            fpsfield=next((k for k in ('mocap_framerate','mocap_frame_rate') if k in f.files),None)
            if fpsfield is None:continue
            fps=float(f[fpsfield].item())
            if not np.isfinite(fps) or fps<=0 or poses.ndim!=2 or trans.shape!=(len(poses),3):continue
            shapes={'poses':list(poses.shape),'trans':list(trans.shape),'betas':list(betas.shape),'fps_declared':fps}
        select.append({'dataset':dataset,'parent_group':parent,'motion_id':chosen['motion_id'],
                       'source_path':chosen['source_path'],'candidate_path':str(source),'candidate_sha256':sha(source),
                       'index_start_20':chosen['index_start_20'],'index_end_20':chosen['index_end_20'],
                       'official_split':'train','path_match':chosen['status'],'shapes':shapes,
                       'group_has_official_splits':sorted(splits)})
    archive=BASE/'texts.zip'; archive_hash=sha(archive)
    with zipfile.ZipFile(archive) as z:
        for row in select:
            data=z.read('texts/'+row['motion_id']+'.txt'); row['caption_member_sha256']=hashlib.sha256(data).hexdigest()
            entries=[line.split('#') for line in data.decode('utf8').splitlines()]
            if any(len(x)!=4 for x in entries):raise ValueError('unexpected caption format')
            row['raw_captions']=[{'caption':x[0],'start_tag':x[2],'end_tag':x[3]} for x in entries]
    out={'mapping_sha256':sha(mapping),'caption_archive_sha256':archive_hash,
         'groups_by_dataset_and_official_splits':{k[0]+'|'+k[1]:v for k,v in sorted(group_counts.items())},
         'selected':select,'training_ready':False,
         'limitations':['Bounded selection: at most two known-parent groups per listed dataset.',
                        'Parent path and official-split purity are not verified subject independence.',
                        'Candidate path and declared FPS do not establish official source or actual time.',
                        'Full captions, model identity, license and physical quality remain unaccepted.']}
    with (HERE/'candidates.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'datasets':sorted({r['dataset'] for r in select}),'candidates':len(select),'training_ready':False}),flush=True)


if __name__=='__main__':main()
