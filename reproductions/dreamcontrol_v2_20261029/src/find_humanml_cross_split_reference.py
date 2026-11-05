"""Bounded search for reference/intermediate files for intersecting HumanML sources.

Matches use subject-directory plus normalized stem, not verified dataset identity.
This inventories names only, avoiding source decoding and data mutation.
"""
import hashlib,json,os,re,zipfile
from collections import defaultdict
from pathlib import Path,PurePosixPath

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN=ROOT/'humanml_source_split_audit_20261030'
OUT=Path(__file__).resolve().parent
ROOTS=(Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass'),
       Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/data'))
INTERSECTION_SHA='cc2787af8e145845520ccca93ac26039a0d4788aa19d873083f75153e22fd235'
EXCLUDED='./pose_data/ACCAD/Female1General_c3d/A12 - crawl backwards_poses.npy'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def normalize(s):return re.sub(r'[^a-z0-9]','',s.lower())

def key(path):
    p=PurePosixPath(path);name=p.name.lower()
    if not name.endswith(('_poses.npy','_poses.npz')):return None
    return normalize(p.parent.name), normalize(re.sub(r'_poses\.(npy|npz)$','',name))

def main():
    source=RUN/'nominal_cross_split_intersections.jsonl'
    if sha(source)!=INTERSECTION_SHA:raise ValueError('intersection source changed')
    groups=[json.loads(s) for s in source.read_text().splitlines()]
    if len(groups)!=555 or len({r['source_path'] for r in groups})!=555:raise ValueError('unexpected target coverage')
    targets=defaultdict(list)
    for g in groups:
        if g['source_path']==EXCLUDED:continue
        k=key(g['source_path'])
        if not k or not all(k):raise ValueError('invalid target key')
        targets[k].append(g['source_path'])
    if sum(map(len,targets.values()))!=554:raise ValueError('target count mismatch')
    files_seen=dirs_seen=zip_seen=0;hits=[];errors=[]
    def onerror(e):errors.append({'path':e.filename,'error':str(e)})
    for root in ROOTS:
        if not root.is_dir():errors.append({'path':str(root),'error':'root unavailable'});continue
        for current,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
            dirs_seen+=1;dirs[:]=[d for d in dirs if d not in ('.git','.cache','__pycache__') and not Path(current,d).is_symlink()]
            for name in files:
                files_seen+=1;p=Path(current,name)
                if p.is_symlink():continue
                k=key(p.as_posix())
                if k in targets:
                    hits.append({'kind':'file','path':str(p),'sha256':sha(p),'matching_index_sources':targets[k]})
                elif name.lower().endswith('.zip'):
                    zip_seen+=1
                    try:
                        with zipfile.ZipFile(p) as z:
                            member_hits=[{'member':i.filename,'matching_index_sources':targets[k]} for i in z.infolist()
                                         if not i.is_dir() and (k:=key(i.filename)) in targets]
                        if member_hits:hits.append({'kind':'zip','path':str(p),'sha256':sha(p),'members':member_hits})
                    except (OSError,zipfile.BadZipFile) as e:errors.append({'path':str(p),'error':str(e)})
    out={'script_sha256':sha(Path(__file__)),'intersection_sha256':INTERSECTION_SHA,'target_source_paths':554,
         'excluded_previously_searched':EXCLUDED,'searched_roots':[str(x) for x in ROOTS],
         'directories_visited':dirs_seen,'files_seen':files_seen,'zip_archives_indexed':zip_seen,'hits':hits,'errors':errors,
         'limitations':'Only subject directory and normalized basename matched; dataset aliases, file contents, model type, FPS and identity are unverified. Does not traverse symlinks or inspect tar/other archives and does not establish server-wide absence. ZIP members were not extracted.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,sort_keys=True)
    print(json.dumps(out,sort_keys=True),flush=True)

if __name__=='__main__':main()
