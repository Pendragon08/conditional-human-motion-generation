"""Search two bounded data roots for a particular ACCAD witness, without extraction."""
import hashlib, json, os, re, zipfile
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
ROOTS=[Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass'),
       Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/data')]
EXPECTED='947d5691fc5decd7d3e1ef42b078e651c84db866efe28ebf2ad04e62e34f933f'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def matches(name):
    # Match only the selected witness stem after normalizing separators.
    lower=name.lower()
    if not lower.endswith(('_poses.npy','_poses.npz','_stageii.npz')):return False
    stem=re.sub(r'(_poses|_stageii)\.(npy|npz)$','',lower)
    return re.sub(r'[^a-z0-9]','',stem)=='a12crawlbackwards'

def main():
    examined=0; directories=0; zips=0; found=[]; errors=[]
    def onerror(e):errors.append({'path':e.filename,'error':str(e)})
    for root in ROOTS:
        if not root.is_dir():errors.append({'path':str(root),'error':'root unavailable'});continue
        for current,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
            directories+=1
            dirs[:]=[d for d in dirs if d not in ('__pycache__','.git','.cache') and not Path(current,d).is_symlink()]
            for name in files:
                examined+=1; path=Path(current,name)
                if path.is_symlink():continue
                if matches(name):
                    found.append({'kind':'file','path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
                elif name.lower().endswith('.zip'):
                    zips+=1
                    try:
                        with zipfile.ZipFile(path) as z:
                            matching=[i.filename for i in z.infolist() if not i.is_dir() and matches(Path(i.filename).name)]
                        if matching:found.append({'kind':'zip_members','path':str(path),'sha256':sha(path),'members':sorted(matching)})
                    except (OSError,zipfile.BadZipFile) as e:errors.append({'path':str(path),'error':str(e)})
    candidate=[x for x in found if x['kind']=='file' and x['sha256']==EXPECTED]
    if not candidate:raise ValueError('known witness missing from selected roots')
    report={'script_sha256':sha(Path(__file__)),'searched_roots':[str(x) for x in ROOTS],
            'target_source':'./pose_data/ACCAD/Female1General_c3d/A12 - crawl backwards_poses.npy',
            'directories_visited':directories,'files_seen':examined,'zip_archives_indexed':zips,
            'matches':found,'errors':errors,
            'limitations':'A bounded basename-normalized search of two data roots and ZIP central directories. Symlinks, tar and other archives were not searched. Name match does not prove identity or model type; missing results do not establish absence elsewhere. ZIP members were not extracted or individually hashed.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,sort_keys=True)
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':main()
