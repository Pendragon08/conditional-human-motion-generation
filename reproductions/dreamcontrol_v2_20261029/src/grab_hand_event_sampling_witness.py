"""Compare 120 Hz declared GRAB label runs with hypothetical 20 Hz sampling phases."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
REGISTRY=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
MEMBERS=['s1/teapot_pour_1.npz','s3/apple_eat_1.npz','s4/waterbottle_drink_1.npz','s5/cup_pour_1.npz']
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def runs(mask):
    change=np.diff(np.r_[False,mask,False].astype(np.int8)); starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)
    return [(int(a),int(b)) for a,b in zip(starts,ends)]
def metrics(mask):
    events=runs(mask); phases=[]
    for phase in range(6):
        sample=np.arange(phase,len(mask),6); missed=[(a,b) for a,b in events if not np.any((sample>=a)&(sample<b))]
        full_bins=[(a,a+6) for a in range(phase,len(mask)-5,6)]
        phases.append({'phase':phase,'sample_count':len(sample),'missed_runs':len(missed),'missed_intervals_half_open':missed,'full_bin_count':len(full_bins),'full_bins_with_any_not_sampled':sum(bool(mask[a:b].any() and not mask[a]) for a,b in full_bins)})
    return {'present_frames':int(mask.sum()),'intervals_half_open':events,'phase_comparison':phases}
def main():
    if sha(REGISTRY)!='27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4':raise ValueError('registry mismatch')
    rows={x['member']:x for x in map(json.loads,REGISTRY.read_text().splitlines()) if x['member'] in MEMBERS};assert set(rows)==set(MEMBERS)
    out=[]
    for member in MEMBERS:
        row=rows[member]; archive=ROOT/'data/GRAB'/row['archive']; assert sha(archive)==row['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            raw=z.read(member); assert hashlib.sha256(raw).hexdigest()==row['member_sha256']
            c=read_grab_field(z,member,'contact')['object']
        assert c.shape==(row['frames'],row['object_vertices']) and row['fps_declared']==120
        right=np.any((c==22)|((c>=41)&(c<=55)),axis=1); left=np.any((c==21)|((c>=26)&(c<=40)),axis=1)
        out.append({'member':member,'member_sha256':row['member_sha256'],'archive_sha256':row['archive_sha256'],'frames':row['frames'],'declared_fps':120,'right':metrics(right),'left':metrics(left)})
        del c
    result={'registry_sha256':sha(REGISTRY),'reader_sha256':sha(Path(sys.modules['grab_restricted_reader'].__file__)),'script_sha256':sha(Path(__file__)),'motions':out,'limitation':'Four nonrandom previously studied sequences. Presence of a labeled vertex and resulting runs are discrete labels, not ground-truth force or measured contact. Frame boundaries may truncate runs. Assumed 120 Hz header only; 20 Hz phase analysis is hypothetical. No robot or physical validation.'}
    with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
if __name__=='__main__':main()
