"""Check GRAB subject archives for raw marker files; no pickle loading."""
import hashlib,io,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
REGISTRY=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
REGISTRY_SHA='27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4'
MARKER_EXTS=('.c3d','.trc','.csv','.tsv','.vsk','.vicon','.h5')

def digest(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def sha(p):
    with Path(p).open('rb') as f:return digest(f)

def main():
    if sha(REGISTRY)!=REGISTRY_SHA:raise ValueError('registry changed')
    groups=defaultdict(list)
    for row in map(json.loads,REGISTRY.read_text().splitlines()):groups[row['subject']].append(row)
    if len(groups)!=10 or sum(map(len,groups.values()))!=1335:raise ValueError('registry coverage')
    archives=[]
    for subject in sorted(groups,key=lambda s:int(s[1:])):
        rows=groups[subject];path=ROOT/'data/GRAB'/rows[0]['archive']
        if any(r['archive_sha256']!=rows[0]['archive_sha256'] for r in rows) or sha(path)!=rows[0]['archive_sha256']:
            raise ValueError('archive changed')
        with zipfile.ZipFile(path) as z:
            files=[x.filename for x in z.infolist() if not x.is_dir()]
            if len(files)!=len(set(files)):raise ValueError('duplicate archive members')
            markers=[name for name in files if name.lower().endswith(MARKER_EXTS)]
            sample=min(rows,key=lambda r:r['member'])
            contents=z.read(sample['member'])
            if hashlib.sha256(contents).hexdigest()!=sample['member_sha256']:raise ValueError('sample member changed')
            with zipfile.ZipFile(io.BytesIO(contents)) as nested:
                names=sorted(x.filename for x in nested.infolist())
            archives.append({'subject':subject,'archive_sha256':rows[0]['archive_sha256'],'archive_files':len(files),
                             'file_extensions':dict(Counter(Path(x).suffix.lower() or '<none>' for x in files)),
                             'raw_marker_like_members':markers,'sample_motion':sample['member'],
                             'sample_motion_sha256':sample['member_sha256'],'sample_npz_fields':names})
    result={'registry_sha256':REGISTRY_SHA,'script_sha256':sha(__file__),'archives':archives,
            'raw_marker_like_member_count':sum(len(a['raw_marker_like_members']) for a in archives),
            'source_paper':'https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490562-supp.pdf',
            'limitations':'Searches only 10 subject ZIP central directories and one NPZ member field list per subject, without loading pickle. Marker data could be embedded in other fields, stored in other files or in unexamined archives; file extension and absence do not prove global absence. Floor marker coordinates require calibration to the GRAB fitted-body coordinate system and independent verification before use.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'raw_marker_like_member_count':result['raw_marker_like_member_count'],'subjects':len(archives)}),flush=True)
if __name__=='__main__':main()
