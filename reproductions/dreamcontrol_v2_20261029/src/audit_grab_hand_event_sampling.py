"""Independently recount event intervals and sampling phases from GRAB labels."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
REGISTRY=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
source=Path(sys.argv[1]); data=json.loads(source.read_text()); assert sha(REGISTRY)==data['registry_sha256']
rows={x['member']:x for x in map(json.loads,REGISTRY.read_text().splitlines())}; checked=0
for item in data['motions']:
    row=rows[item['member']];p=ROOT/'data/GRAB'/row['archive'];assert sha(p)==row['archive_sha256']==item['archive_sha256']
    with zipfile.ZipFile(p) as z:
        raw=z.read(item['member']);assert hashlib.sha256(raw).hexdigest()==row['member_sha256']==item['member_sha256']
        c=read_grab_field(z,item['member'],'contact')['object']
    assert c.shape==(row['frames'],row['object_vertices'])
    for name,ids in [('left',[21,*range(26,41)]),('right',[22,*range(41,56)])]:
        v=np.isin(c,ids).any(axis=1).tolist(); entry=item[name]; spans=[];start=None
        for i,value in enumerate(v+[False]):
            if value and start is None:start=i
            elif not value and start is not None:spans.append([start,i]);start=None
        assert spans==entry['intervals_half_open'] and sum(v)==entry['present_frames']
        for k in range(6):
            expected=entry['phase_comparison'][k]; assert expected['phase']==k
            samples=list(range(k,len(v),6));missing=[span for span in spans if not any(a in range(span[0],span[1]) for a in samples)]
            bins=list(range(k,len(v)-5,6)); occupied_without_first=sum(1 for a in bins if any(v[a:a+6]) and not v[a])
            assert missing==expected['missed_intervals_half_open'] and len(missing)==expected['missed_runs']
            assert len(samples)==expected['sample_count'] and len(bins)==expected['full_bin_count'] and occupied_without_first==expected['full_bins_with_any_not_sampled']
            checked+=1
    del c
result={'status':'passed','source_report_sha256':sha(source),'registry_sha256':sha(REGISTRY),'reader_sha256':sha(Path(sys.modules['grab_restricted_reader'].__file__)),'script_sha256':sha(Path(__file__)),'motions_checked':len(data['motions']),'hand_phase_cases_checked':checked,'limitations':'Same restricted reader as witness, independent label selection/run and phase loops; no independent real FPS or physical contact.'}
with Path(sys.argv[2]).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
