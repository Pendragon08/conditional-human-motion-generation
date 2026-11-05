"""Stream a bounded BMLmovi source selection without modifying the archive."""
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
SELECTION=ROOT/'data/bmlmovi_diverse_selection_20261102.json'
INVENTORY=ROOT/'data/bmlmovi_smplx_inventory_20261102/records.jsonl'
PROGRESS=ROOT/'data/bmlmovi_smplx_inventory_20261102/progress.jsonl'
VIEW=ROOT/'data/amass_bml_eyes_source_view_20261102/records.jsonl'
OUT=ROOT/'data/bmlmovi_diverse_source_20261102'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def guard(extra=0):
    if shutil.disk_usage(ROOT).free < 512*1024**2+extra:
        raise RuntimeError('local disk budget exceeded')


def main():
    selection_hash=sha(SELECTION);selection=json.loads(SELECTION.read_text(encoding='utf8'))
    if (selection['training_ready'] is not False or sha(INVENTORY)!=selection['inventory_sha256']
        or sha(VIEW)!=selection['source_view_sha256']):raise ValueError('selection provenance mismatch')
    rows=selection['members']
    if len(rows)!=20 or len({r['id'] for r in rows})!=20 or len({r['member'] for r in rows})!=20:
        raise ValueError('invalid selection')
    if sum(r['member_size'] for r in rows)!=selection['estimated_member_bytes']:
        raise ValueError('selection size mismatch')
    start=json.loads(PROGRESS.read_text(encoding='utf8').splitlines()[0])
    archive=Path(start['archive']);before=archive.stat()
    if before.st_size!=start['archive_size'] or before.st_mtime_ns!=start['archive_mtime_ns']:
        raise ValueError('archive metadata changed')
    guard(selection['estimated_member_bytes'])
    OUT.mkdir(exist_ok=False)
    remaining={r['member']:r for r in rows};found=[];began=time.monotonic()
    with (OUT/'progress.jsonl').open('x',encoding='utf8',buffering=1) as log:
      with tarfile.open(archive,'r|bz2') as t:
        for i,m in enumerate(t):
          if i%250==0:
            guard();log.write(json.dumps({'members_scanned':i,'found':len(found),'elapsed_s':time.monotonic()-began})+'\n')
          row=remaining.get(m.name)
          if row is None:continue
          if not m.isfile() or m.size!=row['member_size'] or m.size>8_000_000:
              raise ValueError('unexpected target member')
          guard(m.size);data=t.extractfile(m).read()
          if len(data)!=m.size or hashlib.sha256(data).hexdigest()!=row['member_sha256']:
              raise ValueError('member hash mismatch')
          target=OUT/(row['id']+'_smplx_stageii.npz')
          with target.open('xb') as f:f.write(data)
          found.append({**row,'output':target.name,'bytes':len(data),'output_sha256':sha(target)})
          del remaining[m.name]
          log.write(json.dumps({'event':'found','id':row['id'],'members_scanned':i,'elapsed_s':time.monotonic()-began})+'\n')
          if not remaining:break
    after=archive.stat()
    if remaining or before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
        raise ValueError('missing members or changed archive')
    if sha(SELECTION)!=selection_hash or sha(INVENTORY)!=selection['inventory_sha256'] or sha(VIEW)!=selection['source_view_sha256']:
        raise RuntimeError('input changed during extraction')
    result={'selection_sha256':selection_hash,'archive_size':before.st_size,'archive_mtime_ns':before.st_mtime_ns,
            'members':found,'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':['Source names do not prove recording identity or different humans.',
                'Member hashes checked against inventory; no full archive SHA computed.',
                'Captions, actual time, robot conversion and training are not accepted.']}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'completed':len(found),'bytes':sum(r['bytes'] for r in found),'training_ready':False}))


if __name__=='__main__':main()
