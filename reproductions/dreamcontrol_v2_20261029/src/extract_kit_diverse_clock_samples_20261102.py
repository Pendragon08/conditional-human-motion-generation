"""Extract eight small KIT diagnostic witnesses from the existing local archive.

Names are nominal subject directories, not proof of distinct humans. This is
not a training selection; original archive and existing results are untouched.
"""
import hashlib,json,os,shutil,tarfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
INVENTORY=HERE/'data/kit_smplx_stream_inventory_20261102/records.jsonl'
PROGRESS=HERE/'data/kit_smplx_stream_inventory_20261102/progress.jsonl'
OUT=HERE/'data/kit_diverse_clock_samples_20261102'
WANTED={'001199','006160','007754','006923','001171','007259','001893','009236'}
def sha(payload):return hashlib.sha256(payload).hexdigest()
def main():
    initial=json.loads(PROGRESS.read_text().splitlines()[0]);archive=Path(initial['archive']);before=archive.stat()
    if before.st_size!=initial['archive_size'] or before.st_mtime_ns!=initial['archive_mtime_ns']:raise ValueError('archive metadata changed')
    rows={}
    for line in INVENTORY.read_text().splitlines():
        r=json.loads(line)
        for rec in r['nominal_rows']:
            if rec['motion_id'] in WANTED:
                if rec['motion_id'] in rows:raise ValueError('ambiguous ID')
                rows[rec['motion_id']]={'id':rec['motion_id'],'member':r['member'],'member_sha256':r['member_sha256'],
                                        'subject_nominal':r['name_key'][0],'nominal_start':rec['start'],'nominal_end':rec['end']}
    if set(rows)!=WANTED or len({r['subject_nominal'] for r in rows.values()})!=8:raise ValueError('selection mismatch')
    if shutil.disk_usage(HERE).free<512*1024**2:raise RuntimeError('local disk budget')
    OUT.mkdir(exist_ok=False);remaining={r['member']:r for r in rows.values()};found=[];started=time.monotonic()
    with (OUT/'progress.jsonl').open('x',encoding='utf8',buffering=1) as log:
      with tarfile.open(archive,'r|bz2') as t:
        for i,m in enumerate(t):
          if i%250==0:
            log.write(json.dumps({'members_scanned':i,'found':len(found),'elapsed_s':time.monotonic()-started})+'\n')
            if shutil.disk_usage(HERE).free<512*1024**2:raise RuntimeError('local disk budget')
          row=remaining.get(m.name)
          if row is None:continue
          if not m.isfile() or m.size>8_000_000:raise ValueError('unexpected target member')
          data=t.extractfile(m).read()
          if len(data)!=m.size or sha(data)!=row['member_sha256']:raise ValueError('member hash mismatch')
          target=OUT/(row['id']+'_smplx_stageii.npz')
          with target.open('xb') as f:f.write(data)
          row['output']=target.name;row['bytes']=len(data);found.append(row);del remaining[m.name]
          log.write(json.dumps({'event':'found','id':row['id'],'members_scanned':i,'elapsed_s':time.monotonic()-started})+'\n')
          if not remaining:break
    after=archive.stat()
    if remaining or before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:raise ValueError('missing or changed archive')
    result={'archive_size':before.st_size,'archive_mtime_ns':before.st_mtime_ns,'inventory_sha256':sha(INVENTORY.read_bytes()),
            'members':found,'distinct_nominal_subjects':8,'elapsed_s':time.monotonic()-started,'training_ready':False,
            'limitations':['Nominal directory names do not prove different people or recording identity.',
              'Stored member hashes match prior inventory; full archive hash is not computed.',
              'No caption, clock, physical or training acceptance.']}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'completed':len(found),'elapsed_s':result['elapsed_s']}),flush=True)
if __name__=='__main__':main()
