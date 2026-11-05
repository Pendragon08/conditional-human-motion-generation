"""Recompute published OmniRetarget archive provenance for a preview subset.

Verifies file bytes and declared sampling grids, never actual capture time or
the complete rights chain. The dataset card's MIT field is recorded as the
publisher's statement, not a legal determination or training authorization.
"""
from collections import Counter
import hashlib,io,json,shutil,zipfile
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SCHEMA=ROOT/'omomo_geometry_schema_preview_20261102'
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
CARD=ROOT/'omomo_release_evidence_work_20261102/omniretarget_README.md'
ARCHIVE_SHA='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3'
CARD_SHA='cd662e88d89f583d9007f659c32a50e7d4ddb2d0be394d45e144126911b09024'
OUT=ROOT/'omomo_release_provenance_v2_20261102'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    if shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    report=json.loads((SCHEMA/'report.json').read_text());manifest=SCHEMA/'manifest.json';states_path=SCHEMA/'state_manifest.json'
    mh=sha(manifest);sth=sha(states_path)
    if (mh!=report['manifest_sha256'] or sth!=report['state_manifest_sha256'] or
        sha(ARCHIVE)!=ARCHIVE_SHA or sha(CARD)!=CARD_SHA):raise ValueError('pinned evidence changed')
    card=CARD.read_text(encoding='utf8')
    if '\nlicense: mit\n' not in card or 'robot-object' not in card:raise ValueError('release card statement missing')
    records=json.loads(manifest.read_text())['records'];states=json.loads(states_path.read_text())['records']
    index={x['sample_id']:x for x in states}
    if len(records)!=1194 or len(index)!=len(states) or set(index)!={x['sample_id'] for x in records}:raise ValueError('state coverage')
    OUT.mkdir(exist_ok=False);counts=Counter();failures=[];clipped=0;seen=set()
    with zipfile.ZipFile(ARCHIVE) as z,(OUT/'records.jsonl').open('x',encoding='utf8',buffering=1) as out:
      names=z.namelist()
      if len(names)!=len(set(names)):raise ValueError('archive contains duplicate names')
      for i,row in enumerate(records):
        try:
          sid=row['sample_id'];member=row['source_member'];state=index[sid]
          if member in seen or not member.startswith('robot-object/') or not member.endswith('.npz'):raise ValueError('invalid source member')
          seen.add(member);info=z.getinfo(member)
          if info.file_size>64*1024**2:raise ValueError('source too large')
          payload=z.read(info)
          if hashlib.sha256(payload).hexdigest()!=row['source_sha256']:raise ValueError('source member hash')
          with np.load(io.BytesIO(payload),allow_pickle=False) as data:
              q=data['qpos'];fps=float(data['fps'])
              if q.ndim!=2 or len(q)<2 or not np.isfinite(q).all() or fps!=30.:raise ValueError('invalid source state or fps')
              source_frames=len(q)
          feature=(ROOT/row['feature_path']).resolve(strict=True);sp=(ROOT/state['state_path']).resolve(strict=True)
          if (not feature.is_relative_to(ROOT) or not sp.is_relative_to(ROOT) or
              sha(feature)!=row['feature_sha256'] or sha(sp)!=state['state_sha256'] or state['feature_sha256']!=row['feature_sha256']):
              raise ValueError('feature/state binding')
          with np.load(sp,allow_pickle=False) as data:times=np.asarray(data['sample_times'],dtype=float)
          with np.load(feature,allow_pickle=False) as data:n=len(data['features'])
          if (times.shape!=(n+1,) or not np.isfinite(times).all() or
              not np.allclose(times,np.arange(len(times))/20,rtol=0,atol=1e-9) or
              times[-1]>(source_frames-1)/fps+1e-9):raise ValueError('nominal time support')
          uncovered=(source_frames-1)/fps-times[-1];truncated=uncovered>2/30+1e-9
          clipped+=int(truncated);counts[row['split']]+=1
          result={'sample_id':sid,'source_member':member,'source_sha256':row['source_sha256'],
             'source_frames':source_frames,'source_declared_fps':fps,'state_frames':len(times),
             'state_nominal_time_s':[float(times[0]),float(times[-1])],
             'remaining_source_nominal_seconds':float(uncovered),'source_tail_over_two_frames':bool(truncated),
             'release_byte_and_nominal_support_checked':True,'training_ready':False}
        except Exception as e:
          result={'sample_id':row.get('sample_id'),'error':repr(e),'release_byte_and_nominal_support_checked':False,'training_ready':False};failures.append(result)
        out.write(json.dumps(result,allow_nan=False)+'\n')
        if i%100==0 and shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    if sha(manifest)!=mh or sha(states_path)!=sth or sha(ARCHIVE)!=ARCHIVE_SHA or sha(CARD)!=CARD_SHA:
        raise RuntimeError('evidence changed during audit')
    result={'manifest_sha256':mh,'state_manifest_sha256':sth,'archive_sha256':ARCHIVE_SHA,'card_sha256':CARD_SHA,
       'script_sha256':sha(Path(__file__)),'records_sha256':sha(OUT/'records.jsonl'),'selected':len(records),
       'checked':sum(counts.values()),'counts_by_split':dict(counts),'failures':failures,
       'source_tail_over_two_frames':clipped,'publisher_card_license_field':'mit','training_ready':False,
       'limitations':['Verifies bytes against pinned release and self-declared 30 Hz only; actual capture clock and recording identity are not established.',
          'Release card states MIT; rights in upstream OMOMO and associated assets need separate assessment.',
          'Time bounds and source bytes do not verify interaction semantics, contact, physical quality or training readiness.',
          'Machine captions describe FK joint motion, not the original annotation.']}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
