"""Cross-check the BMLmovi/Eyes candidate view using original inventories."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--view',type=Path,required=True)
    p.add_argument('--bml',type=Path,required=True)
    p.add_argument('--eyes',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    paths=[a.view,a.bml/'records.jsonl',a.eyes/'records.jsonl',a.candidate/'records.jsonl',a.candidate/'report.json']
    hashes={str(p):sha(p) for p in paths}
    view={r['motion_id']:r for r in map(json.loads,a.view.read_text(encoding='utf8').splitlines())}
    original={}
    for directory, dataset in ((a.bml,'BMLmovi'),(a.eyes,'Eyes_Japan_Dataset')):
      for line in (directory/'records.jsonl').read_text(encoding='utf8').splitlines():
        s=json.loads(line)
        for r in s['nominal_rows']:
          if r['motion_id'] in original:raise ValueError('duplicate source ID')
          original[r['motion_id']]=(dataset,s,r)
    seen=set();counts=Counter();path_splits=defaultdict(set);hash_splits=defaultdict(set)
    for line in (a.candidate/'records.jsonl').read_text(encoding='utf8').splitlines():
      r=json.loads(line);sid=r['sample_id']
      if sid in seen or sid not in original or sid not in view:raise ValueError('duplicate/missing ID')
      seen.add(sid);dataset,source,row=original[sid];v=view[sid]
      if (r['dataset']!=dataset or r['source_member']!=source['member'] or r['source_member_sha256']!=source['member_sha256']
          or r['source_frames']!=source['poses_shape'][0] or r['source_declared_fps']!=source['declared_fps']
          or r['nominal_index_interval']!=[int(row['start']),int(row['end'])]
          or r['interval_within_declared_count']!=row['interval_within_nominal_count']
          or r['official_split']!=v['official_split'] or r['auxiliary_split']!=v['view_split']
          or r['auxiliary_disposition']!=v['view_disposition'] or r['mirrored']!=v['mirrored']
          or r['caption_accepted'] is not False or r['training_ready'] is not False):
        raise ValueError('candidate differs from input')
      counts['dataset_'+dataset]+=1
      if r['auxiliary_disposition']=='retained':
        counts['retained_'+r['auxiliary_split']]+=1
        path_splits[row['path']].add(r['auxiliary_split']);hash_splits[source['member_sha256']].add(r['auxiliary_split'])
      else:counts['excluded_'+r['official_split']]+=1
    report=json.loads((a.candidate/'report.json').read_text(encoding='utf8'))
    if seen!=set(original) or counts!=Counter(report['counts']) or sha(a.candidate/'records.jsonl')!=report['records_sha256']:
        raise ValueError('coverage or report mismatch')
    if any(len(s)>1 for s in path_splits.values()) or any(len(s)>1 for s in hash_splits.values()):
        raise ValueError('cross-split source or byte hash')
    if any(sha(Path(k))!=v for k,v in hashes.items()):raise RuntimeError('inputs changed')
    result={'records':len(seen),'counts':dict(counts),'inputs_sha256':hashes,
            'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':['Recomputes candidate bindings from inventory JSON, not original archive bytes.',
                           'No verification of actual clocks, identity, captions, physical quality or training eligibility.']}
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'records':len(seen),'counts':dict(counts),'training_ready':False},ensure_ascii=False))


if __name__=='__main__':main()
