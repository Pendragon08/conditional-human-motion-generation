"""Bind newly inventoried BMLmovi/Eyes source members to the existing auxiliary view.

Outputs are source candidates only: no feature conversion or training acceptance.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--view',type=Path,required=True)
    p.add_argument('--bml',type=Path,required=True)
    p.add_argument('--eyes',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    paths=[a.view,a.bml/'records.jsonl',a.bml/'summary.json',a.eyes/'records.jsonl',a.eyes/'summary.json']
    hashes={str(x):sha(x) for x in paths}
    view_rows=[json.loads(x) for x in a.view.read_text(encoding='utf8').splitlines()]
    view={r['motion_id']:r for r in view_rows}
    if len(view)!=len(view_rows):raise ValueError('duplicate auxiliary ID')
    a.out.mkdir(exist_ok=False)
    counts=Counter(); source_splits=defaultdict(set); content_splits=defaultdict(set); seen=set()
    with (a.out/'records.jsonl').open('x',encoding='utf8') as target:
      for inventory, dataset in ((a.bml,'BMLmovi'),(a.eyes,'Eyes_Japan_Dataset')):
        summary=json.loads((inventory/'summary.json').read_text(encoding='utf8'))
        if summary['dataset']!=dataset or summary['event']!='complete':raise ValueError('incomplete source inventory')
        for line in (inventory/'records.jsonl').read_text(encoding='utf8').splitlines():
          source=json.loads(line)
          if source.get('duplicate_key') or 'error' in source or source.get('training_accepted') is not False:
              raise ValueError('source unresolved or duplicate')
          for row in source['nominal_rows']:
            sid=row['motion_id']; v=view.get(sid)
            if sid in seen or v is None or row['path']!=v['exact_source_path']:
                raise ValueError('source/auxiliary binding mismatch')
            seen.add(sid)
            result={'sample_id':sid,'dataset':dataset,'source_member':source['member'],
                    'source_member_sha256':source['member_sha256'],'source_frames':source['poses_shape'][0],
                    'source_declared_fps':source['declared_fps'],'nominal_stride':source['integer_stride'],
                    'nominal_index_interval':[int(row['start']),int(row['end'])],
                    'interval_within_declared_count':row['interval_within_nominal_count'],
                    'mirrored':v['mirrored'],'official_split':v['official_split'],
                    'auxiliary_disposition':v['view_disposition'],'auxiliary_split':v['view_split'],
                    'source_name_is_identity_proof':False,'caption_accepted':False,'training_ready':False}
            if result['auxiliary_disposition']=='retained':
                s=result['auxiliary_split'];source_splits[row['path']].add(s);content_splits[source['member_sha256']].add(s)
                counts['retained_'+s]+=1
            else:counts['excluded_'+result['official_split']]+=1
            counts['dataset_'+dataset]+=1
            target.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
    if any(len(x)>1 for x in source_splits.values()) or any(len(x)>1 for x in content_splits.values()):
        raise ValueError('retained nominal source or exact source bytes cross auxiliary splits')
    if any(sha(Path(k))!=v for k,v in hashes.items()):raise RuntimeError('input changed')
    result={'counts':dict(counts),'records_sha256':sha(a.out/'records.jsonl'),'inputs_sha256':hashes,
            'exact_index_source_cross_split_groups':0,'exact_source_bytes_cross_split_groups':0,
            'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':['Existing nonofficial view excludes some official samples and does not prove subject independence.',
              'Archive members were not re-read here; source hashes come from previous streaming inventory.',
              'Nominal header FPS, source identity, captions, robot features and physical quality unaccepted.']}
    with (a.out/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'counts':dict(counts),'training_ready':False},ensure_ascii=False))


if __name__=='__main__':main()
