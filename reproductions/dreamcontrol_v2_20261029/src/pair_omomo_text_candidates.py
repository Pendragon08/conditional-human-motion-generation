"""Propose exact-name OMOMO captions for pinned OmniRetarget members.

This reads a public author annotation ZIP without extraction. A filename match
is only a candidate association: temporal alignment, semantic preservation in
rotated/transformed variants, dataset rights and motion quality remain unknown.
"""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path,PurePosixPath
import re,zipfile

ZIP_SHA='2d11fb37fef294b7f6816c74740694d12f28d62e0f638e348c678dc51ee1ad7c'
INVENTORY_SHA='2d187a58d308d9ddde198a17b720e1e9baeb04c0a5e11c11cb0964048a386340'
COMMIT='e9d8c52f41866ed0c07ac574cb8f7f9424b6d1e7'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--annotation-zip',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();archive=a.annotation_zip.resolve(strict=True);inventory=a.inventory.resolve(strict=True)
    out=a.output.resolve(strict=True)
    if sha(archive)!=ZIP_SHA or sha(inventory)!=INVENTORY_SHA:raise ValueError('source hash mismatch')
    annotations={};shape=Counter();names=set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            path=PurePosixPath(info.filename)
            if (info.filename in names or path.is_absolute() or '..' in path.parts or len(path.parts)!=2 or
                path.parts[0]!='omomo_text_anno_json_data' or path.suffix!='.json' or info.file_size>65536):
                raise ValueError('unsafe, duplicate, oversized or unexpected annotation member')
            names.add(info.filename);raw=z.read(info);obj=json.loads(raw.decode('utf8'))
            if not isinstance(obj,dict) or len(obj)!=1 or path.stem not in obj:
                raise ValueError('annotation filename/key mismatch')
            text=obj[path.stem]
            if not isinstance(text,str) or not text.strip():raise ValueError('annotation must be nonempty text')
            if path.stem in annotations:raise ValueError('duplicate annotation key')
            annotations[path.stem]={'member':info.filename,'member_sha256':hashlib.sha256(raw).hexdigest(),'text':text}
            shape[type(text).__name__]+=1
    rows=[json.loads(s) for s in inventory.read_text(encoding='utf8').splitlines()]
    object_rows=[r for r in rows if r['archive']=='robot-object.zip']
    if len(object_rows)!=1826:raise ValueError('unexpected pinned inventory')
    counts=Counter();basekeys={r['nominal_action_key'] for r in object_rows};matched=set()
    with (out/'candidate_pairs.jsonl').open('x',encoding='utf8') as f:
        for row in object_rows:
            key=row['nominal_action_key'];ann=annotations.get(key)
            if ann is None:counts['missing_annotation']+=1;continue
            matched.add(key);variant=row['variant_name']
            direction=bool(re.search(r'\b(left|right|clockwise|counterclockwise|forward|backward|backwards)\b',ann['text'],re.I))
            if variant=='original':counts['original']+=1
            else:counts['augmented']+=1
            if direction:counts['contains_direction_word']+=1
            pair={'archive':row['archive'],'archive_sha256':row['archive_sha256'],'motion_member':row['member'],
                  'motion_member_sha256':row['member_sha256'],'nominal_action_key':key,'variant':variant,
                  'provisional_group':row['nominal_source_group'],'annotation_member':ann['member'],
                  'annotation_member_sha256':ann['member_sha256'],'caption':ann['text'],
                  'contains_direction_word':direction,'pair_status':'filename_candidate_only',
                  'timestamp_status':'not_provided','semantic_status':'unreviewed',
                  'license_status':'annotation_and_upstream_rights_unverified','split':'unassigned','training_ready':False}
            f.write(json.dumps(pair,ensure_ascii=False,allow_nan=False)+'\n')
    if sha(archive)!=ZIP_SHA or sha(inventory)!=INVENTORY_SHA:raise RuntimeError('source changed during pairing')
    report={'annotation_commit':COMMIT,'annotation_zip_sha256':ZIP_SHA,'inventory_sha256':INVENTORY_SHA,
            'annotation_members':len(annotations),'annotation_shapes':dict(shape),'object_motion_members':len(object_rows),
            'nominal_action_keys':len(basekeys),'matched_nominal_action_keys':len(matched),
            'unmatched_nominal_action_keys':sorted(basekeys-matched),'candidate_pair_counts':dict(counts),
            'pairs_sha256':sha(out/'candidate_pairs.jsonl'),'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':['Exact filename match does not prove original recording identity, segment alignment or use rights.',
                'Original text is associated with source motion; transformed/rotated variants can change semantics.',
                'No time annotations, author-v2 method, 29-to-27 conversion or physical acceptance.']}
    with (out/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
