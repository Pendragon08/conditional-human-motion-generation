"""Inventory hash-pinned OmniRetarget members with conservative provisional groups.

This creates no train/validation split and no semantic captions. Filename-based
IDs are not independently verified people or source recordings. All in-house
terrain variants remain together because their cross-archive ancestry is unknown.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

OBJECT=re.compile(r'^(sub[0-9]+)_(.+)_([0-9]+)_(original|rot_[0-9]+|trans_[0-9]+)\.npz$')
TERRAIN=re.compile(r'^(climb_[0-9]+)_z_scale_([0-9]+(?:\.[0-9]+)?)\.npz$')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True)
    p.add_argument('--provenance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.archives.resolve(strict=True);out=a.output.resolve(strict=True)
    provenance=json.loads(a.provenance.read_text(encoding='utf8'))
    if provenance.get('status')!='three_official_release_archives_match':raise ValueError('unverified release provenance')
    expected={r['archive']:r['official_pointer_sha256'] for r in provenance['archives']}
    if set(expected)!={'robot-object.zip','robot-terrain.zip','robot-object-terrain.zip'}:raise ValueError('archive set mismatch')
    counts=Counter();groups=Counter();variants=Counter();subject_ids=set();member_total=0
    with (out/'members.jsonl').open('x',encoding='utf8') as sink:
        for archive in sorted(expected):
            source=root/archive
            if sha(source)!=expected[archive]:raise ValueError('archive hash mismatch')
            names=set()
            with zipfile.ZipFile(source) as z:
                for info in z.infolist():
                    if info.is_dir():continue
                    member=info.filename;path=PurePosixPath(member)
                    if (member in names or path.is_absolute() or '..' in path.parts or len(path.parts)!=2 or
                        path.parts[0]!=archive[:-4] or path.suffix!='.npz'):
                        raise ValueError('duplicate, unsafe or unexpected archive entry')
                    names.add(member);stem=path.name;family=archive[:-4]
                    if family=='robot-object':
                        m=OBJECT.fullmatch(stem)
                        if not m:raise ValueError('unrecognized object member name')
                        subject,obj,clip,variant=m.groups();subject_ids.add(subject)
                        group='OMOMO_nominal_subject:'+subject
                        action_key=subject+'_'+obj+'_'+clip
                    elif family=='robot-terrain':
                        m=TERRAIN.fullmatch(stem)
                        if not m:raise ValueError('unrecognized terrain member name')
                        group='inhouse_all_sources_unresolved';action_key=m.group(1);variant='z_scale_'+m.group(2)
                    else:
                        group='inhouse_all_sources_unresolved';action_key=None;variant='unresolved'
                    h=hashlib.sha256();size=0
                    with z.open(info) as f:
                        for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
                    if size!=info.file_size:raise ValueError('member size mismatch')
                    row={'archive':archive,'archive_sha256':expected[archive],'member':member,
                         'member_sha256':h.hexdigest(),'member_bytes':size,'nominal_source_group':group,
                         'nominal_action_key':action_key,'variant_name':variant,'split':'unassigned',
                         'caption_status':'not_provided_by_this_inventory','training_ready':False}
                    sink.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    counts[family]+=1;groups[group]+=1;variants[family+':'+variant.split('_')[0]]+=1;member_total+=1
            if sha(source)!=expected[archive]:raise RuntimeError('archive changed during inventory')
    if counts!={'robot-object':1826,'robot-terrain':145,'robot-object-terrain':148}:
        raise ValueError('unexpected release member counts')
    result={'release_revision':provenance['revision'],'release_provenance_sha256':sha(a.provenance),
            'members_sha256':sha(out/'members.jsonl'),'members':member_total,'counts':dict(counts),
            'nominal_groups':dict(groups),'nominal_subject_labels':sorted(subject_ids),'variant_counts':dict(variants),
            'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':['Names indicate provisional relationships, not verified identity or independence.',
                'OMOMO versus in-house subjects can overlap; in-house sources are conservatively combined.',
                'No source text, timestamps, 29-to-27 conversion, license-chain or physical quality acceptance.']}
    with (out/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
