"""Independently reconcile every official ZIP member against a provisional inventory."""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import zipfile

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;ARCH=ROOT/'data/OmniRetarget_Dataset'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    report=json.loads((HERE/'report.json').read_text());manifest=HERE/'members.jsonl'
    assert sha(manifest)==report['members_sha256']
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    index={(r['archive'],r['member']):r for r in rows}
    assert len(index)==len(rows)==2119
    seen=set();groups=Counter();by_hash=defaultdict(list);actions=defaultdict(set)
    for archive in ('robot-object.zip','robot-terrain.zip','robot-object-terrain.zip'):
        ap=ARCH/archive;archive_digest=sha(ap)
        with zipfile.ZipFile(ap) as z:
            for info in z.infolist():
                if info.is_dir():continue
                key=(archive,info.filename);assert key not in seen and key in index;seen.add(key)
                row=index[key];assert row['archive_sha256']==archive_digest
                h=hashlib.sha256();size=0
                with z.open(info) as src:
                    for block in iter(lambda:src.read(512*1024),b''):h.update(block);size+=len(block)
                assert h.hexdigest()==row['member_sha256'] and size==row['member_bytes']
                assert row['split']=='unassigned' and row['training_ready'] is False
                name=Path(info.filename).name[:-4];parts=name.split('_')
                if archive=='robot-object.zip':
                    assert parts[0].startswith('sub') and parts[0][3:].isdigit()
                    if parts[-1]=='original':base=parts[:-1];variant='original'
                    else:
                        assert len(parts)>=5 and parts[-2] in ('rot','trans') and parts[-1].isdigit()
                        base=parts[:-2];variant='_'.join(parts[-2:])
                    assert base[-1].isdigit() and row['nominal_action_key']=='_'.join(base)
                    assert row['variant_name']==variant and row['nominal_source_group']=='OMOMO_nominal_subject:'+parts[0]
                elif archive=='robot-terrain.zip':
                    assert parts[0]=='climb' and parts[2:4]==['z','scale']
                    assert row['nominal_action_key']=='_'.join(parts[:2]) and row['variant_name']=='_'.join(parts[2:])
                    assert row['nominal_source_group']=='inhouse_all_sources_unresolved'
                else:assert row['nominal_action_key'] is None and row['nominal_source_group']=='inhouse_all_sources_unresolved'
                group=row['nominal_source_group'];groups[group]+=1
                if row['nominal_action_key'] is not None:actions[archive].add(row['nominal_action_key'])
                by_hash[row['member_sha256']].append({'archive':archive,'member':info.filename,'nominal_group':group})
        assert sha(ap)==archive_digest
    assert seen==set(index) and dict(groups)==report['nominal_groups']
    duplicate_groups=[v for v in by_hash.values() if len({x['nominal_group'] for x in v})>1]
    result={'inventory_report_sha256':sha(HERE/'report.json'),'members_sha256':sha(manifest),'entries_checked':len(seen),
            'nominal_action_keys_by_archive':{k:len(v) for k,v in actions.items()},
            'nominal_groups':dict(groups),'identical_npz_hashes_across_nominal_groups':duplicate_groups,
            'training_ready':False,'limitations':['Recomputed ZIP-member hashes and filename parsing only.',
              'Equal motion arrays stored differently or shared subjects/recordings may still cross nominal groups.',
              'This does not certify source consent, text, times, robot adaptation, contact or physics.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
