"""Independently reconcile source text, inventory and provisional pair records."""
from collections import Counter
import hashlib,json
from pathlib import Path,PurePosixPath
import zipfile

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;ZIP=HERE/'omomo_text_anno.zip'
INVENTORY=ROOT/'omni_release_member_inventory_20261101/members.jsonl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=json.loads((HERE/'report.json').read_text());raw=ZIP.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob=='32b33c500d9c42d66662e1682986b442c7c164a8'
    assert sha(ZIP)==report['annotation_zip_sha256'] and sha(INVENTORY)==report['inventory_sha256']
    pairs=[json.loads(s) for s in (HERE/'candidate_pairs.jsonl').read_text().splitlines()]
    source=[json.loads(s) for s in INVENTORY.read_text().splitlines()]
    source={r['member']:r for r in source if r['archive']=='robot-object.zip'}
    assert len(source)==len(pairs)==1826 and sha(HERE/'candidate_pairs.jsonl')==report['pairs_sha256']
    seen=set();keys=set();counts=Counter();subject=Counter();caption_hashes={}
    with zipfile.ZipFile(ZIP) as z:
        assert sum(not i.is_dir() for i in z.infolist())==4912
        for p in pairs:
            member=p['motion_member'];assert member not in seen and member in source;seen.add(member)
            r=source[member]
            assert (p['motion_member_sha256'],p['archive_sha256'],p['nominal_action_key'],p['variant'],p['provisional_group']) == (
                r['member_sha256'],r['archive_sha256'],r['nominal_action_key'],r['variant_name'],r['nominal_source_group'])
            annotation='omomo_text_anno_json_data/'+p['nominal_action_key']+'.json'
            assert p['annotation_member']==annotation
            raw=z.read(annotation);content=json.loads(raw.decode('utf8'))
            assert list(content)==[p['nominal_action_key']] and content[p['nominal_action_key']]==p['caption']
            assert hashlib.sha256(raw).hexdigest()==p['annotation_member_sha256']
            assert p['pair_status']=='filename_candidate_only' and p['timestamp_status']=='not_provided'
            assert p['semantic_status']=='unreviewed' and p['split']=='unassigned' and p['training_ready'] is False
            assert caption_hashes.setdefault(p['nominal_action_key'],p['annotation_member_sha256'])==p['annotation_member_sha256']
            keys.add(p['nominal_action_key']);subject[p['provisional_group']]+=1
            counts['original' if p['variant']=='original' else 'augmented']+=1
    assert seen==set(source) and len(keys)==312 and counts=={'original':312,'augmented':1514}
    assert report['matched_nominal_action_keys']==312 and report['training_ready'] is False
    result={'git_blob_sha1':blob,'annotation_zip_sha256':sha(ZIP),'inventory_sha256':sha(INVENTORY),
            'pair_manifest_sha256':sha(HERE/'candidate_pairs.jsonl'),'total_pairs':len(pairs),
            'nominal_action_keys':len(keys),'variant_counts':dict(counts),'groups':dict(subject),
            'training_ready':False,'limitations':['Independent parse and source hashes; no independent motion identity or visual semantic review.',
                'Original text contains no frame times; association with each temporal window and transformed variant is unverified.',
                'No definitive data/license-chain, physical, 29-to-27 or overall training acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
