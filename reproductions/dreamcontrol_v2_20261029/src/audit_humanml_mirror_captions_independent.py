"""Independent all-lines audit of v2 counts and input integrity."""
import hashlib,json,math,re,zipfile
from collections import Counter
from pathlib import Path

root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
base=root/'humanml_mirror_caption_audit_v2_20261030'
mapping=root/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
archive=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
loader=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen/data_loaders/humanml/data/dataset.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=json.loads((base/'report.json').read_text()); assert sha(mapping)==report['mapping_sha256'] and sha(archive)==report['text_archive_sha256']
    assert sha(base/'audit.py')==report['script_sha256'] and sha(base/'review_rows.jsonl')==report['review_rows_sha256']
    source=loader.read_text(); assert 'f_tag = 0.0 if np.isnan(f_tag) else f_tag' in source and 'to_tag = 0.0 if np.isnan(to_tag) else to_tag' in source
    ids=sorted(r['motion_id'] for r in map(json.loads,mapping.read_text().splitlines()) if not r['mirrored'])
    # Replace whole tokens using placeholders, separately from the primary regex callback.
    words=['counterclockwise','clockwise','left','right']; substitutes=['clockwise','counterclockwise','right','left']
    def invert(s):
        for i,w in enumerate(words):s=re.sub(r'\b'+w+r'\b',lambda m:'__SWAP'+str(i)+('_U' if m.group().isupper() else '_T' if m.group().istitle() else '_L')+'__',s,flags=re.I)
        for i,w in enumerate(substitutes):
            for suffix,converted in [('_U',w.upper()),('_T',w.title()),('_L',w)]:s=s.replace('__SWAP'+str(i)+suffix+'__',converted)
        return s
    c=Counter()
    with zipfile.ZipFile(archive) as z:
        for ident in ids:
            original=z.read('texts/'+ident+'.txt').decode().splitlines();mirror=z.read('texts/M'+ident+'.txt').decode().splitlines()
            assert len(original)==len(mirror)
            bad=False
            for a,b in zip(original,mirror):
                p=a.split('#');q=b.split('#');assert len(p)==len(q)==4;c['paired_lines']+=1
                if invert(p[0])!=q[0]:c['caption_literal_mismatches']+=1;bad=True
                if invert(p[1])!=q[1]:c['token_literal_mismatches']+=1
                for x,y in zip(p[2:4],q[2:4]):
                    x=float(x);y=float(y);c['tag_positions']+=1
                    if not math.isfinite(x) or not math.isfinite(y):c['nonfinite_tag_positions']+=1
                    if math.isnan(x) and math.isnan(y):c['paired_nan_tag_positions']+=1
                    assert (math.isnan(x) and math.isnan(y)) or x==y
            c['pairs']+=1;c['pairs_with_caption_literal_mismatch']+=bad
    for key,value in c.items():assert report['summary'][key]==value,(key,value,report['summary'].get(key))
    result={'independent_counts':dict(c),'primary_report_sha256':sha(base/'report.json'),'legacy_dataset_loader_sha256':sha(loader),'audit_script_sha256':sha(Path(__file__)),'note':'Literal comparison does not validate meaning or true time; paired NaNs are raw labels and legacy loader maps NaN to zero.'}
    with (base/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
if __name__=='__main__':main()
