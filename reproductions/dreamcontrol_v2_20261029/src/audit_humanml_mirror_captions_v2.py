"""Bounded literal mirror and raw time-tag audit; never modifies corpus labels."""
import hashlib, json, math, re, zipfile
from collections import Counter
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PUBLIC=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
OUT=Path(__file__).resolve().parent
MAPPING=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
PAT=re.compile(r'\b(counterclockwise|clockwise|left|right)\b',re.I)
SWAP={'left':'right','right':'left','clockwise':'counterclockwise','counterclockwise':'clockwise'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def swap(text):
    def repl(m):
        old=m.group(); new=SWAP[old.lower()]
        return new.upper() if old.isupper() else new.capitalize() if old.istitle() else new
    return PAT.sub(repl,text)

def parsed(tag):
    try:
        n=float(tag)
        return ('nan',None) if math.isnan(n) else ('inf',None) if math.isinf(n) else ('finite',n)
    except ValueError:return ('unparseable',None)

def main():
    expected='358875ac6aca9e976086778fe317f27f64134e1fafaff51581817c09f22da778'
    if sha(MAPPING)!=expected:raise ValueError('mapping changed')
    archive=PUBLIC/'texts.zip'; archive_hash=sha(archive)
    rows=[json.loads(s) for s in MAPPING.read_text().splitlines()]
    ids={r['motion_id'] for r in rows}; originals=sorted(r['motion_id'] for r in rows if not r['mirrored'])
    if len(originals)!=14614 or len(ids)!=29228 or any('M'+i not in ids for i in originals):raise ValueError('coverage')
    c=Counter(); reviews=[]
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate members')
        for ident in originals:
            a=z.read('texts/'+ident+'.txt').decode('utf8').splitlines()
            b=z.read('texts/M'+ident+'.txt').decode('utf8').splitlines()
            if len(a)!=len(b):c['line_count_mismatch_pairs']+=1
            any_bad=False
            for index,(x,y) in enumerate(zip(a,b),1):
                c['paired_lines']+=1; p=x.split('#'); q=y.split('#')
                if len(p)!=4 or len(q)!=4:c['malformed_lines']+=1;continue
                c['left_right_caption_lines']+=bool(re.search(r'\b(left|right)\b',p[0],re.I))
                c['clockwise_caption_lines']+=bool(re.search(r'\b(counterclockwise|clockwise)\b',p[0],re.I))
                cap=swap(p[0])==q[0]; tok=swap(p[1])==q[1]
                c['caption_literal_mismatches']+=not cap; c['token_literal_mismatches']+=not tok
                any_bad |= not cap
                tagbad=False
                for col in (2,3):
                    aa,bb=parsed(p[col]),parsed(q[col]); c['tag_positions']+=1
                    if aa[0]!='finite' or bb[0]!='finite':c['nonfinite_tag_positions']+=1
                    if aa[0]=='nan' and bb[0]=='nan':c['paired_nan_tag_positions']+=1
                    if aa!=bb:c['tag_value_or_kind_mismatch_positions']+=1;tagbad=True
                    if aa[0] in ('inf','unparseable') or bb[0] in ('inf','unparseable'):c['inf_or_unparseable_tag_positions']+=1
                    norm=lambda item: 0.0 if item[0]=='nan' else item[1]
                    if norm(aa)!=norm(bb):c['nan_zero_normalized_mismatch_positions']+=1
                if not cap or not tok or tagbad:
                    reviews.append({'id':ident,'line':index,'caption':p[0],'mirror_caption':q[0],
                        'caption_matches_literal_swap':cap,'tokens_match_literal_swap':tok,
                        'raw_tags':p[2:4],'mirror_raw_tags':q[2:4]})
            c['pairs']+=1;c['pairs_with_caption_literal_mismatch']+=any_bad
    if sha(archive)!=archive_hash or sha(MAPPING)!=expected:raise ValueError('inputs changed')
    with (OUT/'review_rows.jsonl').open('x',encoding='utf8',newline='\n') as f:
        for row in reviews:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    out={'summary':dict(c),'mapping_sha256':expected,'text_archive_sha256':archive_hash,
         'script_sha256':sha(__file__),'review_rows_sha256':sha(OUT/'review_rows.jsonl'),
         'limitations':'Literal whole-word left/right and clockwise/counterclockwise substitution only; mismatch is not necessarily wrong and a match is not semantic validation. NaN-to-zero counts mimic inspected legacy loader only, not validation of its policy. No label changes; source times and physical FPS remain unverified.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out,sort_keys=True))

if __name__=='__main__':main()
