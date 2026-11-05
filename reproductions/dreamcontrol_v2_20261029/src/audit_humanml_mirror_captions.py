"""Audit official original/mirrored caption pairs without rewriting labels."""
import hashlib,json,math,re,zipfile
from collections import Counter
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PUBLIC=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
OUT=Path(__file__).resolve().parent
MAPPING=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
EXPECTED='358875ac6aca9e976086778fe317f27f64134e1fafaff51581817c09f22da778'
LEFT_RIGHT=re.compile(r'\b(left|right)\b',re.IGNORECASE)
IDIOM=re.compile(r'\b(?:right\s+(?:now|away|after|before|then|here|there)|left\s+(?:behind|over|alone))\b',re.IGNORECASE)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def swap(text):
    def one(m):
        old=m.group(0);new='right' if old.lower()=='left' else 'left'
        return new.upper() if old.isupper() else new.capitalize() if old.istitle() else new
    return LEFT_RIGHT.sub(one,text)

def decode(z,member):
    lines=z.read(member).decode('utf8').splitlines();result=[]
    for line in lines:
        if not line.strip():continue
        parts=line.split('#')
        if len(parts)!=4:result.append({'malformed':True,'raw':line});continue
        try:
            start,end=float(parts[2]),float(parts[3])
            valid=math.isfinite(start) and math.isfinite(end)
        except ValueError:start=end=None;valid=False
        result.append({'caption':parts[0],'tokens':parts[1],'start':start,'end':end,'tags_valid':valid})
    return result

def main():
    if sha(MAPPING)!=EXPECTED:raise ValueError('mapping changed')
    archive=PUBLIC/'texts.zip';archive_hash=sha(archive)
    rows=[json.loads(s) for s in MAPPING.read_text().splitlines()]
    originals=sorted(r['motion_id'] for r in rows if not r['mirrored'])
    allids={r['motion_id'] for r in rows}
    if len(originals)!=14614 or len(allids)!=29228 or any('M'+x not in allids for x in originals):raise ValueError('unexpected ID coverage')
    summary=Counter();details=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate text members')
        for ident in originals:
            orig=decode(z,'texts/'+ident+'.txt');mir=decode(z,'texts/M'+ident+'.txt')
            if len(orig)!=len(mir):
                summary['pairs_with_line_count_mismatch']+=1
                details.append({'motion_id':ident,'kind':'line_count_mismatch','original_count':len(orig),'mirror_count':len(mir)})
            has_direction=False;any_caption=False;any_tokens=False;any_time=False;ambiguous=0
            for i,(a,b) in enumerate(zip(orig,mir),1):
                summary['paired_caption_lines']+=1
                if a.get('malformed') or b.get('malformed'):
                    summary['malformed_line_pairs']+=1;details.append({'motion_id':ident,'line':i,'kind':'malformed'});continue
                if not a['tags_valid'] or not b['tags_valid']:
                    summary['invalid_tag_pairs']+=1;details.append({'motion_id':ident,'line':i,'kind':'invalid_tags'});continue
                direction=bool(LEFT_RIGHT.search(a['caption']) or LEFT_RIGHT.search(a['tokens']))
                has_direction|=direction
                cappass=swap(a['caption'])==b['caption'];tokenpass=swap(a['tokens'])==b['tokens']
                timepass=(a['start'],a['end'])==(b['start'],b['end'])
                any_caption|=not cappass;any_tokens|=not tokenpass;any_time|=not timepass
                if direction:summary['directional_caption_lines']+=1
                if not cappass:summary['caption_text_swap_mismatches']+=1
                if not tokenpass:summary['token_text_swap_mismatches']+=1
                if not timepass:summary['interval_tag_mismatches']+=1
                suspect=bool(IDIOM.search(a['caption']))
                if suspect:summary['possible_idiom_lines']+=1;ambiguous+=1
                if not (cappass and tokenpass and timepass) or suspect:
                    details.append({'motion_id':ident,'line':i,'kind':'review',
                        'original_caption':a['caption'],'mirror_caption':b['caption'],
                        'original_tags':[a['start'],a['end']],'mirror_tags':[b['start'],b['end']],
                        'caption_matches_literal_swap':cappass,'tokens_match_literal_swap':tokenpass,
                        'tags_match':timepass,'possible_idiom':suspect})
            summary['pairs']+=1
            summary['pairs_with_direction']+=has_direction
            summary['pairs_with_caption_swap_mismatch']+=any_caption
            summary['pairs_with_token_swap_mismatch']+=any_tokens
            summary['pairs_with_interval_mismatch']+=any_time
            summary['pairs_with_possible_idiom']+=bool(ambiguous)
    if sha(archive)!=archive_hash or sha(MAPPING)!=EXPECTED:raise ValueError('input changed during scan')
    output=OUT/'review_rows.jsonl'
    with output.open('x',encoding='utf8',newline='\n') as f:
        for r in details:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    result={'script_sha256':sha(__file__),'mapping_sha256':EXPECTED,'text_archive_sha256':archive_hash,
            'review_rows_sha256':sha(output),'summary':dict(summary),
            'limitations':'Mechanical line-order and whole-word left/right swap comparison only; mismatch is not necessarily an annotation error and a matching swap is not semantic validation. Idiom list is incomplete. Timing equality does not establish true physical duration. No text, data, labels or training set was modified.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
