"""Trace two conservative, human-readable mirror caption anomalies to source ZIP."""
import hashlib,json,re,zipfile
from collections import Counter
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'humanml_mirror_caption_audit_v2_20261030'
TEXT=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    report=json.loads((BASE/'report.json').read_text()); assert sha(TEXT)==report['text_archive_sha256'] and sha(BASE/'review_rows.jsonl')==report['review_rows_sha256']
    rows=[json.loads(s) for s in (BASE/'review_rows.jsonl').read_text(encoding='utf8').splitlines()]
    c=Counter(); examples=[];matched=[]
    with zipfile.ZipFile(TEXT) as z:
        for row in rows:
            if row['caption_matches_literal_swap']:continue
            a=row['caption'].lower(); b=row['mirror_caption'].lower();categories=[]
            if re.search(r'\bupright\b',a) and re.search(r'\bupleft\b',b):categories.append('upright_to_upleft')
            if (re.search(r'\bleft-to-right\b',a) and re.search(r'\bleft-to-left\b',b)) or (re.search(r'\bright-to-left\b',a) and re.search(r'\bright-to-right\b',b)):
                categories.append('hyphenated_direction_collapse')
            if not categories:continue
            orig=z.read('texts/'+row['id']+'.txt').decode('utf8').splitlines()[row['line']-1].split('#')[0]
            mirror=z.read('texts/M'+row['id']+'.txt').decode('utf8').splitlines()[row['line']-1].split('#')[0]
            if (orig,mirror)!=(row['caption'],row['mirror_caption']):raise ValueError('source mismatch')
            for category in categories:c[category]+=1
            matched.append({'id':row['id'],'line':row['line'],'categories':categories})
            if len(examples)<12:examples.append({'id':row['id'],'line':row['line'],'categories':categories,'original':orig,'mirror':mirror})
    out={'counts':dict(c),'matched_line_count':len(matched),'examples':examples,
         'source_zip_sha256':sha(TEXT),'input_report_sha256':sha(BASE/'report.json'),'input_review_sha256':sha(BASE/'review_rows.jsonl'),
         'audit_script_sha256':sha(Path(__file__)),
         'limitations':'Two narrow literal patterns only, counts are not all semantic errors; lexical examples do not demonstrate motion or tag correctness. No labels changed or applied to training.'}
    with (OUT/'matches.jsonl').open('x',encoding='utf8') as f:
        for row in matched:f.write(json.dumps(row)+'\n')
    out['matches_sha256']=sha(OUT/'matches.jsonl')
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'counts':dict(c),'matched':len(matched)}))
if __name__=='__main__':main()
