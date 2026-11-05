"""Directly compare all original/mirror ZIP members against flagged example set."""
import hashlib,json,re,zipfile
from pathlib import Path
root=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
mapping=root/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
archive=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
out=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    previous=json.loads((out/'report.json').read_text())
    assert sha(archive)==previous['source_zip_sha256'] and sha(out/'matches.jsonl')==previous['matches_sha256']
    expected={(r['id'],r['line'],tuple(r['categories'])) for r in map(json.loads,(out/'matches.jsonl').read_text().splitlines())}
    ids=[r['motion_id'] for r in map(json.loads,mapping.read_text().splitlines()) if not r['mirrored']]; observed=set()
    with zipfile.ZipFile(archive) as z:
        for mid in ids:
            aa=z.read('texts/'+mid+'.txt').decode().splitlines();bb=z.read('texts/M'+mid+'.txt').decode().splitlines();assert len(aa)==len(bb)
            for i,(a,b) in enumerate(zip(aa,bb),1):
                a=a.split('#')[0].lower();b=b.split('#')[0].lower(); kinds=[]
                if 'upright' in re.findall(r'\b\w+\b',a) and 'upleft' in re.findall(r'\b\w+\b',b):kinds.append('upright_to_upleft')
                if ('left-to-right' in a and 'left-to-left' in b) or ('right-to-left' in a and 'right-to-right' in b):kinds.append('hyphenated_direction_collapse')
                if kinds:observed.add((mid,i,tuple(kinds)))
    assert len(ids)==14614 and observed==expected,(len(observed),len(expected))
    result={'pairs_checked':len(ids),'matched_line_count':len(observed),'source_zip_sha256':sha(archive),'mapping_sha256':sha(mapping),'primary_report_sha256':sha(out/'report.json'),'matches_sha256':sha(out/'matches.jsonl'),'audit_script_sha256':sha(Path(__file__)),'limitations':'Two bounded patterns, not overall text semantic quality; no inference about actual source motion.'}
    with (out/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
