"""Conditional lookup in a fixed PHC issue list without executing its pickle."""
import hashlib
import json
from pathlib import Path
from phc_joblib_static_parser import parse

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'phc_filter_parser_v2_20261031/source.pkl'
MANIFEST=ROOT/'four_semantic_subset_20261101/manifest.json'
PIN='2c336550d452e5769bc12d199c285bb3717af651f3691ee18ac6b82f2de0a3d8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PIN:raise ValueError('PHC file hash changed')
    table,stats=parse(raw)
    if len(table)!=2845 or stats['numeric_arrays']!=1058:raise ValueError('unexpected parse results')
    records=json.loads(MANIFEST.read_text())['records'];cases=[]
    for row in records:
        name=row['source_id']
        if name.startswith('./pose_data/'):relative=name[len('./pose_data/'):]
        elif name.startswith('pose_data/'):relative=name[len('pose_data/'):]
        else:raise ValueError('unsupported source path')
        if not relative.endswith('.npy'):raise ValueError('unexpected source suffix')
        candidate='0-'+relative[:-4].replace('/','_')
        variants=sorted({candidate,candidate.replace(' ','_')})
        matches=[{'key':key,'entry':table[key]} for key in variants if key in table]
        cases.append({'sample_id':row['sample_id'],'source_id':name,'conditional_keys':variants,
                      'matches':matches,'status':'listed' if matches else 'not_listed_under_these_keys'})
    out={'source_sha256':PIN,'manifest_sha256':sha(MANIFEST),'parser_sha256':sha(Path(__import__('phc_joblib_static_parser').__file__)),
         'script_sha256':sha(Path(__file__)),'list_entries':len(table),'parser_stats':stats,'cases':cases,'training_ready':False,
         'limitations':['Conditional name mapping; exact source identity is not proven.',
                        'Absence is not proof of feasibility; version actually used in v2 is unknown.',
                        'The PHC conversion assigns SSM to test and MoSh to validation; not the v2 split.',
                        'No true timing, caption, license, contact or training acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
