"""Rank short annotated candidate clips within previously isolated source groups.

No text meaning or source identity is accepted; reports directional conflicts
for review. Clip lengths refer to HumanML nominal index samples only.
"""
import hashlib
import json
import re
from pathlib import Path
import zipfile

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    selection=ROOT/'independent_amass_selection_v2_20261031/candidates.json'
    chosen=json.loads(selection.read_text()); inventory=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
    if sha(inventory)!=chosen['mapping_sha256']:raise ValueError('inventory changed')
    parents={r['parent_group'] for r in chosen['selected']};groups={p:[] for p in parents}
    from pathlib import PurePosixPath
    for line in inventory.read_text().splitlines():
        row=json.loads(line);parent=str(PurePosixPath(row['source_path']).parent)
        if parent not in parents or row['mirrored'] or row['official_split']!='train' or not row['candidate_local_path']:continue
        start,end=int(row['index_start_20']),int(row['index_end_20']); n=end-start
        if 20<=n<=90:groups[parent].append(row)
    found=[]
    with zipfile.ZipFile(HUMANML/'texts.zip') as z:
        for parent,items in sorted(groups.items()):
            entries=[]
            for r in items:
                raw=z.read('texts/'+r['motion_id']+'.txt')
                parts=[line.split('#') for line in raw.decode('utf8').splitlines()]
                whole=[p[0] for p in parts if len(p)==4 and p[2:]==['0.0','0.0'] and p[0]]
                if not whole:continue
                directions={w for text in whole for w in re.findall(r'\b(?:left|right)\b',text.lower())}
                entries.append({'id':r['motion_id'],'source_path':r['source_path'],
                                'candidate_path':r['candidate_local_path'],'start':int(r['index_start_20']),
                                'end':int(r['index_end_20']), 'captions':whole,
                                'caption_member_sha256':hashlib.sha256(raw).hexdigest(),
                                'contains_both_direction_words':directions=={'left','right'}})
            entries.sort(key=lambda r:(r['contains_both_direction_words'],r['end']-r['start'],r['id']))
            found.extend({'parent_group':parent,**r} for r in entries[:5])
    out={'selection_sha256':sha(selection),'mapping_sha256':sha(inventory),'candidates':found,
         'training_ready':False,'limitations':['Keyword heuristic is not a semantic check.',
              'Source paths, subject independence, physical time and model identity remain conditional.',
              'Whole-caption applicability must be independently assessed.']}
    with (HERE/'ranked.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'candidate_count':len(found),'parents':len({r['parent_group'] for r in found})}),flush=True)


if __name__=='__main__':main()
