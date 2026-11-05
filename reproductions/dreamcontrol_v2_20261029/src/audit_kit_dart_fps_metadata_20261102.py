"""Compare DART's published FPS dictionaries to existing KIT inventory.

Name association is not independently verified recording identity. DART's FPS
values are a third-party correction, not automatically raw capture-time truth.
"""
import hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
INV=ROOT/'data/kit_smplx_stream_inventory_20261102/records.jsonl'
DART=ROOT/'data/dart_fps_metadata_20261102'
OUT=ROOT/'reports/kit_dart_fps_comparison_v2_20261102.json'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(path):
    p=path.replace('\\','/').split('/')
    if len(p)!=3 or p[0]!='KIT':raise ValueError('unexpected KIT key')
    name=re.sub(r'(?i)_(poses|stageii)\.(npy|npz)$','',p[2])
    return p[1],re.sub(r'[^a-z0-9]','',name.casefold())
def main():
    metadata=json.loads((DART/'report.json').read_text());all_file=DART/'fps_dict_all.json';subset_file=DART/'fps_dict.json'
    if digest(all_file)!=metadata['files']['fps_dict_all.json']['sha256'] or digest(subset_file)!=metadata['files']['fps_dict.json']['sha256']:
        raise ValueError('DART metadata changed')
    invhash=digest(INV);inventory=[json.loads(x) for x in INV.read_text().splitlines()]
    all_items=json.loads(all_file.read_text());subset=json.loads(subset_file.read_text())
    all_keys=defaultdict(list);sub_keys=defaultdict(list)
    for target,items in ((all_keys,all_items),(sub_keys,subset)):
        for p,fps in items.items():
            # Subset BABEL paths may contain a dataset prefix before KIT.
            parts=p.replace('\\','/').split('/')
            if 'KIT' not in parts:continue
            index=parts.index('KIT'); tail_parts=parts[index:]
            if len(tail_parts)>=2 and tail_parts[1]=='KIT':tail_parts=tail_parts[1:]
            tail='/'.join(tail_parts)
            try:k=key(tail)
            except ValueError:continue
            if not isinstance(fps,(float,int)) or not math.isfinite(fps) or fps<=0:raise ValueError('invalid fps')
            target[k].append((p,float(fps)))
    counts=Counter();fps_counts=Counter();matched_ids=0;disagreements=[];missing=[];ambiguous=[];witness=[]
    invkeys=set()
    for row in inventory:
        k=key(row['member']);invkeys.add(k);hits=all_keys.get(k,[])
        if len(hits)!=1:
            (ambiguous if hits else missing).append(row['member']);continue
        val=hits[0][1];fps_counts[str(val)]+=1;matched_ids+=len(row['nominal_rows']);counts['matched_members']+=1
        if row.get('declared_fps') is not None and float(row['declared_fps'])!=val:counts['declared_header_disagrees']+=1
        sub=sub_keys.get(k,[])
        if sub:
            counts['matched_subset_members']+=1
            if any(v!=val for _,v in sub):disagreements.append({'member':row['member'],'all_fps':val,'subset':sub})
        if len(witness)<12:witness.append({'member':row['member'],'header_fps':row.get('declared_fps'),'dart_key':hits[0][0],'dart_fps':val})
    extra=[p for k,values in all_keys.items() if k not in invkeys for p,_ in values]
    result={'source_inventory_sha256':invhash,'dart_all_sha256':digest(all_file),'dart_subset_sha256':digest(subset_file),
        'script_sha256':digest(Path(__file__)),'inventory_members':len(inventory),'dart_kit_keys':len(all_keys),
        'matched_members':counts['matched_members'],'matched_nominal_ids':matched_ids,'matched_fps_counts':dict(fps_counts),
        'header_disagrees_count':counts['declared_header_disagrees'],'missing_members':missing,'ambiguous_members':ambiguous,
        'dart_only_keys':extra,'matched_subset_members':counts['matched_subset_members'],'dictionary_fps_disagreements':disagreements,
        'witness':witness,'training_ready':False,
        'limitations':['Name normalization is a candidate association, not proof of same recording.',
           'DART FPS dictionary is an independent project claim; generation methodology and raw capture time are not verified.',
           'Existing 33 oversize members lack header in this original inventory; count does not assume their header.',
           'No source fps, data, training flags, or original files were changed.']}
    if digest(INV)!=invhash:raise RuntimeError('inventory changed')
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing_members','ambiguous_members','dart_only_keys','dictionary_fps_disagreements','witness')},ensure_ascii=False))
    print(json.dumps({'missing_count':len(missing),'ambiguous_count':len(ambiguous),'extra_count':len(extra),'dictionary_disagreement_count':len(disagreements)}))
if __name__=='__main__':main()
