import pathlib,json,collections
base=pathlib.Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass')
groups=collections.Counter();sample=[]
for p in base.rglob('*_stageii.npz'):
    group=next((x for x in p.parts if x in ['ACCAD','BMLmovi','EyesJapanDataset','Eyes_Japan_Dataset','EKUT','MPI_HDM05']),None)
    groups[str(group)]+=1
    if group=='ACCAD' and len(sample)<18:sample.append(str(p.relative_to(base)))
d=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025')
missing=[]
for split in ['train','val']:
    for line in (d/(split+'_sequence_manifest.private.jsonl')).read_text().splitlines():
        x=json.loads(line)
        if x['canonical_source'].startswith('ACCAD/') and not x['raw_stageii_candidates'] and len(missing)<18:missing.append(x['canonical_source'])
print(json.dumps({'raw_counts':dict(groups),'raw_examples':sample,'missing_examples':missing},indent=2))
