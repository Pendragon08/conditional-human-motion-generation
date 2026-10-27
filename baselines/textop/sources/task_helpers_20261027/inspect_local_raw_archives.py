import pathlib,tarfile,json
base=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives')
for group in ['ACCAD','BMLmovi','EyesJapanDataset']:
    p=base/(group+'_smplx_g.tar.bz2');sample=[]
    with tarfile.open(p,'r|bz2') as archive:
        for member in archive:
            if member.name.endswith('_stageii.npz'):sample.append(member.name)
            if len(sample)==3:break
    print(json.dumps({'group':group,'bytes':p.stat().st_size,'sample':sample}))
