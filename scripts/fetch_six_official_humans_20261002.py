from pathlib import Path
import urllib.request,hashlib,json,concurrent.futures
import numpy as np
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')/'人体生成视频'/'六任务官方原始样本'
ROOT.mkdir(parents=True,exist_ok=True)
COMMIT='e1ef719d6702d15a5ab211db0e55be438b2d1c88'
ITEMS=[('Kick','98d40fae4e8436ee5018ff83fbb589a176f81dd025faf92ee5a8eb2296436396',10350449),('Punch','1adeb62cb26511515df348cb0c4126e0d514a800ee4781877ff4864c7e724c44',10350463),('Jump','d51e5ca478053fcc66e328dff89f5aa4202cadf7623f0fcb02436ac1be94f462',1346097),('Sit','c270f5252e160d3883288d10bbdd3eb11c1d40901a16fffebf8ede282bb3903e',2070603),('Pick','696637ac3f7fa6f649479feb75a2b6b86af7fe28cc1c64de576e7ca84b5e8aaf',10350463),('Bimanual_Pick','dca32b664f648edbbe19ea1c48974503df1f7efb080e20767fd9b6e194979a89',932126)]
def fetch(item):
    task,digest,size=item;url=f'https://media.githubusercontent.com/media/GenRobo/DreamControl/main/TrajGen/sample/{task}_sim/results.npy'
    path=ROOT/f'{task}_official_results.npy'
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        with urllib.request.urlopen(url,timeout=120) as response:path.write_bytes(response.read())
    assert path.stat().st_size==size and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    data=np.load(path,allow_pickle=True).item()
    row={'task':task,'file':path.name,'url':url,'pointer_checkout_commit':COMMIT,'sha256':digest,'bytes':size,'motion_shape':list(data['motion'].shape),'lengths':np.asarray(data['lengths']).tolist(),'texts':list(map(str,data['text'])),'keys':list(data),'finite':bool(np.isfinite(data['motion']).all()),'source_type':'official released human generation, not new local diffusion inference'}
    print(json.dumps({k:v for k,v in row.items() if k not in ['texts','lengths']})+' texts='+repr(row['texts'][:2])+' lengths='+repr(row['lengths'][:3]),flush=True)
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,ITEMS))
(ROOT/'source_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
