import concurrent.futures, json, urllib.request, urllib.parse, pathlib, time
OUT=pathlib.Path('D:/具身智能/TextOp复现_20261024')
REV='bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9'
PREFIXES=['TextOpRobotMDAR/dataset','TextOpRobotMDAR/description','TextOpRobotMDAR/logs','TextOpTracker/logs']
def get(prefix):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
    url=f'https://huggingface.co/api/datasets/Yochish/TextOp-Data/tree/{REV}/{prefix}?recursive=true&expand=false&limit=1000'
    files=[]
    while url:
        with opener.open(url,timeout=60) as r:
            files.extend(x for x in json.load(r) if x['type']=='file')
            link=r.headers.get('Link','')
        url=next((x.split('>')[0].strip('< ') for x in link.split(',') if 'rel="next"' in x),None)
    return prefix,files
allfiles=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for prefix,files in pool.map(get,PREFIXES):
        allfiles.extend(files)
        print(prefix,len(files),sum(x['size'] for x in files),flush=True)
doc={'repo':'Yochish/TextOp-Data','revision':REV,'files':allfiles,'total_bytes':sum(x['size'] for x in allfiles)}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'hf_asset_manifest.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
print('TOTAL',doc['total_bytes'],flush=True)
