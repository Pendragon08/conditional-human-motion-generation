import pathlib,urllib.request,urllib.parse,json,time
OUT=pathlib.Path('D:/具身智能/TextOp复现_20261024');REV='bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
url=f'https://huggingface.co/api/datasets/Yochish/TextOp-Data/tree/{REV}/TextOpTracker/artifacts?recursive=true&expand=false&limit=1000'
files=[]
while url:
    with opener.open(url,timeout=60) as r:
        files.extend(x for x in json.load(r) if x['type']=='file');link=r.headers.get('Link','')
    url=next((x.split('>')[0].strip('< ') for x in link.split(',') if 'rel="next"' in x),None)
    print(len(files),sum(x['size'] for x in files),flush=True)
    (OUT/'tracker_asset_manifest.json').write_text(json.dumps({'revision':REV,'files':files,'bytes':sum(x['size'] for x in files),'complete':not bool(url)},indent=2))
