"""Read public upstream commit and LFS pointers; never modify original arrays."""
from pathlib import Path
import datetime,json,re,urllib.request,hashlib
R=Path(r'D:\具身智能\conditional-human-motion-generation')
S=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929\人体生成视频\六任务官方原始样本')
def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'conditional-motion-source-audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=20) as response:return response.read()
ref=json.loads(get('https://api.github.com/repos/GenRobo/DreamControl/git/ref/heads/main'))
commit=ref['object']['sha'];rows=[]
for entry in json.loads((S/'source_manifest.json').read_text(encoding='utf-8')):
    path=entry['url'].split('/main/',1)[1]
    url=f'https://raw.githubusercontent.com/GenRobo/DreamControl/{commit}/{path}'
    raw=get(url);pointer=raw.decode('ascii')
    oid=re.search(r'^oid sha256:([0-9a-f]{64})$',pointer,re.M).group(1)
    size=int(re.search(r'^size (\d+)$',pointer,re.M).group(1))
    assert oid==entry['sha256'] and size==entry['bytes']
    assert hashlib.sha256((S/entry['file']).read_bytes()).hexdigest()==oid
    rows.append({'task':entry['task'],'official_upstream_commit':commit,'repository_path':path,'git_lfs_oid':'sha256:'+oid,'bytes':size,'immutable_pointer_url':url,'pointer_text':pointer,'local_source_file':entry['file']})
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_repository':'https://github.com/GenRobo/DreamControl','official_main_commit_observed':commit,'all_official_pointers_match_local_arrays':True,'original_source_manifest_unchanged':True,'entries':rows}
dest=R/'results/human_generation_quality_20261002/source_identity.json'
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'commit':commit,'verified_lfs_pointers':len(rows)}))
