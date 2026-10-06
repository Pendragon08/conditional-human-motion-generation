"""Filesystem-only artifact audit: links, byte hashes, paper scopes and video provenance."""
import hashlib,json,re
from pathlib import Path
from urllib.parse import unquote,urlparse
from bs4 import BeautifulSoup
root=Path(r'D:\具身智能\conditional-human-motion-generation\reports\teacher_report_20261002')
manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
links=0;html_count=0
for record in manifest['files']:
    p=root/record['path'];assert p.is_file() and sha(p)==record['archived_sha256'],str(p)
    assert 'private_scene' not in p.name and p.name!='actual_poses.json.gz' and p.suffix.lower() not in {'.pt','.pth','.urdf','.mjcf','.usd'},str(p)
    if p.suffix=='.html':
        html_count+=1;soup=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
        for tag in soup.select('[href],[src],[poster]'):
            for attr in ['href','src','poster']:
                u=urlparse(tag.get(attr,''))
                if u.scheme or not u.path:continue
                q=(p.parent/unquote(u.path)).resolve();assert q.is_relative_to(root) and q.exists(),str(q);links+=1
for source in manifest['report_sources']:assert sha(root/source['exact_source_copy'])==source['source_sha256']
assert sha(root/'local_source/实验结果汇报.html')==manifest['source_report_sha256']
closd=BeautifulSoup((root/'CLoSD_实验结果汇报.html').read_text(encoding='utf-8'),'html.parser')
assert len(closd.select('#closd table'))==2 and not closd.find(id='baseline')
assert len(closd.find_all('video'))==9
for task in ['REACH','STRIKE_KICK','STRIKE_PUNCH','SIT','GET_UP']:assert closd.find(id='closd-video-'+task)
assert '画面不动或打不开' in closd.get_text() and '尚未启动' in closd.get_text()
dream=BeautifulSoup((root/'DreamControl_实验结果汇报.html').read_text(encoding='utf-8'),'html.parser')
assert len(dream.select('#generation video'))==18 and len(dream.select('#videos video'))==32
assert len(dream.select('#training details.training-chart'))==11
summary=json.loads((root/'CLoSD补充结果_20261006/证据/summary.json').read_text(encoding='utf-8'))
for model in summary['models']:
    for clip in model['clips']:
        p=root/'CLoSD补充结果_20261006/视频'/clip['file'];assert sha(p)==clip['sha256']
        assert clip['video_verified'] and clip['full_decode_exit_code']==0
assert summary['total_videos']==9 and summary['original4096_evaluation_video_verified'] is False
result={'kind':'filesystem_artifact_audit_not_browser_ui','html_pages':html_count,'local_links_checked':links,
 'archive_file_hashes_checked':len(manifest['files']),'missing_files':0,'licensed_raw_assets_included':False,
 'closd_videos':9,'source_video_full_decode_verified':True,'original4096_evaluation_video_verified':False,
 'formal3072_started':False,'live_gate_passed':False,'dreamcontrol_report_preserved':True}
(root/'archive_validation_20261006.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
