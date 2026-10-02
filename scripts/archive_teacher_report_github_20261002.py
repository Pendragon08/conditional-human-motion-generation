"""Archive an offline report with linked evidence; omit licensed arrays/weights."""
import hashlib, json, shutil, re
from collections import deque, Counter
from pathlib import Path
from urllib.parse import unquote, urlparse, quote
from bs4 import BeautifulSoup

REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
DEST=REPO/'reports/teacher_report_20261002'
SOURCE=ROOT/'实验结果汇报.html'
EXCLUDED={'.npy','.npz','.pkl','.pt','.pth','.onnx','.ckpt','.usd','.usda','.usdc','.urdf','.mjcf'}
DEST.mkdir(parents=True,exist_ok=True)
pending=deque([SOURCE]);seen=set();omitted={};changed=[]
pending.extend(sorted(p for p in (ROOT/'论文复现汇报资料_20261002/PPO原理与策略导出').rglob('*') if p.is_file()))
records=[]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

while pending:
 p=pending.popleft()
 if p in seen:continue
 seen.add(p)
 rel=p.relative_to(ROOT)
 if p.suffix.lower() in EXCLUDED:
  omitted[rel.as_posix()]={'reason':'Raw motion/model/robot asset; no redistribution in this report archive.','bytes':p.stat().st_size,'sha256':sha(p)}
  continue
 if p.suffix.lower()=='.pdf':
  omitted[rel.as_posix()]={'reason':'Paper linked at its official source instead of redistributed.','bytes':p.stat().st_size,'sha256':sha(p)}
  continue
 target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix.lower()=='.html':
  soup=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
  # Companion galleries create video tags from embedded JSON at browser runtime.
  for script in soup.find_all('script'):
   match=re.search(r'const clips\s*=\s*(\[.*?\]);',script.get_text(),re.S)
   if match:
    for clip in json.loads(match.group(1)):
     q=Path(__import__('os').path.abspath(p.parent/unquote(clip['file'])))
     if not q.is_relative_to(ROOT) or not q.is_file():raise ValueError('Invalid gallery video: '+str(q))
     pending.append(q)
  for tag in soup.select('[href],[src],[poster]'):
   for attr in ['href','src','poster']:
    val=tag.get(attr)
    if not val:continue
    url=urlparse(val)
    if url.scheme or not url.path:continue
    # Collapse traversal lexically but preserve symlinks for the requested local artifacts.
    q=Path(__import__('os').path.abspath(p.parent/unquote(url.path)))
    if not q.is_relative_to(ROOT):raise ValueError('Unexpected link outside report folder: '+str(q))
    if not q.exists():raise FileNotFoundError(q)
    if q.is_dir():pending.extend(sorted(x for x in q.rglob('*') if x.is_file()))
    elif q.suffix.lower() in EXCLUDED or q.suffix.lower()=='.pdf':
     pending.append(q)
     if q.suffix.lower()=='.pdf':
      tag[attr]='https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf'
     else:
      relative=__import__('os').path.relpath(DEST/'原始资产保存位置.html',target.parent).replace('\\','/')
      tag[attr]=relative+'#'+quote(q.relative_to(ROOT).as_posix(),safe='')
      if tag.name=='a':tag.append('（本地原始资产）')
     changed.append({'page':rel.as_posix(),'original_link':val,'archive_link':tag[attr]})
    else:pending.append(q)
  if p==SOURCE:
   note=soup.new_tag('div',attrs={'class':'note','id':'github-archive-note'})
   note.string='GitHub汇报归档版：正文、视频、训练曲线与结果来自同一份本地报告。原始动作数组、受许可模型/机器人资产不在本报告包中；对应链接说明本地位置。论文PDF链接作者官方来源。实验快照不代表服务器最新实时进度。'
   (soup.find(id='overview') or soup.find('main')).append(note)
  target.write_text(str(soup),encoding='utf-8',newline='\n')
 else:shutil.copy2(p,target)
 records.append({'path':rel.as_posix(),'bytes':target.stat().st_size,'source_sha256':sha(p),'archived_sha256':sha(target)})

# Keep the exact source document separately as a traceable snapshot, not the portable entry page.
exact=DEST/'local_source/实验结果汇报.html';exact.parent.mkdir(exist_ok=True);shutil.copy2(SOURCE,exact)
report_sources=[]
for paper in ['DreamControl_实验结果汇报.html','CLoSD_实验结果汇报.html']:
 p=ROOT/paper
 if p.exists():
  shutil.copy2(p,DEST/'local_source'/paper)
  report_sources.append({'file':paper,'source_sha256':sha(p),'exact_source_copy':'local_source/'+paper})
rows=''.join('<tr><td>'+__import__('html').escape(rel)+'</td><td>'+__import__('html').escape(str(ROOT/rel))+'</td></tr>' for rel in omitted)
(DEST/'原始资产保存位置.html').write_text('<!doctype html><meta charset="utf-8"><title>原始资产保存位置</title><style>body{font:16px system-ui;line-height:1.7;max-width:1100px;margin:40px auto;padding:20px}td,th{border-bottom:1px solid #ddd;padding:14px;overflow-wrap:anywhere}table{width:100%;table-layout:fixed}</style><h1>原始资产保存位置</h1><p>本GitHub归档保留视频、标量日志、质量审计和结果；原始动作数组及受许可模型、机器人资产没有上传。它们在本地汇报目录或服务器中保留，路径与哈希可追溯。这不表示重新下载数据的许可已自动取得。</p><p><a href="实验结果汇报.html">返回汇报报告</a></p><table><tr><th>资产</th><th>已有本地位置</th></tr>'+rows+'</table>',encoding='utf-8')
manifest={'source_report_path':str(SOURCE),'source_report_sha256':sha(SOURCE),'exact_source_copy':'local_source/实验结果汇报.html','portable_entry':'实验结果汇报.html','files':records,'excluded_assets':omitted,'adjusted_links':changed,'report_content_changed':'Only archive availability note and unavailable raw-asset/paper links; no results changed.','total_bytes':sum(r['bytes'] for r in records)}
manifest['report_sources']=report_sources
(DEST/'bundle_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
(DEST/'.gitattributes').write_text('* -text -whitespace\n*.csv -diff\n*.0 -diff\n',encoding='utf-8',newline='\n')
(DEST/'README.md').write_text('''# 条件人体运动生成：实验结果汇报归档

下载仓库后，用浏览器打开本目录的[实验结果汇报.html](实验结果汇报.html)。GitHub文件页用于保存源码，不会直接播放完整HTML；本目录的相对链接用于下载后的离线观看。

两篇论文已拆分为独立报告：[DreamControl实验报告](DreamControl_实验结果汇报.html)与[CLoSD实验报告](CLoSD_实验结果汇报.html)。原文件名现在是入口页；两份报告共享媒体和证据目录，CLoSD未录制的官方评估视频仍明确标记未交付。

报告包括条件生成说明、六类原始人体视频与质量代理审计、机器人执行及MuJoCo迁移视频、PPO训练曲线/优化原理、冻结文件格式、正式成绩和失败边界。此前用户要求暂不展示创新点实验，当前汇报正文保持这一范围。

正文、视频、标量日志和审计来自本地同一版本；原始数组、受许可模型/机器人资产没有上传，相关入口转为本地位置说明。论文PDF使用作者官网链接。原始本地HTML的逐字节副本保存在`local_source/实验结果汇报.html`，它仅用于追溯；离线观看用本目录顶层入口。完整来源、文件哈希、改写链接和排除记录见`bundle_manifest.json`。

本地原件仍在`D:\\具身智能\\通用条件人体运动生成_已完成实验结果_20260929\\实验结果汇报.html`。本脚本没有修改本地原件或服务器训练；本目录是已归档的实验快照，不是实时状态页。
''',encoding='utf-8',newline='\n')
print(json.dumps({'files':len(records),'bytes':manifest['total_bytes'],'excluded':list(omitted),'source_sha256':manifest['source_report_sha256'],'html_pages':sum(r['path'].endswith('.html') for r in records)},ensure_ascii=False))
