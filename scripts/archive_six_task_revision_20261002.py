from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse,unquote
import json,hashlib,shutil,datetime
OUT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929');R=Path(r'D:\具身智能\conditional-human-motion-generation');PROOF=OUT/'论文复现汇报资料_20261002/生成与验收'
report=OUT/'实验结果汇报.html';s=BeautifulSoup(report.read_text(encoding='utf8'),'html.parser')
bad=[];links=0
for tag in s.select('[href],[src],[poster]'):
 for a in ['href','src','poster']:
  val=tag.get(a)
  if not val:continue
  u=urlparse(val)
  if u.scheme or not u.path:continue
  links+=1;p=(OUT/unquote(u.path)).resolve()
  if not p.exists():bad.append(str(p))
assert not bad,bad
files=[]
for p in (OUT/'人体生成视频/六任务官方原始样本').rglob('*'):
 if p.is_file():files.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for name in ['fetch_six_official_humans_20261002.py','render_six_official_humans_20261002.py','revise_six_task_presentation_20261002.py','check_six_task_presentation_20261002.cjs','archive_six_task_revision_20261002.py']:
 shutil.copy2(R/'scripts'/name,PROOF/name)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'local_links_checked':links,'missing':bad,'new_official_human_video_count':18,'total_main_videos':len(s.find_all('video')),'six_tasks':[v['data-human-task'] for v in s.select('.human-preview')],'files':files,'supersedes_main_report_hash_in':'完整媒体与训练产物最终核验.json (previous revision preserved)'}
(PROOF/'六任务展示最终文件核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
(OUT/'打开汇报说明.txt').write_text('打开实验结果汇报.html。原始人体生成部分已覆盖Kick、Punch、Jump、Sit、Pick、Bimanual-Pick六类官方人体生成样本，各固定展示3段，均9.8秒。六类作者原始数组共342条，保存在人体生成视频/六任务官方原始样本。另保留本课题此前生成的200条Kick完整视频和独立几何代理筛查。官方发布资产与本课题新推理分开统计。机器人执行视频、PPO训练曲线及全部原始证据保留；复制整个汇报文件夹可离线使用。sim-to-sim章节含DreamControl原文和官方MuJoCo预测试说明的来源；本课题扩展测试单独标注。\n',encoding='utf8')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False))
