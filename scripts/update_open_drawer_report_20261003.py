"""Update the independent DreamControl report with completed Open-Drawer evidence."""
from pathlib import Path
import copy
import datetime
import hashlib
import json
import re
import shutil
from urllib.parse import quote
from bs4 import BeautifulSoup

R=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
PACK=ROOT/'论文复现汇报资料_20261002'
TASK='Open-Drawer'; VIDEO='r3_official_open_drawer_8192_eval.mp4'
receipt=json.loads((R/'results/completed_task_receipts/Open-Drawer_20261003.json').read_text(encoding='utf-8'))
a=receipt['result'];assert a['success_percent']==100 and a['train_exit']==a['eval_exit']==0
assert receipt['reference_count']==602 and hashlib.sha256((R/'media'/VIDEO).read_bytes()).hexdigest()==a['video_sha256']
def parse(p):return BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
def frag(t):return BeautifulSoup(t,'html.parser')
def url(p):return quote(str(p).replace('\\','/'),safe='/')
p=ROOT/'DreamControl_实验结果汇报.html';s=parse(p);s.header.find('strong').string='10 / 11'
for x in s.header.find_all('p'):
 if '实验成绩截至' in x.get_text():x.string='实验成绩截至2026年10月3日21:53（北京时间）：新增Open-Drawer官方成绩100%。文件夹沿用用户指定的20260929名称。'
for x in s.find(id='overview').find_all(string=True):x.replace_with(str(x).replace('九项官方','十项官方').replace('9项训练','10项训练'))
b=s.find(id='baseline');b.h2.string='DreamControl：十项官方控制基线'
b.find('p').string='十项均采用官方原参考库、8192个训练环境、2000次PPO迭代、seed42；完成至第1999轮后，用官方1000环境、500帧评估。Punch曾从第1600轮恢复。完成表示训练、评估与产物已归档，不表示所有任务达到预期效果。'
tables=b.find_all('table');body=tables[0].tbody
if not any(row.td and row.td.get_text()==TASK for row in body.find_all('tr')):
 body.append(frag('<tr><td>Open-Drawer</td><td>602</td><td><span class="metric">100.0%</span></td><td>默认Fabric路径；官方柜体LFS资产恢复</td><td>完成</td><td><a href="#task-video-Open-Drawer">查看视频</a></td></tr>').tr)
for row in list(tables[1].tbody.find_all('tr')):
 c=row.find_all('td')
 if c[0].get_text()==TASK:row.decompose()
 elif c[0].get_text()=='Open-Door':c[1].string='正式训练中';c[2].string='2026-10-03 21:53北京时间核实第747/2000轮；8192环境、seed42。原队列自动接续，无最终评估成绩。'
source=b.find('p',class_='source')
if TASK not in source.get_text():source.append(frag(f' · <a href="{url("论文复现汇报资料_20261002/results/r3_dreamcontrol_official_Open-Drawer_8192_eval.json")}">Open-Drawer原始成绩</a>'))
v=s.find(id='videos');v.find('div',class_='eyebrow').string='按任务配对 · 31段执行视频'
block=copy.deepcopy(s.find(id='task-video-Button-Press'));block['id']='task-video-'+TASK;block['data-task']=TASK;block.h3.string=TASK
block.find('p',class_='transfer-result').clear();block.find('p',class_='transfer-result').append(frag('<b>迁移结论：MuJoCo尚未验证</b><br/>Isaac官方任务成功率100%；602条原参考下的任务判据，尚无本任务跨仿真成绩或视频。'))
card=block.find('article');card.h3.string='Isaac Lab｜Open-Drawer官方评估';card.video.attrs.pop('poster',None);card.video.source['src']=url('论文复现汇报资料_20261002/media/'+VIDEO)
card.p.string='100.0%；官方602条参考；默认Fabric路径；1000环境、500帧；501帧，10.02秒。训练与评估正常退出0，使用官方柜体资产。'
card.a['href']=url('机器人执行视频/01_Isaac_Lab_官方基线/'+VIDEO)
if not s.find(id=block['id']):s.find(id='task-video-Button-Press').insert_after(block)
for nav in v.find_all('a',href='#task-video-Button-Press'):
 if not nav.find_parent(class_='task-video-block') and not any(x.get('href')=='#task-video-Open-Drawer' for x in nav.parent.find_all('a')):nav.insert_after(frag(' · <a href="#task-video-Open-Drawer">Open-Drawer</a>'))
summary=json.loads((ROOT/'PPO训练曲线'/TASK/'summary.json').read_text(encoding='utf-8'));m=summary['metrics']
for key in ['Train/mean_reward','Loss/value_function','Loss/surrogate','Train/mean_episode_length']:assert m[key]['points']==2000 and m[key]['last_step']==1999
analysis=f'<div id="ppo-drawer-completed-interpretation"><h3>Open-Drawer：完整训练与任务结果</h3><p>原始TensorBoard日志共2000轮；前100轮与末100轮均值：reward {m["Train/mean_reward"]["first100_mean"]:.3f}→{m["Train/mean_reward"]["last100_mean"]:.3f}，value loss {m["Loss/value_function"]["first100_mean"]:.6f}→{m["Loss/value_function"]["last100_mean"]:.6f}，PPO surrogate loss {m["Loss/surrogate"]["first100_mean"]:.6f}→{m["Loss/surrogate"]["last100_mean"]:.6f}，回合长度 {m["Train/mean_episode_length"]["first100_mean"]:.1f}→{m["Train/mean_episode_length"]["last100_mean"]:.1f}/500步。结合正式评估100%说明当前官方参考分布内任务判据得到满足。不能仅凭训练loss或这一分数认定未见条件泛化、自然抓握或MuJoCo可靠迁移；这些还未单独验收。</p></div>'
if not s.find(id='ppo-drawer-completed-interpretation'):s.find(id='training').append(frag(analysis))
if not s.find(id='training-chart-'+TASK):
 prefix=url('PPO训练曲线/'+TASK+'/')
 s.find(id='training').append(frag(f'<details class="training-chart" id="training-chart-{TASK}"><summary>{TASK}｜reward、value loss、PPO loss、熵、回合长度、动作噪声</summary><img class="curve-img" loading="lazy" alt="Open-Drawer真实训练曲线" src="{prefix}training_curves.svg"/><p><a href="{prefix}iteration_scalars.csv">每轮指标CSV</a> · <a href="{prefix}all_scalars_raw.csv.gz">全量标量</a> · <a href="{prefix}summary.json">来源与哈希</a></p></details>'))
f=s.find(id='failures')
for x in f.find_all(string=True):x.replace_with(str(x).replace('已归档9/11','已归档10/11'))
f=s.find(id='cabinet-lfs-failure');f.clear();f.append(frag('<b>Open-Drawer初始化失败与恢复：</b>旧尝试中柜体仍为130字节Git LFS指针，PPO开始前无法创建关节系统，0次更新、0个模型。已恢复原指针对应官方17599字节USD，独立16环境一次更新诊断通过；恢复原队列后，8192环境2000轮与官方1000环境500帧默认Fabric评估均正常退出0，成功率100%。原失败记录保留，奖励、关节配置和预算未改。Open-Door已自动接续，尚未完成。'))
assert len(s.select('#generation video'))==18 and len(s.select('#videos video'))==31 and len(s.select('.training-chart'))==10
p.write_text(str(s),encoding='utf-8')
robot=ROOT/'机器人执行视频';gallery=robot/'机器人视频观看.html';g=parse(gallery);script=g.find('script');text=script.string;match=re.search(r'const clips=(\[.*?\]);',text);clips=json.loads(match.group(1))
if not any(TASK in c['label'] for c in clips):clips.append({'file':'01_Isaac_Lab_官方基线/'+VIDEO,'label':'01_Isaac_Lab_官方基线｜'+TASK,'info':'官方评估100%；602参考；默认Fabric路径；10.02秒501帧；MuJoCo尚未验证。'})
script.string=text[:match.start(1)]+json.dumps(clips,ensure_ascii=False)+text[match.end(1):];g.title.string=g.h1.string='机器人执行｜本地31段视频'
for x in g.find_all(string=True):
 if '官方Isaac基线9段' in str(x):x.replace_with(str(x).replace('官方Isaac基线9段','官方Isaac基线10段'))
sel=g.find('select');sel.clear()
for i,c in enumerate(clips):o=g.new_tag('option',value=str(i));o.string=c['label'];sel.append(o)
gallery.write_text(str(g),encoding='utf-8')
mf=robot/'视频清单.json';j=json.loads(mf.read_text(encoding='utf-8'))
if not any(VIDEO in x['file'] for x in j['videos']):j['videos'].append({'file':'01_Isaac_Lab_官方基线/'+VIDEO,'category':'01_Isaac_Lab_官方基线','duration_seconds':10.02,'frames':501,'sha256':a['video_sha256'],'source':'论文复现汇报资料_20261002/media/'+VIDEO,'official_success_percent':100.0,'eval_exit':0})
j['total_videos']=len(j['videos']);mf.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
paired=robot/'按任务配对观看.html';ps=parse(paired)
if not ps.find(id=block['id']):
 pb=copy.deepcopy(block)
 for e in pb.select('[src],[href]'):
  for attr in ['src','href']:
   if attr in e.attrs and not e[attr].startswith('#'):e[attr]='../'+e[attr]
 ps.find(id='task-video-Button-Press').insert_after(pb);paired.write_text(str(ps),encoding='utf-8')
combined=ROOT/'PPO训练曲线/summary.json';j=json.loads(combined.read_text(encoding='utf-8'));j['tasks']=[x for x in j['tasks'] if x['task']!=TASK]+[summary];j['scope']='Ten completed official baseline runs; no active task metrics.';combined.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
index=ROOT/'PPO训练曲线/训练曲线总览.html';q=parse(index)
if not q.find(id='ppo-drawer-completed-interpretation'):
 (q.find('body') or q).append(frag(analysis));(q.find('body') or q).append(frag('<img style="width:100%" src="Open-Drawer/training_curves.svg"/><p><a href="Open-Drawer/iteration_scalars.csv">Open-Drawer逐轮CSV</a> · <a href="Open-Drawer/summary.json">原始日志来源</a></p>'));index.write_text(str(q),encoding='utf-8')
official=R/'media/DreamControl_official_baselines.html';os=parse(official)
if not any(TASK==row.td.get_text() for row in os.find_all('tr') if row.td):os.find('table').append(frag('<tr><td>Open-Drawer</td><td>8192环境 · 2000轮 · seed42</td><td>100%（默认Fabric路径；官方资产恢复）</td></tr>').tr)
if not os.find('video',src=VIDEO):(os.find('body') or os).append(frag(f'<section><h2>Open-Drawer｜官方成绩100%</h2><video controls preload="metadata" src="{VIDEO}"></video><p>602条参考；1000环境500帧；501帧10.02秒；MuJoCo尚未验证。</p></section>'))
for x in os.find_all(string=True):
 if '正式归档9 / 11' in str(x):x.replace_with('截至北京时间2026-10-03 21:53，正式归档10 / 11项；Open-Drawer100%，Open-Door已自动接续训练。')
 elif 'Open-Drawer资产恢复验证中' in str(x):x.replace_with(str(x).replace('Open-Drawer资产恢复验证中，Open-Door待训练。','Open-Drawer已完成100%，Open-Door训练中。'))
official.write_text(str(os),encoding='utf-8');shutil.copy2(official,PACK/'media'/official.name)
line='\n## 2026-10-03 13:53 UTC：Open-Drawer完成，Open-Door已接续\n\nOpen-Drawer官方602条参考、8192环境、2000迭代、seed42，run `2026-10-03_02-31-21-Open-Drawer-a` 正常完成1999；官方1000环境500帧默认Fabric视频评估正常退出0，成功率 **100.0%**。模型SHA256 `'+a['checkpoint_sha256']+'`；501帧/50fps/10.02秒视频 `media/'+VIDEO+'`，SHA256 `'+a['video_sha256']+'`。独立核验回执 `results/completed_task_receipts/Open-Drawer_20261003.json`，保留原TensorBoard events、逐轮CSV和曲线。原官方柜体LFS占位文件导致的零更新失败已归档，恢复精确官方资产后才成功；不改PPO、奖励、关节配置或预算。\n\n当前 **10/11控制任务归档**；Open-Door由原唯一队列自动启动，run `2026-10-03_11-04-14-Open-Door-a`，13:53 UTC核实747/2000轮，GPU1持续计算、检查点model_700已保存，实际状态帧step17960→17965有姿态变化，无最终成绩。Open-Drawer尚无MuJoCo迁移或独立未见条件/自然抓握验收；100%仅为官方任务判据结果。CLoSD正式微调仍等待既有本地当前场景确认。\n'
report=R/'reports/03_dreamcontrol_multitask_baselines.md';t=report.read_text(encoding='utf-8')
if '## 2026-10-03 13:53 UTC' not in t:t+=line;report.write_text(t,encoding='utf-8')
shutil.copy2(report,PACK/'reports'/report.name)
p=R/'reports/03_baseline_summary.md';t=p.read_text(encoding='utf-8')
if '2026-10-03 13:53 UTC最新' not in t:t+='\n\n2026-10-03 13:53 UTC最新：Open-Drawer完成100%，正式1999模型、默认Fabric官方1000环境500帧评估正常退出0，10.02秒501帧视频已核验。累计10/11控制任务。原队列Open-Door747/2000轮，没有最终成绩。CLoSD新正式微调仍待既有画面确认，未新增训练。\n';p.write_text(t,encoding='utf-8')
shutil.copy2(p,PACK/'reports'/p.name)
p=R/'README.md';t=p.read_text(encoding='utf-8').replace('当前已归档DreamControl控制任务为 **9/11**','当前已归档DreamControl控制任务为 **10/11**').replace('Ground-Pick-Top0%、Button-Press99.9%。这是保存','Ground-Pick-Top0%、Button-Press99.9%、Open-Drawer100%。这是保存');p.write_text(t,encoding='utf-8')
p=ROOT/'报告拆分清单.json';j=json.loads(p.read_text(encoding='utf-8'));j['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();j['dreamcontrol']={'archived_control_tasks':10,'human_videos':18,'execution_videos':31,'training_curve_tasks':10};j['subsequent_updates']='Open-Drawer completion; split operation itself did not change scores.'
for x in j['reports']:x['sha256']=hashlib.sha256((ROOT/x['file']).read_bytes()).hexdigest()
assert next(x['sha256'] for x in j['reports'] if x['file'].startswith('CLoSD'))=='0a595f5d755877888c4ebc35970f0d831074d46f708cd8fa60dc67c8d25994c4'
p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'tasks':10,'human_videos':18,'robot_videos':31,'training_curves':10,'closd_report_unchanged':True,'reward_first100':m['Train/mean_reward']['first100_mean'],'reward_last100':m['Train/mean_reward']['last100_mean']},ensure_ascii=False))
