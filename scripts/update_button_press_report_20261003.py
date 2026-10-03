"""Archive verified Button-Press outputs in the separate DreamControl report."""
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
TASK='Button-Press'; VIDEO='r3_official_button_press_8192_eval.mp4'
receipt=json.loads((R/'results/completed_task_receipts/Button-Press_20261003.json').read_text(encoding='utf-8'))
a=receipt['result']; assert a['train_exit']==a['eval_exit']==0
assert abs(a['success_percent']-99.9)<.001 and receipt['reference_count']==96
assert hashlib.sha256((R/'media'/VIDEO).read_bytes()).hexdigest()==a['video_sha256']
def parse(p):return BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
def frag(t):return BeautifulSoup(t,'html.parser')
def url(p):return quote(str(p).replace('\\','/'),safe='/')
p=ROOT/'DreamControl_实验结果汇报.html'; s=parse(p)
# The original combined report is an index; only the DreamControl paper is edited.
s.header.find('strong').string='9 / 11'
for t in s.header.find_all('p'):
 if '实验成绩截至' in t.get_text():t.string='实验成绩截至2026年10月3日10:24（北京时间）：新增Button-Press官方成绩99.9%。文件夹沿用用户指定的20260929名称。'
for t in s.find(id='overview').find_all(string=True):
 t.replace_with(str(t).replace('八项官方','九项官方').replace('8项训练','9项训练'))
b=s.find(id='baseline');b.h2.string='DreamControl：九项官方控制基线'
b.find('p').string='九项均采用官方原参考库、8192个训练环境、2000次PPO迭代、seed42；完成至第1999轮后，用官方1000环境、500帧评估。Punch曾从第1600轮恢复。完成表示训练、评估与产物已归档，不表示所有任务达到预期效果。'
tables=b.find_all('table');body=tables[0].tbody
if not any(row.td and row.td.get_text()==TASK for row in body.find_all('tr')):
 body.append(frag('<tr><td>Button-Press</td><td>96</td><td><span class="metric">99.9%</span></td><td>默认Fabric路径</td><td>完成</td><td><a href="#task-video-Button-Press">查看视频</a></td></tr>').tr)
for row in list(tables[1].tbody.find_all('tr')):
 cells=row.find_all('td')
 if cells[0].get_text()==TASK:row.decompose()
 elif cells[0].get_text()=='Open-Drawer':
  cells[1].string='初始化失败；资产恢复验证中'
  cells[2].string='柜体USD仍为130字节Git LFS占位文件；未进入PPO、无模型、无评估分数。官方对应资产已核验并恢复，等待独立短回合诊断。'
 elif cells[0].get_text()=='Open-Door':cells[2].string='等待Open-Drawer恢复；使用同一官方柜体资产，未正式训练与评估。'
source=b.find('p',class_='source')
if TASK not in source.get_text():source.append(frag(f' · <a href="{url("论文复现汇报资料_20261002/results/r3_dreamcontrol_official_Button-Press_8192_eval.json")}">Button-Press原始成绩</a>'))
v=s.find(id='videos');v.find('div',class_='eyebrow').string='按任务配对 · 30段执行视频'
block=copy.deepcopy(s.find(id='task-video-Ground-Pick-Top'));block['id']='task-video-'+TASK;block['data-task']=TASK;block.h3.string=TASK
block.find('p',class_='transfer-result').clear();block.find('p',class_='transfer-result').append(frag('<b>迁移结论：MuJoCo尚未验证</b><br/>Isaac官方任务成功率99.9%；这是原96条参考下的任务判据，尚无本任务跨仿真成绩或视频。'))
card=block.find('article');card.h3.string='Isaac Lab｜Button-Press官方评估';card.video.attrs.pop('poster',None)
card.video.source['src']=url('论文复现汇报资料_20261002/media/'+VIDEO)
card.p.string='99.9%；官方96条参考；默认Fabric路径；1000环境、500帧；501帧，10.02秒。训练与评估均正常退出0。'
card.a['href']=url('机器人执行视频/01_Isaac_Lab_官方基线/'+VIDEO)
if not s.find(id=block['id']):s.find(id='task-video-Ground-Pick-Top').insert_after(block)
for nav in v.find_all('a',href='#task-video-Ground-Pick-Top'):
 if not nav.find_parent(class_='task-video-block') and not any(x.get('href')=='#task-video-Button-Press' for x in nav.parent.find_all('a')):nav.insert_after(frag(' · <a href="#task-video-Button-Press">Button-Press</a>'))
summary=json.loads((ROOT/'PPO训练曲线'/TASK/'summary.json').read_text(encoding='utf-8'));m=summary['metrics']
for key in ['Train/mean_reward','Loss/value_function','Loss/surrogate','Train/mean_episode_length']:assert m[key]['points']==2000 and m[key]['last_step']==1999
analysis=f'<div id="ppo-button-completed-interpretation"><h3>Button-Press：奖励改善与正式任务结果</h3><p>原始TensorBoard日志共2000轮；前100轮与末100轮均值：reward {m["Train/mean_reward"]["first100_mean"]:.3f}→{m["Train/mean_reward"]["last100_mean"]:.3f}，value loss {m["Loss/value_function"]["first100_mean"]:.6f}→{m["Loss/value_function"]["last100_mean"]:.6f}，PPO surrogate loss {m["Loss/surrogate"]["first100_mean"]:.6f}→{m["Loss/surrogate"]["last100_mean"]:.6f}，回合长度 {m["Train/mean_episode_length"]["first100_mean"]:.1f}→{m["Train/mean_episode_length"]["last100_mean"]:.1f}/500步。回报增长、回合延长、价值预测误差减小，与正式任务成功率99.9%一起支持当前参考分布内控制效果。surrogate是带符号的代理损失，不能按越低越好评价动作质量；尚无本任务MuJoCo、实体按钮接触或未见条件泛化验证。</p></div>'
if not s.find(id='ppo-button-completed-interpretation'):s.find(id='training').append(frag(analysis))
if not s.find(id='training-chart-'+TASK):
 prefix=url('PPO训练曲线/'+TASK+'/')
 s.find(id='training').append(frag(f'<details class="training-chart" id="training-chart-{TASK}"><summary>{TASK}｜reward、value loss、PPO loss、熵、回合长度、动作噪声</summary><img class="curve-img" loading="lazy" alt="Button-Press真实训练曲线" src="{prefix}training_curves.svg"/><p><a href="{prefix}iteration_scalars.csv">每轮指标CSV</a> · <a href="{prefix}all_scalars_raw.csv.gz">全量标量</a> · <a href="{prefix}summary.json">来源与哈希</a></p></details>'))
f=s.find(id='failures')
for t in f.find_all(string=True):t.replace_with(str(t).replace('已归档8/11','已归档9/11'))
if not f.find(id='cabinet-lfs-failure'):
 f.append(frag('<p id="cabinet-lfs-failure"><b>Open-Drawer初始化失败：</b>在正式PPO开始前，柜体文件仍是Git LFS指针，无法创建关节系统；没有训练迭代或模型，不能记为0%任务成绩。官方对象已按原指针SHA256恢复，独立诊断尚待通过；恢复不改变任务奖励、关节配置或预算。Open-Door尚未启动。</p>'))
assert len(s.select('#generation video'))==18 and len(s.select('#videos video'))==30 and len(s.select('.training-chart'))==9
p.write_text(str(s),encoding='utf-8')
robot=ROOT/'机器人执行视频';gallery=robot/'机器人视频观看.html';g=parse(gallery);script=g.find('script');text=script.string
match=re.search(r'const clips=(\[.*?\]);',text);clips=json.loads(match.group(1))
if not any(TASK in c['label'] for c in clips):clips.append({'file':'01_Isaac_Lab_官方基线/'+VIDEO,'label':'01_Isaac_Lab_官方基线｜'+TASK,'info':'官方评估99.9%；96参考；默认Fabric路径；10.02秒501帧；MuJoCo尚未验证。'})
script.string=text[:match.start(1)]+json.dumps(clips,ensure_ascii=False)+text[match.end(1):]
g.title.string=g.h1.string='机器人执行｜本地30段视频'
for t in g.find_all(string=True):
 if '官方Isaac基线8段' in str(t):t.replace_with(str(t).replace('官方Isaac基线8段','官方Isaac基线9段'))
sel=g.find('select');sel.clear()
for i,c in enumerate(clips):option=g.new_tag('option',value=str(i));option.string=c['label'];sel.append(option)
gallery.write_text(str(g),encoding='utf-8')
mf=robot/'视频清单.json';j=json.loads(mf.read_text(encoding='utf-8'))
if not any(VIDEO in x['file'] for x in j['videos']):j['videos'].append({'file':'01_Isaac_Lab_官方基线/'+VIDEO,'category':'01_Isaac_Lab_官方基线','duration_seconds':10.02,'frames':501,'sha256':a['video_sha256'],'source':'论文复现汇报资料_20261002/media/'+VIDEO,'official_success_percent':a['success_percent'],'eval_exit':0})
j['total_videos']=len(j['videos']);mf.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
paired=robot/'按任务配对观看.html';ps=parse(paired)
if not ps.find(id=block['id']):
 pb=copy.deepcopy(block)
 for e in pb.select('[src],[href]'):
  for attr in ['src','href']:
   if attr in e.attrs and not e[attr].startswith('#'):e[attr]='../'+e[attr]
 ps.find(id='task-video-Ground-Pick-Top').insert_after(pb);paired.write_text(str(ps),encoding='utf-8')
combined=ROOT/'PPO训练曲线/summary.json';j=json.loads(combined.read_text(encoding='utf-8'));j['tasks']=[x for x in j['tasks'] if x['task']!=TASK]+[summary];j['scope']='Nine completed official baseline runs; no active task metrics.';combined.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
index=ROOT/'PPO训练曲线/训练曲线总览.html';q=parse(index)
if not q.find(id='ppo-button-completed-interpretation'):
 (q.find('body') or q).append(frag(analysis));(q.find('body') or q).append(frag('<img style="width:100%" src="Button-Press/training_curves.svg"/><p><a href="Button-Press/iteration_scalars.csv">Button-Press逐轮CSV</a> · <a href="Button-Press/summary.json">原始日志来源</a></p>'));index.write_text(str(q),encoding='utf-8')
official=R/'media/DreamControl_official_baselines.html';os=parse(official)
if not any(TASK==row.td.get_text() for row in os.find_all('tr') if row.td):os.find('table').append(frag('<tr><td>Button-Press</td><td>8192环境 · 2000轮 · seed42</td><td>99.9%（默认Fabric路径）</td></tr>').tr)
if not os.find('video',src=VIDEO):(os.find('body') or os).append(frag(f'<section><h2>Button-Press｜官方成绩99.9%</h2><video controls preload="metadata" src="{VIDEO}"></video><p>96条参考；1000环境500帧；501帧10.02秒；MuJoCo尚未验证。</p></section>'))
for t in os.find_all(string=True):
 if '正式完成8 / 11' in str(t):t.replace_with('截至北京时间2026-10-03 10:24，正式归档9 / 11项；Open-Drawer初始化失败正在恢复验证，Open-Door待训练。')
 elif 'Button-Press训练中' in str(t):t.replace_with(str(t).replace('Button-Press训练中，Open-Drawer、Open-Door排队。','Button-Press已完成99.9%；Open-Drawer资产恢复验证中，Open-Door待训练。'))
official.write_text(str(os),encoding='utf-8');shutil.copy2(official,PACK/'media'/official.name)
report=R/'reports/03_dreamcontrol_multitask_baselines.md';t=report.read_text(encoding='utf-8')
if '## 2026-10-03：Button-Press' not in t:
 t+='\n## 2026-10-03：Button-Press归档；柜体资产初始化阻塞\n\nButton-Press使用官方96条参考、8192环境、2000迭代、seed42。正常完成1999，官方1000环境500帧默认Fabric视频评估退出0，成功率 **99.9%**。模型SHA256 `'+a['checkpoint_sha256']+'`；本地视频 `media/'+VIDEO+'`，50fps、501帧、10.02秒，SHA256 `'+a['video_sha256']+'`。独立回执 `results/completed_task_receipts/Button-Press_20261003.json`。原始TensorBoard事件与逐轮CSV已导出；前/末100轮平均reward −10.537→6.805，value loss 0.774752→0.000655，回合长度87.5→499.7步。surrogate −0.003514→−0.002023不能单独解释为动作质量变好；尚无本任务MuJoCo迁移验证。\n\n已归档 **9/11控制任务**，不是整篇论文复现完成。Open-Drawer的旧尝试在初始化时失败，没有PPO迭代或模型；原柜体USD只有130字节，是Git LFS占位文件。已恢复官方指针对应17599字节对象，SHA256 `30fddfc98008e380fe37d947e995c7875ead6113544b810d0254dc23450b6a7c`。原始失败记录保全；独立场景/一次更新诊断尚待通过。恢复过程、后续唯一队列启动条件见 `docs/cabinet_asset_recovery_20261003.md`。不把初始化错误记为0%，不重训已完成任务。Open-Door等待。CLoSD新正式微调仍受已有未答复的独立画面确认约束。\n'
 report.write_text(t,encoding='utf-8')
shutil.copy2(report,PACK/'reports'/report.name)
readme=R/'README.md';t=readme.read_text(encoding='utf-8').replace('当前已归档DreamControl控制任务为 **8/11**','当前已归档DreamControl控制任务为 **9/11**').replace('Ground-Pick-Top0%。这是保存','Ground-Pick-Top0%、Button-Press99.9%。这是保存');readme.write_text(t,encoding='utf-8')
print(json.dumps({'tasks':9,'human_videos':18,'robot_videos':30,'training_curves':9,'edited_report':str(p),'closd_report_unchanged':True},ensure_ascii=False))
