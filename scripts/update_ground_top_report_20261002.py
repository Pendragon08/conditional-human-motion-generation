"""Add verified completed results to the local report; never launches simulation."""
import json, re, shutil, copy, datetime
from pathlib import Path
from urllib.parse import quote
from bs4 import BeautifulSoup
R=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
PACK=ROOT/'论文复现汇报资料_20261002'
TASK='Ground-Pick-Top'
VIDEO='r3_official_ground_pick_top_8192_eval.mp4'
receipt=json.loads((R/'results/completed_task_receipts/Ground-Pick-Top_20261002.json').read_text(encoding='utf-8'))
assert receipt['result']['success_percent']==0 and receipt['result']['eval_exit']==0
def parse(p):return BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
def fragment(t):return BeautifulSoup(t,'html.parser')
def link(p):return quote(str(p).replace('\\','/'),safe='/')
p=ROOT/'DreamControl_实验结果汇报.html' if (ROOT/'DreamControl_实验结果汇报.html').exists() else ROOT/'实验结果汇报.html'
backup=ROOT/'实验结果汇报_Ground-Pick-Top结果归档前备份.html'
if not backup.exists():shutil.copy2(p,backup)
s=parse(p)
for node in s.find(id='overview').find_all(string=True):
 node.replace_with(str(node).replace('七项官方','八项官方').replace('7项训练','8项训练'))
s.find('header').find('strong').string='8 / 11'
for node in s.find('header').find_all('p'):
 if '实验成绩截至' in node.get_text():node.string='实验成绩截至2026年10月2日22:18（北京时间）；包含Ground-Pick-Top的正常完成与0%结果。文件夹沿用用户指定的20260929名称。'
b=s.find(id='baseline');b.h2.string='DreamControl：八项官方控制基线'
b.find('p').string='八项均采用官方原参考库、8192个训练环境、2000次PPO迭代、seed42；完成至第1999轮后，用官方1000环境、500帧评估。Punch曾中断，从第1600轮恢复。完成表示训练、评估与产物已归档，不表示所有任务达到预期效果。'
tables=b.find_all('table');body=tables[0].tbody
if not any(TASK==row.find('td').get_text() for row in body.find_all('tr')):
 body.append(fragment('<tr><td>Ground-Pick-Top</td><td>1</td><td><span class="metric">0.0%</span></td><td>默认Fabric路径</td><td>完成；任务未成功</td><td><a href="#task-video-Ground-Pick-Top">查看视频</a></td></tr>').tr)
for row in list(tables[1].tbody.find_all('tr')):
 cells=row.find_all('td');task=cells[0].get_text()
 if task==TASK:row.decompose()
 elif task=='Button-Press':cells[1].string='训练中';cells[2].string='2026-10-02 22:18核对第443/2000轮；无最终评估成绩'
source=b.find('p',class_='source')
if TASK not in source.get_text():source.append(fragment(f' · <a href="{link("论文复现汇报资料_20261002/results/r3_dreamcontrol_official_Ground-Pick-Top_8192_eval.json")}">Ground-Pick-Top原始成绩</a>'))
if not b.find(id='ground-top-zero-result'):
 b.find('h3').insert_before(fragment('<p id="ground-top-zero-result"><b>Ground-Pick-Top：0%是有效失败结果。</b>训练和官方评估均正常退出0，使用默认Fabric运行路径；1000个评估环境均未触发官方成功判据。保留最终模型与视频，不将其写为初始化异常，也不据此宣称论文效果复现通过。当前证据尚不能定位失败的具体原因。</p>'))
v=s.find(id='videos');v.find('div',class_='eyebrow').string='按任务配对 · 29段执行视频'
block=copy.deepcopy(s.find(id='task-video-Ground-Pick'))
block['id']='task-video-Ground-Pick-Top';block['data-task']=TASK
block.h3.string=TASK
block.find('p',class_='transfer-result').clear();block.find('p',class_='transfer-result').append(fragment('<b>迁移结论：MuJoCo尚未验证</b><br/>Isaac官方评估0%；训练和评估正常完成，尚无本任务跨仿真成绩或视频。'))
card=block.find('article');card.h3.string='Isaac Lab｜Ground-Pick-Top官方评估'
card.video.attrs.pop('poster',None);card.video.source['src']=link('论文复现汇报资料_20261002/media/'+VIDEO)
card.p.string='0.0%；默认Fabric路径；1000环境、500帧；501帧，10.02秒。正常退出，不是初始化失败。'
card.a['href']=link('机器人执行视频/01_Isaac_Lab_官方基线/'+VIDEO)
if not s.find(id=block['id']):s.find(id='task-video-Ground-Pick').insert_after(block)
for nav in v.find_all('a',href='#task-video-Ground-Pick'):
 if not nav.find_parent(class_='task-video-block') and not any(a.get('href')=='#task-video-Ground-Pick-Top' for a in nav.parent.find_all('a')):nav.insert_after(fragment(' · <a href="#task-video-Ground-Pick-Top">Ground-Pick-Top</a>'))
summaries=[];rows=[]
for task,score in [('Ground-Pick',38.1),(TASK,0.0)]:
 a=json.loads((ROOT/'PPO训练曲线'/task/'summary.json').read_text(encoding='utf-8'));summaries.append(a);m=a['metrics']
 for key in ['Train/mean_reward','Train/mean_episode_length','Loss/value_function','Loss/surrogate']:
  assert m[key]['points']==2000 and m[key]['first_step']==0 and m[key]['last_step']==1999
 def fmt(k,n=3):return f"{m[k]['first100_mean']:.{n}f} → {m[k]['last100_mean']:.{n}f}"
 rows.append(f'<tr><td>{task}</td><td>{fmt("Train/mean_reward")}</td><td>{fmt("Loss/value_function",6)}</td><td>{fmt("Loss/surrogate",6)}</td><td>{m["Train/mean_episode_length"]["last100_mean"]:.1f}/500步</td><td>{score:.1f}%</td></tr>')
section=fragment('<div id="ppo-ground-completed-interpretation"><h3>新增两项地面拾取：训练改善与正式成功率的差异</h3><p>直接读取两run的原始TensorBoard events，均为0—1999共2000点。前/末阶段分别取100轮均值；surrogate为带符号的PPO代理损失。</p><div class="tablewrap"><table><thead><tr><th>任务</th><th>reward前100→末100</th><th>value loss前100→末100</th><th>surrogate前100→末100</th><th>末期回合长度</th><th>官方评估</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p>两项回报与存活时长改善，价值预测误差下降；但Ground-Pick仅38.1%，Ground-Pick-Top为0%。这说明策略学到了部分奖励所鼓励的行为，不能仅凭高reward、长回合或低value loss认定物体目标已完成。Ground-Pick-Top不是训练/评估进程失败；当前还缺少分项行为诊断，不能将原因确定归于某一个奖励或参数。两项尚未进行MuJoCo迁移验证。</p></div>')
old=s.find(id='ppo-ground-completed-interpretation')
if old:old.replace_with(section)
else:s.find(id='training').append(section)
for task in ['Ground-Pick',TASK]:
 if not s.find(id='training-chart-'+task):
  prefix=link('PPO训练曲线/'+task+'/')
  s.find(id='training').append(fragment(f'<details class="training-chart" id="training-chart-{task}"><summary>{task}｜reward、value loss、PPO loss、熵、回合长度、动作噪声</summary><img class="curve-img" loading="lazy" alt="{task}真实训练曲线" src="{prefix}training_curves.svg"/><p><a href="{prefix}iteration_scalars.csv">每轮指标CSV</a> · <a href="{prefix}all_scalars_raw.csv.gz">全量标量</a> · <a href="{prefix}summary.json">来源与哈希</a></p></details>'))
assert len(s.select('#generation video'))==18 and len(s.select('#videos video'))==29
p.write_text(str(s),encoding='utf-8')
# Keep one consistent robot gallery directory, manifest, and selectable clip.
robot=ROOT/'机器人执行视频';dest=robot/'01_Isaac_Lab_官方基线'/VIDEO;shutil.copy2(R/'media'/VIDEO,dest)
gallery=robot/'机器人视频观看.html';g=parse(gallery);script=g.find('script');text=script.string
match=re.search(r'const clips=(\[.*?\]);',text);clips=json.loads(match.group(1))
if not any(TASK in c['label'] for c in clips):clips.append({'file':'01_Isaac_Lab_官方基线/'+VIDEO,'label':'01_Isaac_Lab_官方基线｜'+TASK,'info':'官方评估0.0%；10.02秒501帧；默认Fabric路径；正常退出；MuJoCo尚未验证。'})
script.string=text[:match.start(1)]+json.dumps(clips,ensure_ascii=False)+text[match.end(1):]
g.title.string=g.h1.string='机器人执行｜本地29段视频'
for textnode in g.find_all(string=True):
 if '官方Isaac基线7段' in str(textnode):textnode.replace_with(str(textnode).replace('官方Isaac基线7段','官方Isaac基线8段'))
sel=g.find('select');sel.clear()
for i,c in enumerate(clips):o=g.new_tag('option',value=str(i));o.string=c['label'];sel.append(o)
gallery.write_text(str(g),encoding='utf-8')
manifest=robot/'视频清单.json';j=json.loads(manifest.read_text(encoding='utf-8'))
if not any(VIDEO in x['file'] for x in j['videos']):j['videos'].append({'file':'01_Isaac_Lab_官方基线/'+VIDEO,'category':'01_Isaac_Lab_官方基线','duration_seconds':10.02,'frames':501,'sha256':receipt['result']['video_sha256'],'source':'论文复现汇报资料_20261002/media/'+VIDEO,'official_success_percent':0.0,'eval_exit':0})
j['total_videos']=len(j['videos']);manifest.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
paired=robot/'按任务配对观看.html'
if paired.exists():
 ps=parse(paired)
 if not ps.find(id=block['id']):
  pb=copy.deepcopy(block)
  for e in pb.select('[src],[href]'):
   for a in ['src','href']:
    if a in e.attrs and not e[a].startswith('#'):e[a]='../'+e[a]
  (ps.find(id='task-video-Ground-Pick') or ps.find('body') or ps).append(pb) if not ps.find(id='task-video-Ground-Pick') else ps.find(id='task-video-Ground-Pick').insert_after(pb)
  paired.write_text(str(ps),encoding='utf-8')
combined=ROOT/'PPO训练曲线/summary.json';j=json.loads(combined.read_text(encoding='utf-8'))
j['tasks']=[a for a in j['tasks'] if a['task'] not in ['Ground-Pick',TASK]]+summaries
j['exported_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();j['scope']='Eight completed official baseline runs; no active task metrics.'
combined.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
index=ROOT/'PPO训练曲线/训练曲线总览.html';q=parse(index)
if not q.find(id='ground-completed-appendix'):
 extra=copy.deepcopy(s.find(id='ppo-ground-completed-interpretation'));extra['id']='ground-completed-appendix'
 for task in ['Ground-Pick',TASK]:extra.append(fragment(f'<h3>{task}</h3><img style="width:100%" src="{task}/training_curves.svg"/><p><a href="{task}/iteration_scalars.csv">逐轮CSV</a> · <a href="{task}/summary.json">原始来源</a></p>'))
 (q.find('body') or q).append(extra);index.write_text(str(q),encoding='utf-8')
# Repo result/video entry and dated report timeline.
official=R/'media/DreamControl_official_baselines.html';os=parse(official)
for node in os.find_all(string=True):
 if '正式完成 6 / 11' in str(node):node.replace_with('截至北京时间2026-10-02 22:18，正式完成8 / 11项；包括Ground-Pick-Top正常完成但0%的结果。视频来自Isaac Lab官方评估。')
table=os.find('table')
for task,score,vid in [('Ground-Pick','38.1%（USD显示适配）','r3_official_ground_pick_8192_eval.mp4'),(TASK,'0.0%（默认Fabric路径）',VIDEO)]:
 if not any(task==row.find('td').get_text() for row in table.find_all('tr') if row.find('td')):table.append(fragment(f'<tr><td>{task}</td><td>8192环境 · 2000轮 · seed42</td><td>{score}</td></tr>').tr)
 if not os.find('video',src=vid):os.append(fragment(f'<section><h2>{task}｜官方成绩{score}</h2><video controls preload="metadata" src="{vid}"></video><p>1000环境500帧；501帧10.02秒；MuJoCo尚未验证。</p></section>'))
for node in os.find_all(string=True):
 if 'Ground-Pick 地面拾取已开始' in str(node):node.replace_with(str(node).replace('Ground-Pick 地面拾取已开始，其余四项排队。','Ground-Pick与Ground-Pick-Top已完成；Button-Press训练中，Open-Drawer、Open-Door排队。'))
official.write_text(str(os),encoding='utf-8');shutil.copy2(official,PACK/'media'/official.name)
readme=R/'README.md';t=readme.read_text(encoding='utf-8');t=t.replace('当前已归档DreamControl控制任务为 **7/11**','当前已归档DreamControl控制任务为 **8/11**').replace('Bimanual-Pick99.9%、Ground-Pick38.1%。这是保存','Bimanual-Pick99.9%、Ground-Pick38.1%、Ground-Pick-Top0%。这是保存').replace('Ground-Pick尚无MuJoCo验证。','Ground-Pick与Ground-Pick-Top尚无MuJoCo验证；0%保留为正式失败成绩，不代表效果复现通过。');readme.write_text(t,encoding='utf-8')
report=R/'reports/03_dreamcontrol_multitask_baselines.md';t=report.read_text(encoding='utf-8')
if '## 2026-10-02 14:18 UTC' not in t:
 t+='\n## 2026-10-02 14:18 UTC（北京时间22:18）：Ground-Pick-Top完成，官方成绩0%\n\n官方1条参考、8192环境、2000迭代、seed42，从零训练至1999，训练和官方默认Fabric路径1000环境500帧视频评估均正常退出0。成功率 **0.0%** 是真实任务失败成绩，不是初始化异常；不重训或重试已正常完成的评估，不宣称效果复现通过。具体原因尚未完成行为诊断。\n\n最终模型SHA256 `'+receipt['result']['checkpoint_sha256']+'`；视频 `media/'+VIDEO+'`，501帧、50fps、10.02秒，SHA256 `'+receipt['result']['video_sha256']+'`。模型/日志/配置/视频均独立核验，回执 `results/completed_task_receipts/Ground-Pick-Top_20261002.json`；正式结果 `results/r3_dreamcontrol_official_Ground-Pick-Top_8192_eval.json`。模型仅保留服务器，不随报告上传。\n\nGround-Pick与Ground-Pick-Top的TensorBoard原始events、全量CSV和曲线已导出本地。前/末100轮平均reward分别−10.108→4.805、−9.759→8.051；末期平均回合约499步；value loss分别0.828871→0.003858、0.737016→0.001828。这支持部分存活/训练奖励改善，但不能代替38.1%/0%官方任务成绩，更不能证明自然抓取与碰撞接触成功。两项均尚无MuJoCo测试。\n\n当前归档 **8/11** 控制任务；原唯一队列已自动进入Button-Press，14:18 UTC核对443/2000轮，随后Open-Drawer、Open-Door。未启动重复队列或修改、中断现有训练。CLoSD新3072正式微调仍等待此前已询问的本地实际场景确认，不重复询问。\n'
 report.write_text(t,encoding='utf-8')
shutil.copy2(report,PACK/'reports'/report.name)
print(json.dumps({'control_tasks_archived':8,'human_video_count':18,'robot_video_count':29,'training_curve_tasks':len(j['tasks']),'official_ground_top_success_percent':0.0},ensure_ascii=False))
