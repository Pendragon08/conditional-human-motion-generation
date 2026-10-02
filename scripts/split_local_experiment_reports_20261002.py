"""Split the teacher HTML into paper-specific reports, sharing existing assets."""
from pathlib import Path
from bs4 import BeautifulSoup
import copy,json,hashlib,datetime,shutil
from urllib.parse import quote
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
evidence=ROOT/'论文复现汇报资料_20261002/reports'
evidence.mkdir(parents=True,exist_ok=True)
shutil.copy2(REPO/'reports/01_closd_reproduction.md',evidence/'01_closd_reproduction.md')
MAIN=ROOT/'实验结果汇报.html'
BACKUP=ROOT/'实验结果汇报_合并版备份_20261002.html'
DREAM='DreamControl_实验结果汇报.html'
CLOSD='CLoSD_实验结果汇报.html'
if not BACKUP.exists():BACKUP.write_bytes(MAIN.read_bytes())
source=BeautifulSoup(BACKUP.read_text(encoding='utf-8'),'html.parser')
assert source.find(id='baseline') and source.find(id='closd')
def frag(t):return BeautifulSoup(t,'html.parser')
def href(p):return quote(p,safe='/#')
def base(title):
 s=BeautifulSoup('<!doctype html><html lang="zh-CN"><head></head><body><nav></nav><main class="main"></main></body></html>','html.parser')
 s.head.replace_with(copy.deepcopy(source.head));s.title.string=title
 style=s.new_tag('style');style.string='header a{color:#d7f4ff;text-decoration:underline}';s.head.append(style)
 s.body.append(frag('<script>document.querySelectorAll("video").forEach(v=>v.addEventListener("play",()=>document.querySelectorAll("video").forEach(o=>{if(o!==v)o.pause()})));const search=document.querySelector("#search");if(search)search.oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll(".searchable").forEach(d=>d.classList.toggle("hidden",!d.textContent.toLowerCase().includes(q)))};function choose(group){document.querySelectorAll(".video-card[data-group]").forEach(c=>{const hide=group!=="all"&&c.dataset.group!==group;c.classList.toggle("hidden",hide);if(hide&&c.querySelector("video"))c.querySelector("video").pause()});document.querySelectorAll("[data-filter]").forEach(b=>b.classList.toggle("active",b.dataset.filter===group))}document.querySelectorAll("[data-filter]").forEach(b=>b.onclick=()=>choose(b.dataset.filter));document.querySelectorAll("[data-select]").forEach(a=>a.onclick=()=>choose(a.dataset.select));</script>'))
 return s
def nav(s,title,items,other):
 s.nav.clear();s.nav.append(frag('<b>'+title+'</b>'))
 for i,(id,label) in enumerate(items,1):s.nav.append(frag(f'<a href="#{id}">{i:02d} {label}</a>'))
 s.nav.append(frag(f'<a href="{href(other)}">另一篇论文报告</a><a href="{href(MAIN.name)}">报告入口</a><small>本地离线档案<br/>结果快照：2026-10-02</small>'))
d=base('DreamControl｜论文复现实验报告')
for element in source.main.children:
 if not element.name:continue
 if element.get('id')=='closd':continue
 d.main.append(copy.deepcopy(element))
head=d.find('header');head.h1.string='DreamControl · 论文复现实验报告';head.find('p',class_='subtitle').string='条件人体运动生成、Isaac Lab PPO控制与MuJoCo跨仿真'
for card in list(head.select('.hero-grid > div')):
 if 'CLoSD' in card.get_text():card.decompose()
overview=d.find(id='overview');overview.h2.string='DreamControl复现概况'
for row in list(overview.find_all('tr')):
 if 'CLoSD' in row.get_text():row.decompose()
methods=d.find(id='methods');closd_heading=next(x for x in methods.find_all('h3',recursive=False) if 'CLoSD' in x.get_text())
clnodes=[closd_heading]
for x in closd_heading.next_siblings:
 if x.name=='div' and 'tablewrap' in x.get('class',[]):break
 if x.name:clnodes.append(x)
clmethod=[copy.deepcopy(x) for x in clnodes]
for x in clnodes:x.decompose()
methods.find('h3',recursive=False).string='DreamControl：先生成参考动作，再训练机器人跟踪'
for row in methods.find_all('tr'):
 if row.find('td') and row.find('td').get_text()=='闭环':row.decompose()
fail=d.find(id='failures')
for row in list(fail.find_all('tr')):
 if 'CLoSD' in row.get_text():row.decompose()
fail.find('div',class_='note').string='DreamControl已归档8/11控制任务，不等于整篇论文完成或全部任务效果达标。人体生成骨架、机器人参考、Isaac执行、MuJoCo执行是不同层次，分别标注。没有实机验证。'
for p in fail.find_all('p'):
 if '后续先继续' in p.get_text():p.string='后续先继续既有DreamControl论文复现队列。对跨仿真失败核对任务场景、惯量/碰撞与驱动映射，按独立协议验证，不覆盖本轮负结果。'
appendix=d.find(id='appendix')
for det in list(appendix.find_all('details',recursive=False)):
 if det.summary and 'CLoSD' in det.summary.get_text():det.decompose();continue
 for row in list(det.find_all('tr')):
  if 'closd' in row.get_text().lower():row.decompose()
for p in d.find(id='videos').find_all('p'):
 if '所有28段机器人视频' in p.get_text():p.string='本报告29段机器人执行视频已保存本地。其中八段是Isaac Lab官方基线评估视频，九段为六任务MuJoCo执行，十二段为Kick/Punch扰动与接触。Ground-Pick和Ground-Pick-Top尚无MuJoCo视频；Button-Press、Open-Drawer、Open-Door尚无已归档正式结果视频。'
items=[('overview','复现概况'),('methods','人体生成方法与输入'),('generation','人体视频与质量筛查'),('training','PPO原理与训练曲线'),('artifacts','流程产物与策略格式'),('baseline','官方控制基线'),('sim','MuJoCo迁移'),('robust','扰动与实体接触'),('failures','问题与后续'),('videos','逐任务执行视频'),('appendix','证据与来源')]
nav(d,'DreamControl实验报告',items,CLOSD)
head.append(frag(f'<p><a href="{href(CLOSD)}">CLoSD独立实验报告</a> · <a href="{href(MAIN.name)}">两份报告入口</a></p>'))
assert not d.find(id='closd') and len(d.select('#generation video'))==18 and len(d.select('#videos video'))==29 and len(d.select('#training details.training-chart'))==8
assert d.find(id='baseline').find_all('table')[0].get_text()==source.find(id='baseline').find_all('table')[0].get_text()
c=base('CLoSD｜论文复现实验报告')
c.main.append(frag('<div class="toolbar"><button onclick="window.print()">打印 / 保存PDF</button><span>浏览器可离线打开；正文与复现证据保存在本文件夹。</span></div>'))
c.main.append(frag('<header id="top"><div class="eyebrow">研究汇报 · 论文复现</div><h1>CLoSD · 论文复现实验报告</h1><p class="subtitle">扩散运动规划与物理控制闭环：官方权重评估及本地微调</p><p>已归档结果截至2026年10月2日。正式3072环境微调仍未完成；当前报告不将预检或官方权重评估记作新训练完成。</p><div class="hero-grid"><div><strong>2组</strong><span>官方权重4096环境评估完成</span></div><div><strong>4类</strong><span>任务状态；踢/拳共用STRIKE</span></div><div><strong>512环境</strong><span>旧微调退化记录保留</span></div><div><strong>待验收</strong><span>3072环境正式微调</span></div></div></header>'))
c.main.append(frag('<section id="overview"><div class="eyebrow">复现概况</div><h2>评估复现已完成，正式训练复现尚未完成</h2><p>已发布的未微调与微调权重均在4096环境、500帧、CFG7.5、seed0下完成官方评估，并正常退出。旧512环境本地微调在保存epoch66000后异常退出，匹配评估显示多数任务退化。新的3072环境三次更新诊断正常退出，但不等于4000轮正式微调完成。</p><div class="note">两次4096环境评估没有录制视频；本报告不使用DreamControl视频或实时监控回放作为CLoSD结果视频。</div></section>'))
m=frag('<section id="methods"><div class="eyebrow">方法与条件</div><h2>DiP扩散规划＋RL物理跟踪的闭环</h2></section>').section
for x in clmethod:m.append(x)
m.h3.string='CLoSD：短段生成与物理执行不断交替'
m.append(frag('<div class="tablewrap"><table><tr><th>环节</th><th>作用</th><th>本次复现范围</th></tr><tr><td>条件与历史</td><td>任务文本、目标位置及实际执行的人体运动历史</td><td>按官方任务与评估器运行，不将G1参考库当作CLoSD输入</td></tr><tr><td>DiP运动规划</td><td>预训练扩散规划器生成下一小段人体参考；实际执行反馈进入下一轮历史</td><td>使用官方DiP权重；没有从零训练扩散模型</td></tr><tr><td>物理控制</td><td>Isaac Gym中的人体控制器跟踪计划，通过PPO更新物理控制策略</td><td>官方权重评估、旧512微调与3072三次更新诊断分别记录</td></tr><tr><td>闭环验收</td><td>按触达、共用攻击、坐下、起身的官方状态统计</td><td>不将两次评估混算成新训练效果，也不把踢/拳视为独立估计</td></tr></table></div>'))
c.main.append(m);c.main.append(copy.deepcopy(source.find(id='closd')))
c.main.append(frag('<section id="training"><div class="eyebrow">训练配置与状态</div><h2>三个阶段必须区分</h2><div class="tablewrap"><table><tr><th>阶段</th><th>配置与检查点</th><th>状态与结论</th></tr><tr><td>旧缩容微调</td><td>512环境；epoch62000→66000</td><td>最终检查点已保存，训练退出139；前后匹配评估正常完成，但除起身外明显退化。不是官方同预算复现。</td></tr><tr><td>官方规模短程诊断</td><td>3072环境；三次更新；epoch62003</td><td>正常退出0并完成Gym清理；仅验证容量与生命周期，不验证4000轮训练效果。</td></tr><tr><td>待完成正式微调</td><td>3072环境；CFG2.5；epoch62000→66000，4000轮；后续4096环境评估</td><td>等待本轮当前实际场景的本地可视确认；尚未开始这一正式阶段，不能计为训练中或训练完成。</td></tr></table></div><p>固定官方源码与模型、PPO损失和奖励，外部适配负责串行动作加载、只读非阻塞遥测与退出清理。旧139的具体根因尚未证实；新短程正常退出并未证明旧异常的唯一原因。训练与监控分进程，监控故障不得阻塞计算。</p><p>本次拆分报告没有新增CLoSD TensorBoard曲线导出；DreamControl的reward/value/PPO曲线不移作CLoSD训练证据。CLoSD后续完整训练与logger统计须另行验收。</p></section>'))
pack='论文复现汇报资料_20261002/'
c.main.append(frag('<section id="artifacts"><div class="eyebrow">产物与追溯</div><h2>CLoSD各阶段保存的文件</h2><div class="tablewrap"><table><tr><th>文件</th><th>用途</th><th>保存与验收边界</th></tr><tr><td>DiP预训练模型、Humanoid.pth控制器</td><td>扩散规划和物理控制的官方先验权重</td><td>服务器保留路径与SHA；受许可模型及数据不随汇报包上传。</td></tr><tr><td>Humanoid_00066000.pth</td><td>旧512微调最终检查点</td><td>保存成功不等于训练正常退出；与后续退化评估一起保留。</td></tr><tr><td>epoch62003检查点与容量诊断JSON</td><td>3072环境三次更新和退出清理证据</td><td>不是正式4000轮结果。</td></tr><tr><td>官方评估日志与结果JSON</td><td>口径、模型SHA、退出码及每状态至少1000次重置成功率</td><td>本地报告链接直接读取归档指标；踢/拳共用计数。</td></tr><tr><td>实际场景遥测/显示状态</td><td>独立监控当前Isaac Gym角色和物体状态</td><td>实时状态抽样不是保存的评估MP4，也不是Isaac Gym原生Viewer或WebRTC。</td></tr></table></div><p><a href="'+href(pack+'reports/01_closd_reproduction.md')+'">完整CLoSD复现时间线</a> · <a href="'+href(pack+'results/r3_closd_official_eval4096_20261001_summary.json')+'">两组官方4096环境评估JSON</a></p></section>'))
c.main.append(frag('<section id="videos"><div class="eyebrow">可视化交付</div><h2>官方结果视频尚未补齐</h2><p>两组已发布权重的4096环境正式评估没有保存MP4，evaluation_video_verified=false。当前没有可作为这两次正式评估证据的本地视频。后续需按独立记录保存真实CLoSD场景、规划目标与执行过程，核验时长和来源后补入本报告。</p><p>本地7001页面是独立实际场景监控；限时推理预检结束后的STALE仅表示没有新帧，不是新的正式训练失败，也不能将旧预检画面称作当前正式训练直播。</p></section>'))
c.main.append(frag('<section id="failures"><div class="eyebrow">问题与后续</div><h2>退化与退出异常分别处理</h2><p>旧512环境微调后的触达、打击和坐下表现明显下降，不能声称微调有效；512与官方3072环境的交互预算不同，但尚未证明这是唯一退化原因。保存后的139退出缺少完整原生栈，不能直接确定为显存不足或某一API错误。</p><p>后续先完成当前实际场景可视验收，再通过既有唯一续跑入口执行3072环境4000轮及4096环境评估；不重复已完成的官方权重评估和容量诊断，也不重复旧512同配置训练。正式模型、退出码、逐任务指标、日志与视频均须核验。</p></section>'))
ap=frag('<section id="appendix"><div class="eyebrow">可复查证据</div><h2>CLoSD复现明细与来源</h2><input id="search" placeholder="搜索CLoSD证据"/></section>').section
for det in source.find(id='appendix').find_all('details',recursive=False):
 if det.summary and 'CLoSD' in det.summary.get_text():ap.append(copy.deepcopy(det))
c.main.append(ap);c.main.append(copy.deepcopy(source.find('footer')))
nav(c,'CLoSD实验报告',[('overview','复现概况'),('methods','闭环方法与输入'),('closd','评估与退化结果'),('training','配置与正式训练状态'),('artifacts','各阶段产物'),('videos','视频交付状态'),('failures','问题与后续'),('appendix','证据与来源')],DREAM)
assert [t.get_text() for t in c.find(id='closd').find_all('table')]==[t.get_text() for t in source.find(id='closd').find_all('table')]
assert not c.find(id='baseline') and not c.find_all('video')
for name,s in [(DREAM,d),(CLOSD,c)]:
 for a in s.select('a[href^="#"]'):
  assert s.find(id=a['href'][1:]),(name,a['href'])
 (ROOT/name).write_text(str(s),encoding='utf-8')
index=base('论文复现实验报告｜DreamControl与CLoSD')
index.find('script').decompose()
index.nav.append(frag(f'<b>论文复现实验报告</b><a href="{href(DREAM)}">DreamControl报告</a><a href="{href(CLOSD)}">CLoSD报告</a>'))
index.main.append(frag(f'<header id="top"><div class="eyebrow">研究汇报 · 独立论文档案</div><h1>条件人体运动生成 · 论文复现实验报告</h1><p class="subtitle">DreamControl与CLoSD分别汇报</p><p>两份报告共享本文件夹已有视频、曲线和结果文件；可离线打开，整体复制文件夹即可携带资料。</p></header><section><h2>DreamControl</h2><p>离线条件扩散生成人体参考 → G1重定向 → Isaac Lab PPO → 官方评估与MuJoCo迁移。已归档8/11控制任务；保留Ground-Pick-Top的0%成绩。</p><p>包含18段六类官方人体展示视频、29段执行视频、八项训练曲线，以及生成质量筛查和策略文件格式。</p><p><a class="metric" href="{href(DREAM)}">打开DreamControl实验报告 →</a></p></section><section><h2>CLoSD</h2><p>DiP短段扩散规划与RL物理控制闭环。两组官方4096环境权重评估已完成；旧512微调退化保留，新的3072环境正式微调待验收。</p><p>本报告区分官方权重评估、本地微调、容量诊断与视频尚未交付的边界。</p><p><a class="metric" href="{href(CLOSD)}">打开CLoSD实验报告 →</a></p></section>'))
# Preserve old bookmarks such as the PPO section while keeping two separate files.
legacy={x.get('id') for x in d.select('[id]')} - {'top'}
router=index.new_tag('script')
router.string='const legacy='+json.dumps(sorted(legacy))+';const id=decodeURIComponent(location.hash.slice(1));if(id==="closd")location.replace('+json.dumps(CLOSD)+'+location.hash);else if(legacy.includes(id))location.replace('+json.dumps(DREAM)+'+location.hash);'
index.body.append(router)
MAIN.write_text(str(index),encoding='utf-8')
manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_backup':str(BACKUP),'source_sha256':hashlib.sha256(BACKUP.read_bytes()).hexdigest(),'reports':[{ 'file':name,'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in [DREAM,CLOSD]],'dreamcontrol':{'archived_control_tasks':8,'human_videos':18,'execution_videos':29,'training_curve_tasks':8},'closd':{'official_evaluation_groups':2,'formal3072_training_complete':False,'evaluation_videos_verified':False},'results_changed':False,'media_duplicated':False}
(ROOT/'报告拆分清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(REPO/'results/report_split_20261002.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
