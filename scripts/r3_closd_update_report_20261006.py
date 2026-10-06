"""Update only the separate CLoSD teacher report from verified archived outputs."""
import hashlib, html, json, shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from bs4 import BeautifulSoup

REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
PAGE=ROOT/'CLoSD_实验结果汇报.html'
SUB='CLoSD补充结果_20261006'
SUMMARY=json.loads((REPO/'results/closd_supplementary_20261006/summary.json').read_text(encoding='utf-8'))
assert SUMMARY['total_videos']==9 and not SUMMARY['formal3072_training_started']
backup=REPO/'artifacts/closd_recordings_20261006/CLoSD_report_before_20261006.html'
if not backup.exists():shutil.copy2(PAGE,backup)
soup=BeautifulSoup(PAGE.read_text(encoding='utf-8'),'html.parser')
def link(sub,file):return quote(SUB+'/'+sub+'/'+file,safe='/')
def section(id,text):soup.find(id=id).replace_with(BeautifulSoup(text,'html.parser').find('section'))
def filelink(sub,name,label):return f'<a href="{link(sub,name)}">{label}</a>'

soup.select_one('nav small').clear();soup.select_one('nav small').append('本地离线档案 · 结果快照：2026-10-06')
for a in soup.select('nav a'):
    if a.get('href')=='#training':a.string='04 PPO曲线与训练状态'
    if a.get('href')=='#videos':a.string='06 五类任务执行演示'
header=soup.find('header')
header.find('p',class_='subtitle').string='扩散运动规划与物理控制闭环：官方评估、执行视频及微调曲线'
for p in header.find_all('p',recursive=False):
    if not p.get('class'):p.string='已归档结果截至2026年10月6日。补齐两组冻结策略执行演示与旧512环境训练曲线；新3072环境正式微调尚未开始。'
header.find(class_='hero-grid').clear()
header.find(class_='hero-grid').append(BeautifulSoup('<div><strong>2组</strong><span>官方权重4096环境评估</span></div><div><strong>9段</strong><span>8段完整演示＋1段起身截取</span></div><div><strong>3998点</strong><span>旧512训练每项主要曲线</span></div><div><strong>待验收</strong><span>3072环境正式微调</span></div>','html.parser'))
section('overview',f'''<section id="overview"><div class="eyebrow">复现概况</div><h2>官方评估、执行演示与训练诊断分别归档</h2>
<p>两组已发布官方权重均已完成4096环境、500帧、CFG7.5、seed0的官方评估并正常退出。2026年10月6日，使用相同源权重另行运行12环境、seed42的冻结策略推理，补录触达、踢击、拳击、坐下及起身状态的执行演示；不更新策略、不重跑原正式评估。</p>
<div class="note">新增8段完整视频，每段30秒、900帧、640×480、30fps；另有1段8.27秒、248帧的起身状态截取。视频由真实Isaac Gym刚体与物体状态独立三维渲染生成，所有帧均已解码核验。原两次4096环境评估未录视频，该历史事实与成绩不变。</div>
<p>旧512环境微调保存epoch66000后退出139，匹配评估显示多数任务退化；其原始TensorBoard标量现已导出。新3072环境4000轮正式微调仍待实时可视验收。本轮用户反馈“画面不动或打不开”，因此没有记录Gate通过，也没有启动正式训练。</p>
<p>{filelink('证据','summary.json','视频、模型哈希与来源清单')}</p></section>''')
section('training',f'''<section id="training"><div class="eyebrow">训练证据</div><h2>旧512环境PPO曲线：代理奖励提高，任务评估仍退化</h2>
<div class="tablewrap"><table><tr><th>实验</th><th>配置与状态</th><th>结论范围</th></tr>
<tr><td>旧本地微调</td><td>512环境，epoch62000→66000；最终保存后退出139</td><td>保留退化结果，不能称正常训练完成或论文同配置复现。</td></tr>
<tr><td>3072容量诊断</td><td>仅3次更新，epoch62003；退出0</td><td>验证短程运行与清理，不验证4000轮训练效果。</td></tr>
<tr><td>新正式微调</td><td>3072环境、CFG2.5，追加4000轮；随后4096评估</td><td>尚未启动，本轮实时画面验收未通过。</td></tr></table></div>
<img class="curve-img" src="{link('训练曲线','old512_training_curves.png')}" alt="旧512环境微调的reward、episode length、actor loss、critic loss、clip fraction和判别器奖励曲线"/>
<p>图中浅色为原始标量，深色为向后100点均值。六项主要指标各3998点，logger的epoch范围为62004—66001；评估模型固定为已保存的epoch66000。下表对比前100点与后100点均值，不将日志点数当作额外正式训练预算。</p>
<div class="tablewrap"><table><tr><th>重点指标</th><th>前100点 → 后100点</th><th>变化解释</th></tr>
<tr><td>Reward：rewards0/iter</td><td>32.469 → 103.851</td><td>训练中的奖励目标提高，后期上升明显；奖励是任务混合代理，不能代替独立成功率。</td></tr>
<tr><td>回合长度</td><td>307.226 → 380.042</td><td>平均执行更久；没有单独证明触达、攻击或坐下完成。</td></tr>
<tr><td>Value/critic loss：c_loss</td><td>0.03043 → 0.00723</td><td>价值函数对当前采样回报的拟合误差下降；不等于动作质量提高。</td></tr>
<tr><td>PPO actor loss：a_loss</td><td>−0.04698 → −0.02826</td><td>剪切代理目标的优化统计；其符号与尺度不能按“越小越好”判断任务成绩。</td></tr>
<tr><td>剪切比例：clip_frac</td><td>0.21414 → 0.16664</td><td>被剪切的样本比例减少；仅说明策略更新统计变化。</td></tr>
<tr><td>动作先验判别器奖励</td><td>0.21096 → 0.17530</td><td>该先验代理未同步改善；需结合自然性、接触与任务成功率。</td></tr></table></div>
<div class="warning">同口径512环境固定模型评估：触达100%→0%，共用STRIKE87%→5%，坐下77%→7%，起身56%→69%。因此旧微调多数任务退化，reward上升、critic loss下降均不能推翻该结果。任务分布、遗忘与更新配置仍需进一步诊断；尚未证明单一根因。</div>
<p>先看独立任务成功率和执行视频，再看reward/回合长度，最后用critic loss、actor loss与clip fraction诊断学习过程。官方已发布权重没有提供本次可用训练logger；新的3072训练未启动，均不绘制虚构曲线。</p>
<p>{filelink('训练曲线','scalar_summary.json','曲线统计JSON')} · {filelink('训练曲线','all_scalars.json.gz','全部51项原始标量导出')} · {filelink('训练曲线','event_source.json','TensorBoard来源及SHA256')} · {filelink('训练曲线','old512_training_curves.svg','矢量曲线图')}</p>
<details><summary>主要曲线CSV</summary><p>{' · '.join(filelink('训练曲线',n,n) for n in ['rewards0_iter.csv','episode_lengths_iter.csv','a_loss.csv','c_loss.csv','clip_frac.csv','disc_reward_mean.csv'])}</p></details>
<p>固定官方模型、PPO与奖励；外部适配负责串行动作加载、只读非阻塞遥测和退出清理。训练与显示保持独立，监控故障不应阻塞计算。</p></section>''')

groups=[]
names={'REACH':'触达／行走至目标','STRIKE_KICK':'踢击目标','STRIKE_PUNCH':'拳击目标','SIT':'坐下及后续状态','GET_UP':'起身状态'}
for task,label in names.items():
    cards=[]
    for entry in SUMMARY['models']:
        model=entry['model'];name='官方已发布微调权重' if 'multitask' in model else '官方未微调权重'
        clips=[x for x in entry['clips'] if x['task']==task]
        if clips:
            c=clips[0];vid=link('视频',c['file']);poster=link('视频',model+'_'+('SIT' if task=='GET_UP' else task)+'_300.jpg')
            note='完整固定环境记录，保留状态切换、重置与失败过程。' if task!='GET_UP' else '从坐下视频截取连续GET_UP状态（源step653—900）；不是额外样本或独立成功率。'
            cards.append(f'<article class="video-card"><h3>{name}</h3><video controls preload="metadata" poster="{poster}"><source src="{vid}" type="video/mp4"/></video><p>{c["duration_seconds"]:.2f}秒 · {c["frames"]}帧 · 30fps · 640×480。{note}</p><p><a href="{vid}">打开本地视频文件</a></p></article>')
        else:
            sit=link('视频',model+'_SIT.mp4')
            cards.append(f'<article class="video-card"><h3>{name}</h3><div class="pending-placeholder">本固定样本未触发起身状态</div><p>记录的坐下环境在900帧中始终为SIT，没有可导出的GET_UP片段。这不是对官方56%起身成功率的重新估计。</p><p><a href="{sit}">查看该环境完整30秒记录</a></p></article>')
    groups.append(f'<div class="task-video-block" id="closd-video-{task}"><h3>{label}</h3><div class="video-grid">'+''.join(cards)+'</div></div>')
section('videos',f'''<section id="videos"><div class="eyebrow">实际物理执行</div><h2>两组冻结策略 · 五类任务状态演示</h2>
<p>橙色为SMPL人体角色，蓝色为仿真中的目标物体或座椅。这是CLoSD在Isaac Gym中的实际执行状态，使用独立MeshCat渲染呈现；没有改成DreamControl的G1机器人，也没有在第二物理引擎重新计算，因而不是sim-to-sim测试。</p>
<p>12环境、seed42、CFG7.5、500帧回合；预先固定记录触达env4、踢击env1、拳击env2、坐下env0。每个模型推进900物理控制步（30秒），未按成功样本筛选。坐下可转入起身，攻击可转入HALT，500帧后可能重置。视频仅展示抽样执行，不重新估计4096环境正式成功率。</p>
<div class="note">原4096评估使用seed0；本次补录使用seed42。两个阶段各自保留配置与哈希；原评估的evaluation_video_verified=false保持不变，新增演示的video_verified=true单独记录。</div>
<div class="task-links">{''.join(f'<a href="#closd-video-{t}">{n}</a>' for t,n in names.items())}</div>{''.join(groups)}
<p>{filelink('证据','summary.json','全部视频时长、帧数、SHA256及模型来源')} · {filelink('证据','CLoSD_multitask_finetune_state_trace.json','微调权重状态与文本轨迹')} · {filelink('证据','CLoSD_no_finetune_state_trace.json','未微调权重状态与文本轨迹')} · {filelink('证据','source_state_validation.json','全部实际帧的有限值与姿态校验')}</p></section>''')

artifacts=soup.find(id='artifacts')
body=artifacts.find('table')
for tr in list(body.find_all('tr')):
    if tr.find('td') and tr.find('td').get_text() in {'补录视频MP4与summary.json','state_trace.json与resolved_config.yaml','TensorBoard标量JSON／CSV、PNG／SVG'}:tr.decompose()
for title,use,boundary in [
 ('补录视频MP4与summary.json','两组冻结策略的完整演示、起身截取、模型与视频SHA','独立于原4096正式评估；全部解码通过，不作为新的成功率统计。'),
 ('state_trace.json与resolved_config.yaml','每帧任务状态、文本、重置与实际推理配置','分享版本不含刚体数组、机器人网格或源模型权重。'),
 ('TensorBoard标量JSON／CSV、PNG／SVG','旧512微调六项主要曲线及51项原始标量','旧失败训练的诊断证据；不冒充新3072训练。')]:
    tr=soup.new_tag('tr')
    for text in [title,use,boundary]:td=soup.new_tag('td');td.string=text;tr.append(td)
    body.append(tr)
for p in list(artifacts.find_all('p')):
    if '微调演示配置' in p.get_text():p.decompose()
artifacts.append(BeautifulSoup(f'<p>{filelink("证据","CLoSD_multitask_finetune_resolved_config.yaml","微调演示配置")} · {filelink("证据","CLoSD_no_finetune_resolved_config.yaml","未微调演示配置")} · {filelink("证据","summary.json","补录结果清单")}</p>','html.parser'))
section('failures','''<section id="failures"><div class="eyebrow">边界与后续</div><h2>退出异常、性能退化与可视验收分别记录</h2>
<p>旧512微调保存后的139退出没有完整原生栈，其具体原因仍未证实；该固定策略多数任务退化是独立的性能问题。今回补录中，三次限定的原生相机检查均能产出短帧但在解释器退出时139，故停止该路线并保全失败日志，没有无界重试或运行完整原生相机任务。不能据此断言它与旧训练139同源。</p>
<p>随后采用无图形上下文的真实Isaac Gym状态导出与独立渲染，两组900帧推理均正常退出0，显式仿真清理通过、源模型SHA未变。离线观看视频不依赖实时网页，也不构成当前训练画面Gate。</p>
<div class="warning">2026-10-06本轮用户反馈实时场景“画面不动或打不开”，可视验收未通过。新的3072环境、4000轮正式微调尚未启动；只有当前真实场景持续变化并得到本地确认后，才使用既有唯一续跑入口继续这一剩余阶段。</div>
<p>两组已完成官方评估、3072短程容量诊断和旧512退化结果均已保全，不重复计算。受许可数据、模型权重及原始场景资源不上传GitHub。</p></section>'''.replace('今回','本次'))
appendix=soup.find(id='appendix')
for p in appendix.find_all('p'):
    if '当前仅运行30分钟9环境推理预检' in p.get_text():p.string='2026-10-06补录已经正常结束，实时页无新帧时不称训练停滞。新的3072正式微调仍待本轮真实场景可视确认，不能将离线演示替代Gate。'
for d in list(appendix.find_all('details')):
    if d.find('summary') and d.find('summary').get_text()=='2026-10-06 补录与训练曲线证据':d.decompose()
appendix.append(BeautifulSoup(f'<details><summary>2026-10-06 补录与训练曲线证据</summary><p>{filelink("证据","summary.json","核验清单")} · {filelink("训练曲线","scalar_summary.json","曲线统计")} · <a href="{link("证据","")}">全部日志与配置目录</a></p><p>原生相机失败日志、正常状态导出日志、两组配置和来源哈希保存在CLoSD补充结果目录。完整900帧与起身截取分别标记，不将额外片段计为新的样本。</p></details>','html.parser'))
PAGE.write_text(str(soup),encoding='utf-8')
manifestpath=ROOT/'报告拆分清单.json';manifest=json.loads(manifestpath.read_text(encoding='utf-8'))
for record in manifest['reports']:record['sha256']=hashlib.sha256((ROOT/record['file']).read_bytes()).hexdigest()
manifest['closd'].update(supplementary_execution_videos=9,supplementary_videos_verified=True,
 formal3072_training_complete=False,formal3072_training_started=False,current_local_scene_gate_passed=False,
 old512_training_curve_points=3998,evaluation_videos_verified=False)
manifest['updated_at']=datetime.now(timezone.utc).isoformat()
manifest['subsequent_updates']='CLoSD supplementary frozen-policy videos and archived512 TensorBoard curves; original official scores unchanged.'
manifestpath.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated independent CLoSD report; DreamControl report unchanged.')
