"""Add source-grounded PPO interpretation to the existing offline teacher report."""
import hashlib, html, json, shutil
from pathlib import Path
from bs4 import BeautifulSoup

REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
REPORT=ROOT/'实验结果汇报.html'
RESULT=REPO/'results/ppo_training_interpretation_20261002'
PROOF=ROOT/'论文复现汇报资料_20261002/PPO原理与策略导出'
data=json.loads((RESULT/'six_task_curve_analysis.json').read_text(encoding='utf-8'))
evidence=json.loads((RESULT/'source_and_policy_format_evidence.json').read_text(encoding='utf-8'))
before=REPORT.read_text(encoding='utf-8');soup=BeautifulSoup(before,'html.parser')
before_counts={k:len(soup.select(f'#{k} video')) for k in ['generation','videos']}
backup=ROOT/'实验结果汇报_PPO解读与策略格式补充前备份.html'
if not backup.exists():shutil.copy2(REPORT,backup)
for ident in ['ppo-metric-priorities','ppo-six-task-interpretation','ppo-optimization-flow','frozen-policy-file-formats']:
 old=soup.find(id=ident)
 if old:old.decompose()

def table(head,rows):
 return '<div class="tablewrap"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def add_fragment(content):return BeautifulSoup(content,'html.parser').find()

priorities='''<div id="ppo-metric-priorities"><h3>训练曲线的阅读顺序与指标含义</h3>
<div class="note"><b>先看回报、回合长度和任务执行，再看优化诊断。</b>训练曲线回答“是否学到了当前奖励所鼓励的行为”；冻结策略的正式评估、接触和跨仿真测试回答“任务是否真正完成、能否迁移”。二者应结合汇报。</div>'''+table(
 ['重点','真实logger指标','什么变化值得关注','本次解释边界'],[
 ['① 行为改善','<code>Train/mean_reward</code>＋<code>Train/mean_episode_length</code>','同一任务中reward上升、回合延长；持续大幅回落或回合骤降提示退化。','reward是最近完成回合的累计回报均值；不是任务成功率。长度接近500步＝接近10秒存活，不等于动作或接触合格。'],
 ['② 跟踪与任务奖励分项','<code>Episode_Reward/*</code>：关节/关键点偏差、位置/方向、touch_goal、feet_slide等','总reward改善时，同时核对跟踪、目标与滑脚等分项，防止只有存活项变好。','这些是加权奖励/惩罚贡献，不全是原始米/弧度误差；不同项、任务不可直接比数值大小。'],
 ['③ Critic预测质量','<code>Loss/value_function</code>（Value loss）','减小通常表示更能拟合当前采样的回报目标；尖峰需要结合reward、恢复点与策略变化诊断。','它是价值预测损失，不是状态价值V(s)的曲线；不能读作“value越大/越小，动作越好”。本实现采用裁剪价值损失。'],
 ['④ Actor更新诊断','<code>Loss/surrogate</code>（PPO代理损失）','与reward、熵和学习率共同观察；可以为负、波动或靠近零，无须单调下降。','是负的裁剪代理目标，不是分类误差，也不是PPO所有损失之和。靠近零既可能更新变小，也可能受裁剪/优势标准化影响。'],
 ['⑤ 探索是否适当','<code>Loss/entropy</code>＋<code>Policy/mean_noise_std</code>','探索逐渐减少较常见；过早塌缩、后期突增要联系实际reward判断。','熵是多维动作分布的统计量。更低不自动更好，也不能比较不同任务的动作质量。'],
 ['⑥ 更新幅度与步长','<code>Loss/learning_rate</code>；算法内部按KL自适应','名义0.001不是恒定步长；本次末100轮平均实际学习率约2.4×10⁻⁵—6.9×10⁻⁵。','KL、clip fraction、explained variance、实际梯度范数未在这六份logger中单独导出；不能声称已从曲线核验这些诊断。'],
 ['⑦ 最终效果','冻结模型官方成功率＋机器人视频＋MuJoCo记录','与曲线一起给出实际任务判据、摔倒、抬物与接触。','不同环境数量、重置和参考口径的Isaac与MuJoCo成绩不可直接相减当作迁移成功率。']
 ])+'<p>同一任务先后阶段可比较；不同任务的奖励函数和尺度不同，不能用Pick的reward高于Kick来排名。损失变小与长回合只能支持当前训练目标下的改善，不能替代自然动作、碰撞接触或动力学验收。</p></div>'

interpretations={
 'Kick':('持续改善，末期接近平台。','85.3%；仍有未成功环境。'),
 'Punch':('恢复点1600后，1600—1699均值降至3.462、回合约446步；末期回到约499步，但reward仍低于恢复前高位。','87.2%；恢复后能执行，但未证明恢复前后完全等效。'),
 'Jump':('约1610—1709达到较高回报，之后回落；回合长度仍接近满时限，不能称单调收敛。','Isaac 100%；MuJoCo约2.9秒摔倒，跨平台稳定性尚不足。'),
 'Sit':('末期小幅波动、接近回报平台，价值损失较低。','Isaac 100%为骨盆目标判据；MuJoCo到达目标但无持续座椅接触。'),
 'Pick':('中后期回报较高，末期较最佳100轮均值小幅下降，仍保持长回合。','Isaac 93.9%；MuJoCo 67条参考可运行完整10秒，但未抬物。'),
 'Bimanual-Pick':('900—999均值3.251、1400—1499为4.955、末100为5.990；结束时仍在改善，不能宣称已充分收敛。','Isaac 99.9%（单参考）；MuJoCo完整10秒但未抬物，不能等同双手搬运成功。')}
rows=[]
for t in data['tasks']:
 task=t['task'];m=t['metrics'];r=m['Train/mean_reward'];v=m['Loss/value_function'];length=m['Train/mean_episode_length'];best=t['reward_best100']
 rows.append([task,f"{r['first100_mean']:.3f} → {r['last100_mean']:.3f}",
 f"{best['mean']:.3f}（{best['start']}—{best['end']}）<br>末期差值 {best['last_minus_best']:+.3f}",
 f"{v['first100_mean']:.4f} → {v['last100_mean']:.6f}<br>末期回合 {length['last100_mean']:.1f}/500步",
 interpretations[task][0],interpretations[task][1]])
interpretation='''<div id="ppo-six-task-interpretation"><h3>六项真实曲线：改善程度与未解决问题</h3><p>以下数字直接从本地TensorBoard导出的2000轮CSV重算。前/末阶段取各100轮算术均值；“最佳100轮”只是训练回报的最高滑动窗口，不是已评估的最佳策略，也未据此更换最终检查点。</p>'''+table(
 ['任务','reward前100→末100','最高100轮均值 / 末期差值','Value loss / 回合','曲线判断','独立执行证据'],rows)+'''
<p><b>整体判断：</b>六任务回合均由约64—79步提高到约499—500步，value loss由约0.69—1.06降至约0.00070—0.00173，支持存活与当前回报预测改善。<b>Punch的恢复波动、Jump的后期回落和Bimanual-Pick的继续上升必须分别保留</b>，不能统一写成“全部已经稳定收敛”。上述窗口差异为描述统计，未做多随机种子显著性检验。</p>
<p>Punch曲线由初次0—1599轮与恢复1600—1999轮拼接，原始重叠分支已保留。曲线能定位恢复后的退化与恢复过程；其具体成因尚未完成因果诊断，不仅凭恢复时间归因于某一参数或网络故障。</p>
<p>以Kick为例，滑脚加权惩罚前/末100轮为−0.00998 → −0.02133，关节偏差惩罚−0.12006 → −0.17923；总回报提高并不意味着每个分项都改善。这些分项受持续执行时间、权重和状态分布影响，不能直接当作滑动距离或关节误差增加的定量测量。</p>
<p><a href="论文复现汇报资料_20261002/PPO原理与策略导出/six_task_curve_analysis.json">六任务窗口统计、CSV哈希及未记录诊断</a></p></div>'''

optimization='''<div id="ppo-optimization-flow"><h3>PPO怎样把参考动作变成可执行的控制策略</h3>
<p><b>参考动作固定，控制策略学习。</b>扩散模型先生成运动，重定向得到G1参考；PPO随后在物理仿真中训练机器人怎样跟踪这些参考。Actor π<sub>θ</sub>根据机器人状态和当前/未来参考输出动作分布，Critic V<sub>φ</sub>估计未来折扣回报。这里更新θ、φ及探索噪声，不更新OmniControl权重，也不逐次修改人体轨迹。</p>
<div class="method-flow"><span>状态＋参考</span><b>→</b><span>Actor采样动作</span><b>→</b><span>物理执行、奖励</span><b>→</b><span>GAE优势 / 回报目标</span><b>→</b><span>裁剪PPO更新</span><b>→</b><span>新策略重新采样</span></div>
<ol class="ppo-steps">
<li><b>采集当前策略的真实交互。</b>8192个环境各采24个控制步（每个环境约0.48秒仿真时间），得到196608条状态、动作、奖励、终止记录，并保存旧策略动作概率与旧价值估计。训练从高斯分布采样动作，以探索不同控制方式；参考在观测中提供当前与未来目标。</li>
<li><b>判断哪些动作比预期更好。</b>Critic提供基线；GAE（γ=0.99、λ=0.95）结合逐步奖励和价值预测得到优势A。A为正表示比当前价值预测更好，负表示更差。回报目标由未标准化优势加旧价值构成；随后整个rollout的优势标准化。真实终止截断后续回报，时间上限按实现做价值bootstrap。</li>
<li><b>有限度地提高有利动作的概率。</b>比较同一动作在新旧策略下的概率比，最大化裁剪代理目标：鼓励正优势动作、抑制负优势动作，并限制过度利用同一批数据带来的更新激励。ε=0.2的裁剪区间是0.8—1.2；这是目标函数裁剪，<b>不是所有概率比或KL都被硬性限制在这个区间</b>。</li>
<li><b>联合更新Actor与Critic。</b>实际最小化“负裁剪代理目标＋1.0×价值损失−0.008×熵”。价值损失训练Critic预测回报，熵项保留探索。每批分4个mini-batch（每个49152条）、重复5个epoch：每个PPO迭代共20次Adam更新；梯度范数上限1.0。KL目标0.01用于自适应学习率，不是已在logger中展示的KL曲线。</li>
<li><b>重新交互并冻结验证。</b>用更新后的策略收集新rollout，再重复2000个PPO迭代；因此本配置约40000次优化器更新、393216000条环境转移，不能把“2000轮”理解为只做2000次梯度下降。最终冻结Actor和归一化统计，以确定性均值动作进行官方评估与MuJoCo验证；评估不继续学习。</li>
</ol>
<details><summary>PPO公式与本次实现对应关系</summary>
<div class="ppo-equations">δ<sub>t</sub> = r<sub>t</sub> + γ V<sub>old</sub>(s<sub>t+1</sub>) − V<sub>old</sub>(s<sub>t</sub>)<br>
A<sub>t</sub> = δ<sub>t</sub> + γλ A<sub>t+1</sub>；R̂<sub>t</sub> = A<sub>t</sub> + V<sub>old</sub>(s<sub>t</sub>)<br>
q<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>old</sub>(a<sub>t</sub>|s<sub>t</sub>)<br>
L<sup>clip</sup> = E[min(q<sub>t</sub>A<sub>t</sub>, clip(q<sub>t</sub>,0.8,1.2)A<sub>t</sub>)]<br>
实际loss = −L<sup>clip</sup> + L<sub>value</sub> − 0.008 H(π<sub>θ</sub>)</div>
<p>以上递推写的是非终止步骤；终止mask、超时bootstrap和优势标准化以归档源码为准。L<sub>value</sub>取普通平方预测误差与裁剪价值平方误差的较大者再求均值。每次日志中的surrogate/value/entropy是本轮20个mini-batch更新的平均值，不是完整训练过程的一次性固定目标。</p></details>
<p>原理依据：<a href="https://arxiv.org/abs/1707.06347">Schulman等，PPO原始论文（2017）</a>；具体超参数、GAE、损失和logger解释以本次服务器归档的RSL-RL/Isaac Lab源码与各agent.yaml为准，不能将新版文档默认值当成本次实验设置。</p></div>'''

format_rows=[]
for task,item in evidence['exports'].items():
 assert item['interface']['policy_sha256']==item['files']['exported/policy.pt']['sha256']
 format_rows.append([task,html.escape(item['run']),f"{item['interface']['obs_dim']} → 29",
 f"{item['files']['exported/policy.pt']['bytes']/1024**2:.2f} MiB / {item['files']['exported/policy.onnx']['bytes']/1024**2:.2f} MiB"])
formats='''<div id="frozen-policy-file-formats"><h3>冻结策略文件：训练检查点、TorchScript与ONNX</h3>
<p><b>本次MuJoCo实际使用 <code>exported/policy.pt</code>（TorchScript）</b>，由官方导出器通过<code>torch.jit.script</code>保存、通过<code>torch.jit.load(...).eval()</code>加载。<code>.pt</code>只是扩展名，训练检查点与部署模型虽都叫“.pt”，内容和加载方法不同。</p>'''+table(
 ['文件 / 格式','保存内容','用途 / 限制'],[
 ['<code>model_1999.pt</code><br>PyTorch训练检查点','字典中的model_state_dict（Actor/Critic、训练动作分布参数）、optimizer_state_dict、iter、infos，以及观测归一化统计。','使用torch.load并恢复训练器；可做断点续训或评估。它不等于已经封装的TorchScript推理模型。'],
 ['<code>exported/policy.pt</code><br>TorchScript部署模型','Actor＋训练观测normalizer；输入状态/参考观测，输出确定性均值动作。当前为前馈网络。','本次跨仿真实际加载格式。无Critic、优化器、奖励函数、完整参考轨迹或仿真场景；不能靠这个文件单独恢复完整PPO训练。'],
 ['<code>exported/policy.onnx</code><br>ONNX计算图','同一Actor＋normalizer的ONNX导出；本导出器opset11，输入obs、输出actions。','六个run均已核实文件存在且非空；本次MuJoCo没有使用ONNX后端，不能宣称ONNX执行等价性已验证。'],
 ['<code>&lt;Task&gt;_interface.json</code> / <code>&lt;Task&gt;_references.npz</code>','JSON记录观测顺序、维度、关节/驱动、参考来源；NPZ保存根位置、姿态、关节、关键点等参考数组。','部署还需机器人MJCF/XML、相同观测/动作映射、驱动和控制周期；策略文件没有包含这些场景和参考内容。']
 ])+'''
<p>输入先用保存的均值/方差归一化，再送入Actor；推理期间这些统计与权重保持固定，不添加训练探索噪声。29维输出对应27个身体关节动作及左右手各一个开合动作通道，不是直接的29维关节力矩。输出还需经任务动作映射与执行器作用到物理系统。</p>'''+table(
 ['任务','冻结策略来源run','观测维数 → 动作维数','TorchScript / ONNX文件大小'],format_rows)+'''
<p>观测长度随任务目标项不同，不能给六任务强行使用同一输入维数。此前MuJoCo接口核验已比较导出的Actor、normalizer张量与训练检查点，并检查观测/动作快照；这支持接口一致性，不替代跨引擎动力学与任务接触验收。</p>
<p><b>保存位置：</b>六项部署模型在服务器相应<code>logs/rsl_rl/g1/&lt;run&gt;/exported/</code>目录；Kick来自原运行目录，其余来自新项目目录。本地本汇报文件夹保存完整路径、文件大小、SHA-256和源码核验资料，<b>没有在本次报告修订中下载策略权重</b>。</p>
<p><a href="论文复现汇报资料_20261002/PPO原理与策略导出/source_and_policy_format_evidence.json">六项模型路径、文件格式依据及SHA-256</a> · <a href="论文复现汇报资料_20261002/PPO原理与策略导出/源码核验/exporter.py">本次实际导出源码</a> · <a href="https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html">Isaac Lab官方导出接口说明</a></p></div>'''

training=soup.find(id='training')
first_table=training.find('div',class_='tablewrap')
first_table.insert_before(add_fragment(optimization))
first_table.insert_after(add_fragment(priorities))
first_chart=training.find('details',class_='training-chart')
first_chart.insert_before(add_fragment(interpretation))
artifact=soup.find(id='artifacts')
artifact.append(add_fragment(formats))
for row in artifact.select('tbody tr'):
 cells=row.find_all('td')
 if cells and cells[0].get_text().startswith('8 冻结策略'):
  cells[1].clear();cells[1].append(add_fragment('<span>Actor＋normalizer → <code>exported/policy.pt</code>（TorchScript）及 <code>policy.onnx</code>；参考→NPZ；MuJoCo场景→MJCF/XML；输出→JSON/MP4</span>'))
style=soup.new_tag('style');style.string='''#ppo-optimization-flow,#ppo-metric-priorities,#ppo-six-task-interpretation,#frozen-policy-file-formats{scroll-margin-top:25px}.ppo-steps{padding-left:24px}.ppo-steps li{margin:12px 0}.ppo-equations{padding:18px;border-left:4px solid #16828d;background:#eff7f9;overflow-wrap:anywhere;font-family:Cambria,serif}#ppo-six-task-interpretation table{min-width:980px}#ppo-metric-priorities table{min-width:800px}#frozen-policy-file-formats code{overflow-wrap:anywhere}'''
existing=soup.find(id='ppo-report-extra-style')
if existing:existing.decompose()
style['id']='ppo-report-extra-style';soup.head.append(style)
REPORT.write_text(str(soup),encoding='utf-8')
after=BeautifulSoup(REPORT.read_text(encoding='utf-8'),'html.parser')
assert {k:len(after.select(f'#{k} video')) for k in before_counts}==before_counts
assert len(after.select('#six-task-human-quality-screening tbody tr'))>=6
for ident in ['ppo-metric-priorities','ppo-six-task-interpretation','ppo-optimization-flow','frozen-policy-file-formats']:assert len(after.select('#'+ident))==1
summary={'report_path':str(REPORT),'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'preserved_video_counts':before_counts,'sections_added':['ppo-metric-priorities','ppo-six-task-interpretation','ppo-optimization-flow','frozen-policy-file-formats'],'training_mutated':False,'policy_weights_downloaded':False}
(RESULT/'report_revision_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(RESULT/'report_revision_manifest.json',PROOF/'report_revision_manifest.json')
print(json.dumps(summary,ensure_ascii=False))
