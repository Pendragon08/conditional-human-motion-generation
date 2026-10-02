from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, unquote
import hashlib, json, shutil, datetime

ROOT = Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
REPORT = ROOT / '实验结果汇报.html'
PROOF = ROOT / '论文复现汇报资料_20261002/生成与验收'
backup = ROOT / '实验结果汇报_扩散输入与参考动作说明前备份.html'
if not backup.exists(): shutil.copy2(REPORT, backup)
s = BeautifulSoup(REPORT.read_text(encoding='utf-8'), 'html.parser')
def fragment(text): return BeautifulSoup(text, 'html.parser')
old = s.find(id='dreamcontrol-input-reference')
if old: old.decompose()
block = fragment('''
<div id="dreamcontrol-input-reference">
<h3>扩散模型输入与66条G1参考动作的阶段划分</h3>
<div class="note"><b>Kick的66条G1参考动作是PPO的跟踪参考库，不是OmniControl扩散生成的条件输入。</b>本次官方基线直接使用作者发布的<code>Kick_sim2</code>参考库训练控制策略，没有将66条G1动作输入扩散模型，也没有在这轮基线中重新生成它们。</div>
<div class="tablewrap"><table><thead><tr><th>阶段</th><th>输入与来源</th><th>输出 / 作用</th></tr></thead><tbody>
<tr><td><b>人体条件扩散生成</b></td><td>任务文本，例如“A person stands and kicks with his right leg.”；指定关节、位置和时刻的空间—时间提示；随机噪声。完整G1参考动作不作为本Kick生成流程的输入。</td><td>重定向前的22关节人体运动轨迹。文本规定动作语义，空间—时间提示规定关键关节应何时到达哪里。</td></tr>
<tr><td><b>人体运动先验</b></td><td>作者发布的OmniControl预训练权重；论文说明其训练数据为HumanML3D。</td><td>权重中已学习人体动作分布。本次使用已有先验，没有从零训练或微调该扩散模型；先验不等于66条G1参考文件。</td></tr>
<tr><td><b>重定向、筛选与修整</b></td><td>生成的人体轨迹＋G1机器人模型；按照官方任务流程转换并处理。</td><td>可供控制训练读取的G1根位置、根姿态和关节等参考，保存在<code>*_sim2/*.pkl</code>。本地作者发布资产中Kick人体端有100条，最终G1参考库有66条；这是两个阶段的数量。</td></tr>
<tr><td><b>Isaac Lab PPO训练</b></td><td>最终G1参考库、机器人及场景状态、官方奖励与任务配置。</td><td>通过仿真交互优化动作跟踪和任务控制策略。这里训练的是控制网络，不是OmniControl扩散模型。</td></tr>
</tbody></table></div>
<div class="method-flow"><span>文本＋关节位置/时刻＋噪声</span><b>→</b><span>预训练OmniControl生成人体轨迹</span><b>→</b><span>G1重定向、筛选与修整</span><b>→</b><span>G1参考库</span><b>→</b><span>PPO控制训练</span></div>
<p><b>本次复现实际执行了什么：</b>得到85.3%成绩的Kick基线，读取作者发布的66条G1参考，从零进行8192环境、2000轮PPO训练，再做1000环境、500帧官方评估。报告中的六类官方人体视频，是对作者发布的人体生成输出进行本地渲染；它们不是本轮重新执行扩散推理的证据。此前本课题自行生成的两批Kick共200条另行归档，没有混入这一官方66条基线。</p>
<details><summary>生成代码核验与方法边界</summary><p>官方<code>collect_Kick.sh</code>先调用<code>sample.generate</code>，然后调用重定向和修整脚本。Kick预定义条件由<code>text_control_example.py</code>构造文本与右脚提示。生成入口使用<code>init_image=None</code>与<code>noise=None</code>，采样器内部产生随机初始噪声；这里没有加载<code>Kick_sim2</code>的66条G1轨迹作为初始动作。生成代码仍会读取人体数据的表示和归一化统计，这不等于将66条G1轨迹作为条件输入。</p><p>以上区分针对Kick的OmniControl生成路线。原论文对部分分布外任务还描述了基础站立/弯腰轨迹结合IK优化的构造方法；不能将这些特殊任务路径与Kick的66条控制参考混为一谈。</p></details>
<p class="source">依据：<a href="https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf">DreamControl §III-A：文本/时空条件、人体先验与重定向</a> · <a href="https://github.com/GenRobo/DreamControl/blob/main/TrajGen/sample/generate.py">官方扩散采样入口</a> · <a href="https://github.com/GenRobo/DreamControl/blob/main/TrajGen/collect_Kick.sh">Kick生成流程</a> · <a href="https://github.com/GenRobo/DreamControl#phase-1-generate-reference-trajectories">官方说明：可跳过生成，直接使用已发布参考训练RL</a></p>
</div>
''').find(id='dreamcontrol-input-reference')
methods = s.find(id='methods')
closd_title = next(h for h in methods.find_all('h3', recursive=False) if 'CLoSD' in h.get_text())
closd_title.insert_before(block)

generation = s.find(id='generation')
old = generation.find(id='generation-reference-distinction')
if old: old.decompose()
generation.find('p', recursive=False).insert_after(fragment('<p id="generation-reference-distinction"><b>数量与用途区分：</b>Kick原始人体样本100条，作者发布的最终G1控制参考66条。本节视频展示前者；PPO基线使用后者。66条不是扩散条件输入，也不是扩散模型的预训练数据。<a href="#dreamcontrol-input-reference">查看各阶段输入、输出与本次复现范围</a>。</p>').p)

baseline = s.find(id='baseline')
old = baseline.find(id='baseline-reference-source')
if old: old.decompose()
baseline.find('p', recursive=False).insert_after(fragment('<p id="baseline-reference-source"><b>参考库来源：</b>本节官方控制基线直接读取作者发布、已重定向和修整的G1参考库。Kick为66条；这些文件供PPO跟踪，未作为OmniControl输入重新扩散生成。<a href="#dreamcontrol-input-reference">完整流程说明</a>。</p>').p)

artifacts = s.find(id='artifacts')
old = artifacts.find(id='artifact-input-reference-note')
if old: old.decompose()
artifacts.find('p', recursive=False).insert_after(fragment('<div class="note" id="artifact-input-reference-note"><b>两条产物来源须分开：</b>自行扩散推理导出人体<code>results.npy</code>；官方控制基线则直接读取作者已发布的G1<code>.pkl</code>参考。Kick的66个<code>.pkl</code>属于后者，不是生成模型的输入。下表第1–2步的“两批各100条”指本课题此前独立Kick推理记录，不代表本轮六类官方人体样本重新生成。</div>').div)
REPORT.write_text(str(s), encoding='utf-8')

missing=[]; checked=0
for tag in s.select('[href],[src],[poster]'):
    for attr in ['href','src','poster']:
        val=tag.get(attr)
        if not val: continue
        u=urlsplit(val)
        if u.scheme: continue
        if u.path:
            checked+=1
            if not (ROOT/unquote(u.path)).exists(): missing.append(val)
        elif u.fragment:
            if not s.find(id=u.fragment): missing.append(val)
assert not missing, missing
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'local_links_checked':checked,'missing_links':missing,'human_videos_preserved':len(s.find(id='generation').find_all('video')),'robot_videos_preserved':len(s.find(id='videos').find_all('video')),'task_pairs_preserved':len(s.select('.task-video-block')),'kick_diffusion_inputs':['task text','right-foot spatiotemporal hint','random noise'],'kick_rl_reference_count':66,'official_baseline_reused_released_references':True,'pretrained_prior':'OmniControl trained on HumanML3D','supersedes_report_hash_receipt':'逐任务展示最终文件核验.json'}
(PROOF/'扩散输入与参考动作说明核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__), PROOF/Path(__file__).name)
print(json.dumps(receipt,ensure_ascii=False))
