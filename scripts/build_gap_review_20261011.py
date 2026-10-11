"""Incremental literature comparison + measured diagnostic in existing report."""
from pathlib import Path
import json
import csv
import html
import re
import hashlib

repo=Path(__file__).resolve().parents[1]
lit=Path('D:/具身智能/R46_相关工作缺口核查_20261010')
out=repo/'results/gap_diagnostic_20261011/v3'
out.mkdir(parents=True,exist_ok=True)
source=json.loads((lit/'papers.json').read_text(encoding='utf-8'))
keys=['mdm','omnicontrol','dno','rgdno','physdiff','robotmdm','morph',
    'stablemotion','parc','claims','limmt','closd','diffusecloc','motionbricks',
    'pamotion','interprior','interphys','omniretarget','beyondmimic','omnixtreme']
papers=sorted([p for p in source if p['key'] in keys],key=lambda p:(p['year'],p['name']))
assert len(papers)==20 and all(p['core'] for p in papers)
bib=(lit/'references.bib').read_text(encoding='utf-8')
assert all(re.search(r'@\w+\s*\{\s*'+re.escape(p['key'])+r'\s*,',bib) for p in papers)
gaps=[
 dict(id='A',title='在较强条件下兼顾条件精度、物理质量与动作连续性',
      status='有已发表文献直接支持的剩余问题；具体改进尚未确定',
      done='任意关节控制、噪声优化、检索初始化和物理精化已经实现。不能再把“加条件”“加物理”“闭环”本身称作创新。',
      evidence=['rgdno','robotmdm','parc','stablemotion'],
      remaining='RG-DNO仍报告不自然动作与文本偏差；RobotMDM没有硬可行性保证；PARC的混合系数实验显示地形/接触符合与连续性之间的权衡；StableMotion修复可能改变原片段内容。这些问题的任务接口不同，不能称为所有方法相同失败。',
      test='在同一条件与输入上测量约束误差、语义/关键事件与物理质量的联合合格率，记录连续性和计算成本。本轮CLoSD无显式空间目标，只诊断参考接缝，尚不能直接验证条件保持。'),
 dict(id='B',title='对修正结果进行逐片段、局部质量验收，并区分修复失败与不适用输入',
      status='综合比较提出的候选缺口；不能宣称首次数据筛选',
      done='Morph已有MPJPE筛选及语义/跨任务分析；PARC已有接触/地形筛选；CLAIMS结合物理指标与VLM；LIMMT定义物理、多样性和复杂度三维质量。',
      evidence=['morph','parc','limmt','claims'],
      remaining='Morph对缺少环境支撑的动作采用回退，其表格仍有残余误差；PARC承认不自然行为未完全消除；LIMMT把低质量动作修复和利用列为未来方向。逐片段验收是否能找到平均分或完成标志遗漏的缺陷，需要实测。',
      test='保存所有首轮尝试，比较“未提前终止”与预先定义的误差/接触代理门控，保留局部尾部错误与失败前缀。接触代理必须人工或几何复核后才可判定真实缺陷。'),
 dict(id='C',title='对执行反馈与数据难度进行客观校准，减少把控制器不足误判为数据缺陷',
      status='候选研究方向；跨控制器与数据价值仍需独立实验',
      done='CLAIMS已做等数据预算与多跟踪器验证，LIMMT已证明小而高质量的子集有效。因此“此前未验证合成数据有用”不成立。',
      evidence=['claims','limmt','diffusecloc','motionbricks'],
      remaining='CLAIMS明确变量库缺少客观校准与全面覆盖，生成质量受基础模型限制。LIMMT的规则与自适应比例仍需更广语料验证。研究推论：执行失败由数据、任务适用范围与控制能力共同影响，不能用单一失败率对参考数据定性。',
      test='先固定控制器诊断，再在相同片段上换第二个开放跟踪器；最后等数据量、等训练步数、多种子测试。当前192次诊断不能证明训练收益或跨控制器不变性。')]
fresh_sources=[
 dict(paper='CLAIMS',url='https://arxiv.org/html/2602.21599v1',location='§3.2–3.5、§4.4、§4.5',note='固定MDM采样+物理/VLM筛选+控制反馈；已做等数据量对照。剩余限制为基础生成能力及人工库客观校准。'),
 dict(paper='Morph',url='https://arxiv.org/html/2411.14951v3',location='§3.3、§4与附录E/F',note='已做语义对齐和跨任务分析，不能误写成缺少这些实验；记录回退规则与残余物理指标。'),
 dict(paper='PARC',url='https://arxiv.org/html/2505.04002',location='§7.4、§8',note='接触符合与连续性权衡有消融表格；不自然行为、场景覆盖和实时速度仍是作者局限。'),
 dict(paper='LIMMT',url='https://proceedings.mlr.press/v306/guan26f.html',location='正式ICML论文附录B.4/C',note='本轮下载正式PDF：规则质量评价、仅两语料验证的比例启发式；修复低质量动作是未来方向。'),
 dict(paper='RG-DNO',url='https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Towards_Highly-Constrained_Human_Motion_Generation_with_Retrieval-Guided_Diffusion_Noise_Optimization_CVPR_2026_paper.html',location='本地正式PDF §5.5、结论末尾',note='本轮重新读取已下载正式PDF；检索与约束优化后仍有质量/语义偏差与额外优化代价。')]
comparison=dict(date='2026-10-11',direction='面向人形机器人的物理引导条件动作生成',
    mode='已有30篇调研中的20篇核心方法增量对比，不新建独立研究项目',
    papers=papers,gaps=gaps,fresh_source_checks=fresh_sources,
    survey_role='2篇已有综述用于方法/条件/表示/评价分类；最新gap由2025–2026正式论文直接支持。2025 Design and Artificial Intelligence综述的Q1状态未核验，只作分类辅助。',
    note='作者明确局限、方法范围和研究推论区分；候选缺口不等于已完成的新方法。')
(lit/'gap_comparison_20261011.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
fields=['name','year','venue','method','solved','remaining','location','url']
with (lit/'gap_comparison_20261011.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(papers)

e=lambda x:html.escape(str(x))
link=lambda p:f'<a href="{e(p["url"])}">{e(p["name"])} · {e(p["venue"])}</a>'
by={p['key']:p for p in papers}
sec='<section id="gap-validation-20261011"><h2>2026-10-11 增量核查：尚未完全解决的缺口与实验诊断</h2>'
sec+='<div class="callout"><b>结论：</b>不再把闭环数据回流、物理修正、语义评价或合成数据训练收益本身作为新意。先比较现有方法尚未消除的具体问题，再用固定基线检验问题是否在当前环境出现。</div>'
sec+='<p>'+e(comparison['mode'])+'。研究重点是动作生成与参考数据质量，跟踪与强化学习只作为验证工具。研究不限定高动态动作，不开展本轮实机实验。</p>'
sec+='<p><b>综述对照：</b>'+e(comparison['survey_role'])+' 已有TPAMI综述<a href="https://doi.org/10.1109/TPAMI.2023.3330935">Human Motion Generation: A Survey</a>与<a href="https://doi.org/10.1016/j.daai.2025.100007">Context-aware Human Motion Generation</a>支持分类框架，不能代替最新逐篇证据。</p>'
sec+='<h3>已解决与剩余问题的逐篇对比</h3><p>表中20篇均沿用已有正式会议/期刊记录；历史论文用于技术演变，当前缺口优先依赖2025–2026证据。未核验新的期刊影响因子，不填入猜测值。</p><div class="table-wrap"><table><thead><tr><th>论文/发表</th><th>已经解决什么</th><th>还留下什么（明确证据类型）</th><th>原文位置</th></tr></thead><tbody>'
for paper in papers:
    sec+='<tr><td>'+link(paper)+'</td><td>'+e(paper['solved'])+'</td><td>'+e(paper['remaining'])+'</td><td>'+e(paper['location'])+'</td></tr>'
sec+='</tbody></table></div><h3>收窄后的三项候选缺口</h3>'
for g in gaps:
    sec+='<div class="gap"><h4>'+e(g['id']+'：'+g['title'])+'</h4><p><b>证据等级：</b>'+e(g['status'])+'</p><p><b>已经改进：</b>'+e(g['done'])+'</p><p><b>仍然存在：</b>'+e(g['remaining'])+'</p><p><b>如何证伪/验证：</b>'+e(g['test'])+'</p><p>依据：'+'；'.join(link(by[k]) for k in g['evidence'])+'</p></div>'
sec+='<h3>本轮原文复核与纠正</h3><ul>'
for s in fresh_sources:
    sec+=f'<li><a href="{e(s["url"])}">{e(s["paper"])}</a> {e(s["location"])}：{e(s["note"])}</li>'
sec+='</ul><h3>实验设计与范围</h3><p>CLoSD的开放权重、物理执行与离线评价已在服务器跑通，因此本轮选择它做测量工具，不调用VLM API。三种子（11/12/13）×64环境，每个环境只取第一次尝试，最长约10秒；包括失败前缀。v3参考来自策略当前观测，在应用动作前记录，实际状态在该步物理执行后记录。预先定义阈值与敏感性分析，未修改模型、奖励或策略。</p><p><b>测量纠正：</b>首轮v2误用了继承奖励函数读取的独立数据缓存，该缓存不是生成参考。v2记录保留但排除。v3改用策略观测参考并检查调用顺序和时间变化；这不是算法修改或调阈值。</p>'
sec+='<ul><li>假设H1：没有提前终止，是否仍有较大尾部跟踪误差或接触速度代理异常？</li><li>假设H2：重规划边界的参考变化，是否伴随比内部帧更大的变化或跟踪误差？</li></ul>'
sec+='<div class="warning">这是SMPL模拟角色的T2M诊断；不是目标机器人动力学认证，不含显式空间目标输入。足部刚体中心速度是接触代理，旋转/转轴可能引起中心移动；不能据此直接判定真实表面滑移、接触力错误或任务语义错误。跳跃离地不能统一计为漂浮。</div>'
analysis=out/'analysis.json'
if analysis.exists():
    a=json.loads(analysis.read_text(encoding='utf-8'))
    sec+='<h3>实际完成的诊断结果</h3><div class="table-wrap"><table><tr><th>种子</th><th>首次尝试</th><th>未提前终止</th><th>跟踪误差门控通过</th><th>误差+接触代理门控通过</th><th>存活片段平均误差（mm）</th></tr>'
    for s,v in list(a['per_seed'].items())+[('合计',a['pooled'])]:
        sec+=f'<tr><td>{e(s)}</td><td>{v["n"]}</td><td>{v["completed_without_termination"]}</td><td>{v["tracking_gate_accepted"]}</td><td>{v["quality_proxy_gate_accepted"]}</td><td>{v["survivor_mean_gmpjpe_mm"]:.2f}</td></tr>'
    sec+='</table></div><p>门控预设：均值g-MPJPE≤100mm、帧误差p95≤200mm；接触代理使用持续净力&gt;10N的足部刚体中心，速度&gt;0.5m/s的占比≤5%。另保存27组合阈值敏感性分析；这些值是诊断选择，不是论文通用标准。</p>'
    v=a['pooled']
    sec+=f'<p><b>接缝观察：</b>参考在边界相对内部帧的平均速度差为{v["reference_boundary_minus_internal_speed_mps"]:.4f}m/s，实际动作差为{v["actual_boundary_minus_internal_speed_mps"]:.4f}m/s，边界与内部的平均跟踪误差差为{v["boundary_minus_internal_gmpjpe_mm"]:.2f}mm。仅作描述性关联，不作为因果证明。</p>'
    audit=json.loads((out/'hypothesis_audit.json').read_text(encoding='utf-8'))
    sec+='<p><b>H1的有限结论：</b>174次未提前终止的尝试中，15次未通过预先设定的平均/尾部跟踪门槛（15/174=8.6%）。这给出“未提前终止不能替代跟踪质量检查”的反例。它不能判定原因一定来自生成数据；控制器能力、延迟和任务适用范围也会影响误差。</p>'
    sec+='<p><b>H2未得到支持：</b>虽然参考在重规划边界变化较大，实际运动的平均速度差仅约0.058m/s，边界跟踪误差反而降低。源码利用已执行状态重新生成并对齐后续参考，计划更新不能当作物理身体的跳变。本轮不把“接缝导致跟踪恶化”写成已验证缺口。</p>'
    sec+=f'<p><b>接触代理敏感性：</b>27组预设阈值组合下，联合门控通过数为{audit["contact_proxy"]["sensitivity_min"]}–{audit["contact_proxy"]["sensitivity_max"]}/192（主设置85/192）。故85不是物理合格样本的真值计数，不能据此宣布“其余107条脚滑”。</p>'
    sec+='<h4>未提前终止但误差门控未通过的全部15条记录</h4><div class="table-wrap"><table><tr><th>种子/环境</th><th>平均g-MPJPE（mm）</th><th>帧误差p95（mm）</th><th>根相对姿态平均误差（mm）</th></tr>'
    for r in audit['h1']['rejected_non_terminated']:
        sec+=f'<tr><td>{r["seed"]}/{r["environment"]}</td><td>{float(r["mean_gmpjpe_mm"]):.1f}</td><td>{float(r["p95_gmpjpe_mm"]):.1f}</td><td>{float(r["mean_lmpjpe_mm"]):.1f}</td></tr>'
    sec+='</table></div><p>例如12/3为原地尽量跳高：平均误差120.9mm、p95为341.2mm；11/61为舞蹈：平均123.6mm、p95为342.5mm。13/25含上台阶与扶栏要求，但本次T2M场景没有相应支撑，不能把该样本的大误差归因于通用生成缺陷。原始记录及提示在受许可限制的本地/服务器文件中保留。</p>'
    sec+=f'<p><b>能支持的结论：</b>192次固定基线首轮诊断已完成，可以检验完成标志与上述质量指标是否等价。不能据此声称全面验证三项gap、胜过Morph/PARC/CLAIMS、合成数据提高训练效果或具备实机可执行性。</p>'
else:
    sec+='<p><b>当前状态：</b>独立四环境文档见证通过；192次正式诊断已启动，尚未在本报告中填入结果。</p>'
sec+='<h3>后续决定规则</h3><ol><li>先排除时序、坐标与接触代理误差，检查被门控拒绝的片段。若主要是合理脚尖转动，不能称为物理缺陷。</li><li>只有复核后仍有局部缺陷，才尝试局部修正/质量验收；与同输入的开放优化方法配对对比，冻结计算预算与评价输入。</li><li>最后开展等规模、等训练预算的数据对照，测试独立条件；在此之前不写“已经提升模仿学习”或把候选gap当作确定创新。</li></ol>'
sec+='<h3>基线最新完成记录</h3><p>上一轮使用发布权重，在4096并行环境中生成并保留20,000条官方完整、未终止轨迹，约2.25小时，未重新训练。10次官方评价：FID1.2972±0.018，R@3=0.6908±0.0035（作者评价程序给出的95%区间）。这是通过官方成功筛选后的结果，不是无条件成功率，也不能与不同输入集合直接推导改进收益。</p>'
sec+='<p>指标核查：作者代码中的mean_penetration是每片段最深值累加，不是平均穿透毫米值；floating是最低关节的高度代理，不能把约19.8mm直接解释为机器人漂浮。本轮避免用这些聚合代理单独认证执行。</p>'
sec+='</section>'
css='body{font:16px/1.8 Microsoft YaHei,system-ui;color:#223449;background:#eef3f7;margin:0}main{max-width:1350px;margin:auto;padding:24px}section{background:white;padding:28px}a{color:#12649c}h2,h3{color:#163e5d}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd8e2;padding:12px;vertical-align:top}th{background:#edf3f8}td:first-child{min-width:200px}.gap,.callout,.warning{padding:18px;margin:18px 0;background:#f2f8fa;border-left:4px solid #29818e}.warning{background:#fff7e7;border-color:#ce981d}@media print{body{background:white;font-size:10pt}main,section{padding:0}table{font-size:8pt}.table-wrap{overflow:visible}td,th{padding:5px}a{color:inherit}}'
standalone='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>文献缺口对比与CLoSD诊断</title><style>'+css+'</style><main>'+sec+'</main></html>'
(lit/'gap_comparison_and_pilot_20261011.html').write_text(standalone,encoding='utf-8')
(out/'gap_comparison_and_pilot_20261011.html').write_text(standalone,encoding='utf-8')
# Update the existing LaTeX research report via a small, auditable input file.
# This is an audit supplement, not a claimed contribution in the manuscript.
tex=r'''% Incremental evidence audit; not a manuscript novelty claim.
\section{2026年10月11日增量核查：剩余问题与基线诊断}
本次沿用已有文献库中的20篇正式会议或期刊方法论文，不新增重复BibTeX条目。综述用于方法分类，剩余问题优先由2025--2026年原文支持。作者明确局限、方法适用范围与本研究推论分别记录；本轮不提前确定创新方法。
\subsection{已经改进与仍需检验的问题}
\begin{enumerate}
\item \textbf{强条件下的联合质量。} 任意关节控制、噪声优化、检索和物理精化已经有方法。RG-DNO仍报告不自然动作与文本偏差，RobotMDM没有硬可行性保证，PARC的混合消融显示接触符合与连续性的权衡。不能把加入条件、物理信息或闭环本身称作创新。\citep{rgdno,robotmdm,parc}
\item \textbf{修正结果的局部验收与适用范围。} Morph已有MPJPE门控及语义、跨任务分析；PARC、CLAIMS与LIMMT已有质量评价或筛选。剩余候选问题是残余局部缺陷、不适用输入与修复失败的辨识，需要逐片段实验验证，不能声称首次数据筛选。\citep{morph,parc,claims,limmt}
\item \textbf{执行反馈和难度的客观校准。} CLAIMS已有等数据预算、多跟踪器验证，LIMMT已验证高质量子集有效。CLAIMS明确指出人工变量库缺少客观校准与全面覆盖；LIMMT规则和选择比例仍需更多语料检验。执行失败同时受参考、场景和控制器影响，这是待验证的归因问题，不能将失败一概解释为错误数据。\citep{claims,limmt}
\end{enumerate}
\subsection{固定基线与测量边界}
使用发布版CLoSD的生成模型与跟踪策略，在已有Isaac Gym环境中运行三个随机种子（11、12、13），每个64环境，全部保留第一次尝试和失败前缀，不更新模型、策略或奖励。记录策略当前观测中应用动作前的生成参考，以及同一步物理执行后的状态。初始30帧仅从质量统计排除，终止统计保留全部帧。\citep{closd}
首轮v2错误读取了继承奖励函数使用的独立数据缓存，其跟踪分析已作废并保留记录。v3改用实际观测参考，通过独立调用顺序、坐标、关节顺序和时间变化检查。这是测量修正，不是控制算法改进。
本轮为SMPL物理角色的文本到动作诊断，不是目标人形机器人执行器认证，不含显式空间目标，也没有验证下游再训练收益。足部刚体中心速度只能作接触代理，旋转可能使其移动，不能直接当作表面滑移真值。
'''
if analysis.exists():
    tex+=r'''\subsection{实际结果与假设判定}
\begin{center}\begin{tabular}{lrrrr}\hline
种子 & 首次尝试 & 未提前终止 & 误差门控通过 & 联合代理门控通过\\\hline
'''
    for s,v in list(a['per_seed'].items())+[('合计',a['pooled'])]:
        tex+=f'{s} & {v["n"]} & {v["completed_without_termination"]} & {v["tracking_gate_accepted"]} & {v["quality_proxy_gate_accepted"]}'+r'\\'+'\n'
    tex+=r'''\hline\end{tabular}\end{center}
预设误差门槛为平均g-MPJPE不超过100毫米、帧误差p95不超过200毫米；联合门控增加持续净接触力超过10牛顿时，足部中心速度超过0.5米/秒的比例不超过5\%。阈值为本次诊断选择，不是通用合格标准。
\begin{itemize}
\item \textbf{H1：有限支持。} 174条未提前终止的片段中15条未通过误差门控。没有提前终止不能替代跟踪质量检查，但尚不能将误差归因于生成器。
\item \textbf{H2：跟踪恶化假设未得到支持。} 参考边界速度比内部高约2.202米/秒，实际运动差约0.058米/秒，边界跟踪误差反而平均低34.64毫米。已执行状态参与后续规划与坐标对齐；参考更新不等同于身体运动不连续。
\item \textbf{接触代理尚不能认证物理缺陷。} 27组阈值敏感性设置下联合通过63--109条，主设置为85条。应结合表面接触几何复核，不能把门控拒绝的样本全部解释为脚滑。
\end{itemize}
原地跳高样本（种子12/环境3）的平均误差120.9毫米、p95为341.2毫米；舞蹈样本（11/61）分别为123.6与342.5毫米。含台阶和扶栏的提示缺少对应场景支撑，应与生成缺陷分开。本轮未支持三项缺口全部成立，也未证明合成数据提高模仿学习。
'''
tex+=r'''\subsection{下一步的证据要求}
先在相同输入和条件上比较普通生成与通用物理精化，检查关键事件、跟踪误差和接触几何；排除缺少场景支撑的请求。只有确认具体失败模式后，才确定新的生成或修正方法。进一步在同一片段上使用第二个固定跟踪器分离控制器影响，最后以等数据量、等训练预算、多种子和独立测试条件验证训练价值。当前结论限定于仿真。
'''
(lit/'gap_comparison_and_pilot_20261011.tex').write_text(tex,encoding='utf-8')
(out/'gap_comparison_and_pilot_20261011.tex').write_text(tex,encoding='utf-8')
latex_report=lit/'literature_review_report.tex'
latex_backup=lit/'literature_review_report_before_20261011.tex'
if not latex_backup.exists():latex_backup.write_bytes(latex_report.read_bytes())
report_source=latex_report.read_text(encoding='utf-8')
input_line='\\input{gap_comparison_and_pilot_20261011}'
if input_line not in report_source:
    report_source=report_source.replace('\\section{记录与一致性}',input_line+'\n\n\\section{记录与一致性}',1)
    assert input_line in report_source
    latex_report.write_text(report_source,encoding='utf-8')
# Append the same section to the existing consolidated report. Preserve its
# previous version on the first edit; reruns replace only our own section.
original=lit/'research_gap_review.html'
backup=lit/'research_gap_review_before_20261011.html'
if not backup.exists(): backup.write_bytes(original.read_bytes())
t=original.read_text(encoding='utf-8')
pattern=r'<section id="gap-validation-20261011">.*?</section>'
if re.search(pattern,t,re.S): t=re.sub(pattern,lambda _:sec,t,flags=re.S)
else: t=t.replace('</main>',sec+'</main>',1)
if '#gap-validation-20261011' not in t:
    t=t.replace('</nav>','<a href="#gap-validation-20261011">10月11日缺口与实验诊断</a></nav>',1)
original.write_text(t,encoding='utf-8')
manifest=dict(date='2026-10-11',papers=20,bib_keys_reused=20,new_bib_entries=0,
    original_html_backup=str(backup),experiment_complete=analysis.exists(),
    output=str(original),candidate_gaps=3,
    facts='No premature manuscript novelty claim; experiment measures hypotheses only')
(lit/'gap_update_manifest_20261011.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
