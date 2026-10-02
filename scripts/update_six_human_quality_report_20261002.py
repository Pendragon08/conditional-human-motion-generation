"""Insert reproducible six-task human screening and paper filtering evidence."""
from pathlib import Path
from bs4 import BeautifulSoup
import datetime, hashlib, json, shutil, urllib.request

REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
RESULT=REPO/'results/human_generation_quality_20261002'
LOCAL=ROOT/'人体生成视频/六任务官方质量筛查'
PACK=ROOT/'论文复现汇报资料_20261002'
PROOF=PACK/'生成与验收'
REPORT=ROOT/'实验结果汇报.html'
PDF_URL='https://genrobo.github.io/DreamControl/static/paper/DreamControl_arxiv.pdf'
LOCAL.mkdir(parents=True,exist_ok=True)
PROOF.mkdir(parents=True,exist_ok=True)
for file in RESULT.glob('*'):
    if file.is_file():shutil.copy2(file,LOCAL/file.name)
for name in ['audit_six_official_humans_20261002.py','update_six_human_quality_report_20261002.py','pin_six_human_source_identity_20261002.py']:
    shutil.copy2(REPO/'scripts'/name,LOCAL/name)
paper=LOCAL/'DreamControl_arxiv.pdf'
if not paper.exists():
    with urllib.request.urlopen(PDF_URL,timeout=40) as response:paper.write_bytes(response.read())
assert paper.read_bytes().startswith(b'%PDF-')
facts={
    'url':PDF_URL,'local_pdf_sha256':hashlib.sha256(paper.read_bytes()).hexdigest(),
    'references':['III-A, printed page 4','IX-B, printed pages 13-14','Table XI, printed page 14'],
    'filtering_stage':'G1 post-retargeting, before RL tracking',
    'filter_conditions':['torso x-axis angle relative to world x-axis exceeds task threshold','pelvis height below task threshold','body collision with scene'],
    'table_XI':[{'task':'Pick','before':100,'after':67,'torso_threshold_degrees':45,'pelvis_height_threshold_m':0.6},
                {'task':'Precise Punch','before':100,'after':100,'torso_threshold_degrees':45,'pelvis_height_threshold_m':0.6},
                {'task':'Precise Kick','before':100,'after':66,'torso_threshold_degrees':90,'pelvis_height_threshold_m':0.5},
                {'task':'Press Button','before':100,'after':96,'torso_threshold_degrees':60,'pelvis_height_threshold_m':0.5}],
    'other_tasks':'One trajectory manually selected except Open Drawer/Open Door; not an automatic all-sample physics acceptance rate.',
    'not_claimed':'Paper describes heuristic selection, not full per-trajectory torque/contact-force/friction/stability certification.',
    'reported_paper_counts_not_recomputed':True,
    'new_human_proxy_audit_is_not_the_paper_filter':True
}
(RESULT/'paper_filtering_evidence.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(RESULT/'paper_filtering_evidence.json',LOCAL/'paper_filtering_evidence.json')
summary=json.loads((RESULT/'summary.json').read_text(encoding='utf-8'))
labels={'Kick':'Kick｜踢击','Punch':'Punch｜出拳','Jump':'Jump｜跳跃','Sit':'Sit｜坐下','Pick':'Pick｜单手取物','Bimanual_Pick':'Bimanual-Pick｜双手取物'}
notes={'Kick':'踢腿的快速位移不单独判为失败','Punch':'快速出拳与滑脚提示分别统计','Jump':'13/13出现腾空；腾空属于任务预期','Sit':'正常下蹲、坐下不按骨盆低判为失败','Pick':'未在人体骨架上验证真实抓握或抬物','Bimanual_Pick':'未验证双手—工具箱接触力'}
rows=[];motion_rows=[];sensitivity=[];ground_rows=[]
for t in summary['tasks']:
    c=t['counts_overlapping'];n=t['count'];task=labels[t['task']]
    rows.append(f"<tr><td><b>{task}</b></td><td>{n}</td><td>{c['foot_below_nominal_ground']}/{n}</td><td>{c['near_ground_slide_proxy']}/{n}</td><td>{c['bone_length_variation_proxy']}/{n}</td><td>{notes[t['task']]}</td></tr>")
    motion_rows.append(f"<tr><td>{task}</td><td>{c['large_joint_step_descriptive']}/{n}</td><td>{t['max_joint_step_m']:.3f} m</td><td>{c['airborne_descriptive']}/{n}</td></tr>")
    sensitivity.append(f"<tr><td>{task}</td><td>{c['near_ground_slide_proxy']}/{n} → {c['slide04_warning']}/{n}</td><td>{c['bone_length_variation_proxy']}/{n} → {c['bone20_warning']}/{n}</td></tr>")
    ground_rows.append(f"<tr><td>{task}</td><td>{c['any_foot_y_below_zero']}/{n}</td><td>{t['min_foot_y_m']*1000:.2f} mm</td><td>{c['foot_below_nominal_ground']}/{n}</td></tr>")
section=f'''
<div id="six-task-human-quality-screening">
<h3>六类原始人体动作的质量筛查</h3>
<p><b>覆盖全部342条官方原始人体样本，非仅18段展示视频。</b>每条196帧、20fps、名义视频时长9.8秒；未混入此前seed10/42自行生成的Kick批次。六个来源文件逐一核对SHA256，检查前后原始数据不变。342条均为有限数值、长度有效，结构无效样本为0条。</p>
<div class="note"><b>结论：可报告几何提示数量，尚不能给出“物理不合理动作”的确证数量或物理通过率。</b>下表为重定向前人体骨架的离线几何与时序代理筛查，各列可重叠；提示不是淘汰数，也不是动力学失败率。未删除或修正任何样本，未改变官方PPO参考库。未触发−3厘米持续越界阈值，不等于脚点从未低于y=0，更不等于接触、稳定性、碰撞或力矩约束已满足。</div>
<div class="tablewrap"><table><thead><tr><th>任务</th><th>样本数</th><th>同脚低于−3 cm<br/>连续≥3帧</th><th>近地滑动提示</th><th>骨段变化提示</th><th>任务解释</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h4>统一筛查标准与适用范围</h4>
<div class="tablewrap"><table><thead><tr><th>检查项</th><th>计算与阈值</th><th>判读边界</th></tr></thead><tbody>
<tr><td>数据结构</td><td>22关节×3坐标；长度在有效帧范围内；所有坐标须为有限数值。</td><td>仅验证数据可读取、可计算。</td></tr>
<tr><td>脚点深度与持续时间提示</td><td>原始HumanML坐标y向上、y=0为名义地面；左/右脚点10/11。任一<strong>同一脚点</strong> y &lt; −0.03 m，持续≥3个采样帧。</td><td>3帧首尾跨度0.10秒。脚关节不是足底碰撞网格；此项只提示超出声明容差的骨架越界，0次提示不等于没有微小负值。</td></tr>
<tr><td>近地滑动代理</td><td>相邻两帧同一脚点均在[−0.03, 0.05] m；水平速度‖Δ(x,z)‖/0.05 s &gt; 0.20 m/s，连续≥3个时间间隔（0.15秒）。</td><td>近地不等于真实支撑接触；不能据此确认足底滑动或摩擦失效。左右脚不交替拼接成一次持续事件。</td></tr>
<tr><td>骨段长度变化代理</td><td>21条相邻关节骨段分别计算长度；相对<strong>该样本、该骨段自身的时间中位数</strong>偏差 &gt; 10%，同一骨段持续≥3帧（首尾0.10秒）。</td><td>没有外部人体骨长真值。此项是自身一致性提示；CSV同时给出中位骨长与绝对偏差（米），不能解释为解剖或动力学认证。</td></tr>
<tr><td>快速位移描述</td><td>任一关节相邻帧位移 &gt; 0.15 m（相当于3 m/s）。</td><td>只作时序变化描述；踢击、出拳、跳跃可以快速运动，未据此判“质量差”。</td></tr>
<tr><td>腾空描述</td><td>两脚点均高于0.10 m，持续≥6帧（首尾0.25秒）。</td><td>只描述运动模式；Jump的腾空不作失败，下蹲/坐下也不套用站立任务的骨盆高度下限。</td></tr>
</tbody></table></div>
<p><b>阈值来源：</b>以上是本报告预先固定的探索性代理标准，不是论文表XI的官方过滤阈值，也没有通过人工质量标注校准。骨架没有质量、惯量、驱动力矩、足底形状、场景物体与接触力；本阶段未做逆动力学、平衡/摩擦锥检验、自碰撞网格检测或文本语义人工评分。完整物理可执行性需结合后续机器人动力学与接触测试，不能由这些几何计数代替。</p>
<details><summary>快速运动、腾空与阈值敏感性</summary>
<p>原始坐标中有{sum(t['counts_overlapping']['any_foot_y_below_zero'] for t in summary['tasks'])}/342条至少出现一次脚点y&lt;0，但全部没有触发−3 cm持续3帧阈值。下面保留零高度越界和最小原始值；未重新对齐地面或抬高动作。</p>
<div class="tablewrap"><table><thead><tr><th>任务</th><th>任一帧脚点y&lt;0</th><th>全任务最小脚点高度</th><th>同脚低于−3 cm连续≥3帧</th></tr></thead><tbody>{''.join(ground_rows)}</tbody></table></div>
<div class="tablewrap"><table><thead><tr><th>任务</th><th>快速位移样本</th><th>最大关节单帧位移</th><th>腾空样本</th></tr></thead><tbody>{''.join(motion_rows)}</tbody></table></div>
<p>为展示阈值依赖，另给出更宽松阈值的计数；两套结果全部保留，不挑选较好数值作为成绩。</p>
<div class="tablewrap"><table><thead><tr><th>任务</th><th>近地速度阈值0.2→0.4 m/s</th><th>骨段偏差阈值10%→20%</th></tr></thead><tbody>{''.join(sensitivity)}</tbody></table></div>
<p>例如Kick骨段提示100/100、Punch近地滑动提示100/100，都不能解读为全部动作物理失败。这些提示针对原始人体表示，与后续G1参考筛选或PPO任务成功率属于不同指标。</p></details>
<p class="source">本地证据：<a href="人体生成视频/六任务官方质量筛查/summary.json">六任务统计</a> · <a href="人体生成视频/六任务官方质量筛查/per_sample.csv">全部342条逐样本CSV</a> · <a href="人体生成视频/六任务官方质量筛查/per_sample.json">完整原始指标JSON</a> · <a href="人体生成视频/六任务官方质量筛查/thresholds.json">阈值与计算定义</a> · <a href="人体生成视频/六任务官方质量筛查/audit_six_official_humans_20261002.py">可复算筛查代码</a> · <a href="人体生成视频/六任务官方质量筛查/source_identity.json">官方提交与六个Git LFS内容标识</a></p>
</div>
<div id="dreamcontrol-paper-physical-filtering">
<h3>DreamControl原文：生成后的物理相关筛选</h3>
<p><b>有，筛选对象是训练参考轨迹；位置在G1重定向之后、RL训练之前。</b>正文§III-A将足部接触、自碰撞及足部朝向代价加入重定向；附录§IX-B按躯干方向、骨盆高度及身体—场景碰撞作任务相关启发式筛选。它不是对原始人体样本逐条完成力矩、接触力与稳定性认证，也不是评估rollout或人类自然度样本的筛选协议。<a href="人体生成视频/六任务官方质量筛查/DreamControl_arxiv.pdf#page=14">原论文第14页，表XI与§IX-B</a></p>
<div class="tablewrap"><table><thead><tr><th>论文任务</th><th>筛前→筛后</th><th>躯干方向夹角上限</th><th>骨盆高度下限</th></tr></thead><tbody>
<tr><td>Precise Kick</td><td>100→66</td><td>90°</td><td>0.5 m</td></tr>
<tr><td>Precise Punch</td><td>100→100</td><td>45°</td><td>0.6 m</td></tr>
<tr><td>Pick</td><td>100→67</td><td>45°</td><td>0.6 m</td></tr>
<tr><td>Press Button（补充原文口径）</td><td>100→96</td><td>60°</td><td>0.5 m</td></tr>
</tbody></table></div>
<p>躯干角度指躯干x轴相对世界x轴的方向夹角，非仅俯仰角。表中数值是<strong>论文报告</strong>，不是本轮重跑过滤所得。Kick减少34条，表示未通过这些启发式规则，不表示34条均经完整动力学判定失败。</p>
<p>Jump、Sit、Bimanual-Pick按附录“其他任务”口径人工选1条供训练，<strong>人工挑选准则未披露</strong>；本地发布人体数组分别13、20、9条，其余样本不能统称自动物理过滤淘汰。Open Drawer/Open Door另走右臂优化构造路径。论文筛选与上方人体骨架代理筛查分属不同对象、阶段和标准。</p>
<p class="source">来源：<a href="{PDF_URL}">DreamControl §III-A、§IX-B、表XI</a> · <a href="人体生成视频/六任务官方质量筛查/paper_filtering_evidence.json">原文位置、表值与本地PDF校验记录</a></p>
</div>'''
backup=ROOT/'实验结果汇报_六任务质量筛查补充前备份.html'
if not backup.exists():shutil.copy2(REPORT,backup)
soup=BeautifulSoup(REPORT.read_text(encoding='utf-8'),'html.parser')
for id in ['six-task-human-quality-screening','dreamcontrol-paper-physical-filtering']:
    old=soup.find(id=id)
    if old:old.decompose()
generation=soup.find(id='generation')
anchor=next(p for p in generation.find_all('p',recursive=False) if '共342条' in p.get_text())
fragment=BeautifulSoup(section,'html.parser')
for block in reversed(fragment.find_all('div',recursive=False)):anchor.insert_after(block)
audit_file=RESULT/'EXPERIMENT_AUDIT.json'
audit=json.loads(audit_file.read_text(encoding='utf-8')) if audit_file.exists() else None
if audit:
    soup.find(id='six-task-human-quality-screening').append(BeautifulSoup('<p class="source"><a href="人体生成视频/六任务官方质量筛查/EXPERIMENT_AUDIT.md">指标、来源与表述的复核记录</a>：独立复算与声明范围已核验；该复核不属于人体动力学认证。</p>','html.parser').p)
reference=soup.find(id='dreamcontrol-input-reference')
if not reference.find(id='paper-filter-link'):
    p=soup.new_tag('p',id='paper-filter-link')
    p.append(BeautifulSoup('<b>生成后筛选：</b>原文在G1重定向后进行物理相关启发式过滤；Kick的100→66明确见表XI。<a href="#dreamcontrol-paper-physical-filtering">筛选位置、标准与六类任务的选择方式</a>，以及<a href="#six-task-human-quality-screening">本报告对全部342条人体样本的代理筛查</a>。','html.parser'))
    reference.append(p)
for row in soup.find(id='artifacts').find_all('tr'):
    cells=row.find_all('td',recursive=False)
    if cells and cells[0].get_text().strip().startswith('2 人体可视化'):
        cells[1].clear();cells[1].append('人体数组 → 骨架视频、逐条CSV/JSON、质量代理统计；来源分开归档。')
        cells[2].clear();cells[2].append('六类官方人体342条：全部离线筛查，固定展示18段视频。此前自行生成两批Kick各100条：另存200段视频与历史筛查。原始坐标不平滑、不修正；阈值提示不作为物理通过标签。')
        if not cells[3].find('a',href='人体生成视频/六任务官方质量筛查/per_sample.csv'):
            cells[3].append(BeautifulSoup(' · <a href="人体生成视频/六任务官方质量筛查/per_sample.csv">六类342条筛查明细</a>','html.parser'))
REPORT.write_text(str(soup),encoding='utf-8')
md='# 六任务人体生成质量筛查与DreamControl原文筛选\n\n'
md+='检查范围：342条作者发布的原始22关节人体轨迹；不是重新生成、不是G1筛选重跑。18段视频仅为固定样本展示。\n\n'
md+='|任务|样本|同脚y<−0.03m连续≥3帧|近地滑动代理|骨段变化代理|\n|---|---:|---:|---:|---:|\n'
for t in summary['tasks']:
    c=t['counts_overlapping'];md+=f"|{t['task']}|{t['count']}|{c['foot_below_nominal_ground']}|{c['near_ground_slide_proxy']}|{c['bone_length_variation_proxy']}|\n"
md+='\n阈值均为探索性：脚点y<−0.03m同脚≥3帧；近地[−0.03,0.05]m同脚水平速度>0.2m/s连续3间隔；骨段相对自身中位长度偏差>10%同骨≥3帧。三帧首尾0.10秒，三间隔0.15秒。骨段为自身一致性代理，没有外部骨长真值；完整原始指标与绝对偏差在per_sample.json/CSV。\n\n'
md+=f"原始数据有{sum(t['counts_overlapping']['any_foot_y_below_zero'] for t in summary['tasks'])}/342条至少一次脚点y<0；没有任何动作触发y<−0.03m连续3帧判据。不能将阈值计数0解读为完全没有越过名义地面。\n\n"
md+='各计数可重叠，不是物理失败率；未删除动作。快速位移和腾空只作描述。342条结构有效；物理失败数量、动力学通过率和语义通过率均未确定，没有接触力/力矩/稳定性验证。阈值敏感性也在summary.json。\n\n'
md+='论文§III-A与§IX-B/表XI说明有物理相关启发式筛选，位于G1重定向后、RL之前。过滤躯干方向夹角、骨盆高度、场景碰撞。Kick100→66、90°/0.5m；Punch100→100、45°/0.6m；Pick100→67、45°/0.6m；PressButton100→96、60°/0.5m。Jump/Sit/Bimanual按其他任务口径人工选1条。不能称论文完全未筛查，也不能称其完整认证每条动作的动力学可行性。这些论文数字不是本次重新过滤结果。\n\n'
md+=f'来源：[DreamControl]({PDF_URL})，第4、13–14页；本地PDF SHA256 `{facts["local_pdf_sha256"]}`。\n'
identity=json.loads((RESULT/'source_identity.json').read_text(encoding='utf-8'))
md+=f"\n源身份：官方上游提交 `{identity['official_main_commit_observed']}`；六个Git LFS OID和固定提交URL见source_identity.json，均与本地原始数组SHA一致。人工选择准则未披露；不推断未选样本均为物理失败。\n"
if audit:
    md+='\n完整性复核：修正阈值措辞后PASS，见EXPERIMENT_AUDIT.md/JSON。同模型家族语义审查为暂定；确定性复算11,628字段一致。不是人体动力学或物理质量验收。\n'
(RESULT/'README.md').write_text(md,encoding='utf-8');shutil.copy2(RESULT/'README.md',LOCAL/'README.md')
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),
         'sequences':342,'task_count':6,'data_unchanged_sha_verified':True,'audit_status':audit['verdict'] if audit else 'fresh_reviewer_pending',
         'generation_videos':len(soup.find(id='generation').find_all('video')),'robot_videos':len(soup.find(id='videos').find_all('video')),
         'supersedes_report_hash_receipt':'Kick单批与历史累计数量澄清.json'}
(PROOF/'六任务人体质量筛查与论文标准核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
