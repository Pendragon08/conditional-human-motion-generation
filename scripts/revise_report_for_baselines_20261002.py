"""Create the teacher-facing baseline-only edition; preserve the original full archive."""
from pathlib import Path
import runpy,shutil
ROOT=Path(r'D:\具身智能\conditional-human-motion-generation')
OUT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
old=OUT/'实验结果汇报.html';backup=OUT/'实验结果汇报_完整档案备份.html'
if old.exists() and not backup.exists():shutil.copy2(old,backup)
s=(ROOT/'scripts/build_all_results_report_20261002.py').read_text(encoding='utf-8')
def replace_block(start,end,body):
    global s
    a=s.index(start);b=s.index(end,a);s=s[:a]+body+'\n'+s[b:]
s=s.replace("PACK=OUT/'汇报资料_20261002'","PACK=OUT/'论文复现汇报资料_20261002'")
s=s.replace("        if not p.is_file():continue\n        rel=p.relative_to(ROOT).as_posix()",'''        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if sub=='media':
            permitted = p.name.startswith('r3_official_') or any('/'+name+'/' in '/'+rel for name in ['sim2sim_mujoco','sim2sim_completed_20261002','mujoco_robust_contact','original_human_generation_20261002'])
        elif sub=='reports':
            permitted = p.name.startswith(('03_dreamcontrol_multitask','10_','11_','12_','13_','14_'))
        else:
            permitted = p.name.startswith(('r3_dreamcontrol_official_','r3_dreamcontrol_Sit_eval_failure','r3_dreamcontrol_Bimanual-Pick_eval_failure','r3_closd_','r3_dreamcontrol_remaining_manifest','official_other_tasks_assets_manifest','dreamcontrol_paper_task_matrix')) or any('/'+name+'/' in '/'+rel for name in ['configs','sim2sim_mujoco','sim2sim_completed_20261002','mujoco_robust_contact'])
            permitted = permitted and 'i1' not in rel.lower() and 'smoke' not in rel.lower()
        if not permitted:continue''')
s=s.replace("kick=read('dreamcontrol_r3_completed.json')['experiments'][0]",'''kick=read('dreamcontrol_r3_completed.json')['experiments'][0]
(PACK/'results/DreamControl_Kick_official_evidence.json').write_text(json.dumps(kick,ensure_ascii=False,indent=2),encoding='utf-8')
humans=read('../media/original_human_generation_20261002/manifest.json')
assert humans['status']=='COMPLETE' and len(humans['videos'])==6
for v in humans['videos']:assert sha(PACK/'media/original_human_generation_20261002'/v['file'])==v['sha256']''')
s=s.replace('论文基线复现、条件生成对照与跨仿真执行验证','人体动作生成方法、论文控制基线与跨仿真验证')
s=s.replace('<strong>8次</strong><span>I1匹配预算交叉评估已完成</span>','<strong>6段</strong><span>原始扩散生成人体动作展示</span>')
replace_block('overview=note(',"basebody=",'''overview=note('<b>本次汇报重点：</b>先说明人体动作怎样由文本与空间条件生成，再观看原始人体视频，最后查看机器人控制基线与跨仿真结果。六项官方控制任务已有结果；跨仿真测试仍存在稳定性和接触失败。','conclusion')
overview+=table(['环节','当前已有内容','应怎样理解'],[
 ['人体动作生成','OmniControl官方预训练模型的两批Kick输出；固定前3条，共6段','人体22关节原始动画；不是机器人执行，不是仿真物理校验'],
 ['DreamControl控制基线','6项训练＋官方评估＋视频','使用官方原参考库训练；人体展示视频不等于这些策略的全部训练输入'],
 ['CLoSD闭环方法','两组官方权重4096环境评估','扩散规划＋物理跟踪闭环；本地3072环境正式微调尚未完成'],
 ['跨仿真执行与接触','六项MuJoCo测试；踢/拳2324次压力测试','限定协议下的执行结果，不能直接套用原Isaac成功率']])
overview+='<p>所有展示视频已保存本地，可离线观看。观察人体动作质量与评估机器人任务成功率是两个不同环节。</p>'
methods='<h3>1. DreamControl：先生成参考动作，再训练机器人跟踪</h3><div class="method-flow"><span>文本＋关节目标位置/时刻</span><b>→</b><span>OmniControl条件扩散</span><b>→</b><span>人体关节轨迹</span><b>→</b><span>原始G1重定向与修整</span><b>→</b><span>Isaac Lab PPO跟踪</span></div>'
methods+='<p>模型从噪声逐步去噪，生成一段人体运动。文本说明“做什么”，空间提示说明“哪一个关节在什么时刻靠近哪里”。预训练模型从人体运动数据学习了动作分布，这就是人体动作先验；本次使用官方OmniControl权重，没有从零训练人体扩散模型。空间引导帮助动作满足给定位置，自然性引导帮助保持动作外观。生成后再转换成G1参考，PPO学习物理执行。该路线先离线生成参考，与控制训练分阶段。</p>'
methods+='<p class="source">方法依据：<a href="https://github.com/GenRobo/DreamControl/blob/main/TrajGen/README.md">DreamControl官方生成代码说明</a> · <a href="https://github.com/neu-vi/omnicontrol">OmniControl官方代码与HumanML3D预训练模型</a> · <a href="https://genrobo.github.io/DreamControl/">DreamControl原论文项目页</a></p>'
methods+='<h3>2. CLoSD：短段生成与物理执行不断交替</h3><div class="method-flow"><span>文本＋目标位置＋实际历史动作</span><b>→</b><span>DiP扩散规划下一段</span><b>→</b><span>RL物理跟踪执行</span><b>↻</b><span>实际执行帧反馈给DiP</span></div><p>CLoSD的DiP是自回归扩散规划器：它根据文本、目标和已经执行的动作，生成下一小段计划；控制器逐帧跟踪，再把实际运动作为下一轮历史输入。因此它是在执行中持续重规划的闭环，不是一次生成完整动作后永远按原计划播放。当前结果来自官方已发布控制器/规划器；本地新3072环境训练尚未验收完成。</p><p class="source">方法依据：<a href="https://guytevet.github.io/CLoSD-page/">CLoSD原论文方法与规划—执行闭环</a> · <a href="https://github.com/GuyTevet/CLoSD">官方代码</a></p>'
methods+=table(['术语','容易理解的解释'],[['条件','文字要求，以及某个关节在某个时刻应到的位置'],['扩散模型','从噪声逐步生成动作的模型'],['人体动作先验','从已有的人体运动数据中学到的动作规律'],['重定向','把人体姿态换成机器人能使用的关节参考'],['PPO跟踪','让控制器通过仿真试错，学习跟随参考动作'],['闭环','用真正执行出来的动作更新下一段计划']])
generation='<p>下面六段都来自已完成的官方方法推理输出：同一句右腿踢文本，采样种子10/42，各100条；每批固定展示编号0、1、2，没有看过结果后挑选好看的动作。两批空间提示也不同，因此不能把所有差异只归因于随机种子。</p>'
generation+=note('<b>这是人体生成结果，尚未变成机器人。</b>按原始20fps、196帧/9.8秒播放，蓝色为身体，橙色为右腿与足迹，红色叉号为当前时刻的空间目标，绿色为左脚轨迹。左右分别是斜视和侧视。只交换显示坐标轴，没有平滑、额外IK或人为移动身体；地面保持原y=0。采用22关节骨架而非皮肤网格，不能凭骨架证明碰撞或真实接触质量。')
generation+=table(['老师可观察的问题','怎样看视频'],[['是否符合文本','是否明显出现右腿踢出并回收，是否混入多余动作'],['目标是否接近','红色提示出现时，右脚是否接近目标；不把看起来接近当作量化通过'],['动作是否自然','看躯干、双臂与支撑腿是否协调，是否出现突然跳变或抖动'],['脚是否滑动/漂浮','看左脚绿色轨迹和y=0地面；这是视觉观察，不是力学接触测量'],['展示覆盖范围','当前保存的人体视频只有Kick一种文本；其余任务展示的是机器人执行视频']])
generation+='<div class="video-grid">'
for v in humans['videos']:
    prefix=PACK.name+'/media/original_human_generation_20261002/'
    generation+=f'<article class="video-card human-preview"><h3>原始人体生成｜seed {v["seed"]}｜编号{v["sample_id"]:03d}</h3><video controls preload="none" poster="{url(prefix+Path(v["file"]).with_suffix(".jpg").name)}"><source src="{url(prefix+v["file"])}" type="video/mp4"></video><p>文本：一个人站立，并用右腿踢。原始9.8秒、20fps；不是Isaac或MuJoCo执行视频。</p></article>'
generation+='</div><p>本节未声称其余五类任务的人体扩散生成已经重新复现。CLoSD两组正式评估没有保存生成/执行结果视频，本次也未用其他示例冒充。</p><p class="source">'+link('media/original_human_generation_20261002/manifest.json','原始轨迹来源、固定样本编号与视频哈希')+'</p>'
''')
# Only the teacher-visible evidence link changes; the source result is read normally.
s=s.replace("basebody+=source('dreamcontrol_r3_completed.json'","basebody+=source('DreamControl_Kick_official_evidence.json'")
replace_block('failurebody=table(', '# All copied videos', '''failurebody=table(['记录','现象','处理及边界'],[
 ['Sit/Bimanual视频初始化','默认路径native malloc异常','保留模型/失败日志，官方USD显示读写适配后正常评估；底层根因未证实'],
 ['CLoSD旧512微调','保存66000后退出139、部分任务退化','同口径前后评估已保存，不视为论文3072环境完整训练'],
 ['直播曾影响训练','耦合WebRTC/GLFW停滞','现已分离训练与实际状态监控，显示故障不阻塞PPO'],
 ['MuJoCo接触/动作失败','新增任务没有可靠迁移通过','保留场景/驱动差异与实际负结果，不把站稳当成抓取成功'],
 ['摩擦测试首版','实际接触摩擦与设定不符','V2完整重跑并逐接触验证，汇报使用最终V2结果']])
failurebody+=note('当前DreamControl只完成6/11控制任务，不等于整篇论文完成；CLoSD官方权重评估完成不等于本地重新训练完成。人体生成骨架、机器人参考、Isaac执行、MuJoCo执行是不同层次，报告分别标注。没有实机验证。')
failurebody+='<p>后续先继续既有论文复现队列、补齐CLoSD正式训练；对跨仿真失败核对任务场景、惯量/碰撞与驱动映射，再按独立协议验证，不覆盖本轮负结果。</p>'
''')
replace_block("for x in read('dreamcontrol_r3_completed.json')['experiments'][1:]", 'for folder,group,label', '')
s=s.replace("[('all','全部'),('baseline','论文控制基线'),('i1','I1同预算对照'),('early','早期增强'),('sim','MuJoCo迁移'),('robust','扰动与实体接触')]","[('all','全部机器人执行'),('baseline','论文控制基线'),('sim','MuJoCo迁移'),('robust','扰动与实体接触')]")
s=s.replace("appendix='<p>以下明细报告保留原时间线；历史“尚未完成”描述以本汇报最新结果为准。未纳入源Git的探索性文件单独标记，不能自动当作已验收实验。可搜索文件名/主题。</p>","appendix='<p>本节只收录论文复现和跨仿真证据。历史进度以本汇报最新结构化结果为准，可搜索文件名/主题。</p>")
s=s.replace("for p in sorted((ROOT/'reports').glob('*.md')):\n    rel=", "for p in sorted((ROOT/'reports').glob('*.md')):\n    if not p.name.startswith(('10_','11_','13_','14_')):continue\n    rel=")
s=s.replace('全部原始结果文件索引（包括诊断、见证与失败记录）','本次论文复现结果文件索引（包括环境见证与失败记录）')
s=s.replace('文件索引覆盖本地已保存结果，不把文件数量当成实验完成数。未纳入源Git的探索性文件只供诊断查阅。','文件索引仅覆盖本次汇报的复现资料，不把文件数量当成实验完成数。')
s=s.replace('所有展示视频已保存本地，可离线观看。','所有展示视频已保存本地，可离线观看。')
replace_block('nav=[','js=', '''nav=[('overview','01 汇报概况'),('methods','02 人体运动怎样生成'),('generation','03 先观看人体生成视频'),('baseline','04 DreamControl控制基线'),('closd','05 CLoSD复现'),('sim','06 MuJoCo跨仿真'),('robust','07 扰动与实体接触'),('failures','08 运行问题与后续'),('videos','09 机器人执行视频'),('appendix','10 复现明细与证据')]
body=hero+section('overview','论文复现汇报概况',overview,'先看结论')+section('methods','人体运动生成方法：从条件到执行',methods,'方法说明')+section('generation','原始人体生成：固定六条样本',generation,'供老师观察动作质量')+section('baseline','DreamControl：六项官方控制基线',basebody)+section('closd','CLoSD：权重评估与本地微调',clbody)+section('sim','Isaac Lab → MuJoCo：六项冻结策略迁移',simbody)+section('robust','Kick/Punch：扰动鲁棒性与实体打击',robbody)+section('failures','运行问题、证据边界与后续',failurebody)+section('videos','机器人执行视频',video_body,f'{len(categories)} 段执行视频')+section('appendix','论文复现明细与来源',appendix,'可复查证据')
css+='.method-flow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#f0f6f8;padding:18px;border-radius:12px}.method-flow span{padding:12px;background:white;border:1px solid #cadde6;border-radius:8px;font-size:14px}.human-preview{border:2px solid #9acac3}.human-preview h3{color:#146d66}'
''')
s=s.replace('通用条件人体运动生成｜全部已完成实验汇报','通用条件人体运动生成｜论文复现与人体生成展示')
s=s.replace('负结果与运行适配保留','人体生成与控制执行分别展示')
s=s.replace('论文/创新验收','论文验收')
s=s.replace("'matched_i1_eval_count':8,","'human_motion_videos':6,")
s=s.replace("'video_count':len(categories)","'video_count':len(categories)+6")
s=s.replace("'science_gate_passed':False,","'report_scope':'baseline reproduction and original human generation; research comparison experiments excluded',")
s=s.replace('源提交','源提交')
s=s.replace('原有归档不删除、不覆盖。','原完整档案版另存备份；本次正文仅汇报论文复现。')
s=s.replace('资料在“汇报资料_20261002”内','资料在“论文复现汇报资料_20261002”内')
s=s.replace('离线视频和原始证据在“汇报资料_20261002”内','离线视频和原始证据在“论文复现汇报资料_20261002”内')
s=s.replace("document.querySelectorAll('.video-card').forEach","document.querySelectorAll('.video-card[data-group]').forEach")
s=s.replace('没有使用原先的 98 条增强组或 I1 候选策略。','')
s=s.replace('没有替换成中途恢复策略或创新策略。','')
s=s.replace("REPORT.write_text(page,encoding='utf-8')", "page=page.replace('没有使用原来的 98 条增强组或 I1 候选策略。','').replace('没有替换成中途恢复策略或创新策略。','')\nREPORT.write_text(page,encoding='utf-8')")
new=ROOT/'scripts/build_baseline_report_20261002.py'
new.write_text(s,encoding='utf-8');runpy.run_path(str(new),run_name='__main__')
