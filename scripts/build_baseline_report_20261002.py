"""Build a portable presentation report from archived evidence, without running experiments."""
from pathlib import Path
import json, html, shutil, hashlib, subprocess, re, datetime, statistics
from urllib.parse import quote

ROOT=Path(r'D:\具身智能\conditional-human-motion-generation')
OUT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
PACK=OUT/'论文复现汇报资料_20261002'
REPORT=OUT/'实验结果汇报.html'
def read(p):return json.loads((ROOT/'results'/p).read_text(encoding='utf-8-sig'))
def esc(x):return html.escape(str(x))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def url(p):return quote(str(p).replace('\\','/'),safe='/:#')
def link(p,label=None):return f'<a href="{url(PACK.name+"/"+p)}">{esc(label or p)}</a>'
def table(headers,rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def pct(x):return f'{x:.1f}%'
def badge(s,kind='done'):return f'<span class="badge {kind}">{esc(s)}</span>'
def source(*paths):return '<p class="source">数据依据：'+' · '.join(link('results/'+p,Path(p).name) for p in paths)+'</p>'
def section(id,title,body,kicker='实验结果'):
    return f'<section id="{id}"><div class="eyebrow">{kicker}</div><h2>{title}</h2>{body}</section>'
def note(text,kind='note'):return f'<div class="{kind}">{text}</div>'

OUT.mkdir(parents=True,exist_ok=True);PACK.mkdir(parents=True,exist_ok=True)
tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True,encoding='utf-8').splitlines())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
manifest=[]
for sub in ['results','reports','media']:
    for p in sorted((ROOT/sub).rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if sub=='media':
            permitted = p.name.startswith('r3_official_') or any('/'+name+'/' in '/'+rel for name in ['sim2sim_mujoco','sim2sim_completed_20261002','mujoco_robust_contact','original_human_generation_20261002'])
        elif sub=='reports':
            permitted = p.name.startswith(('03_dreamcontrol_multitask','10_','11_','12_','13_','14_'))
        else:
            permitted = p.name.startswith(('r3_dreamcontrol_official_','r3_dreamcontrol_Sit_eval_failure','r3_dreamcontrol_Bimanual-Pick_eval_failure','r3_closd_','r3_dreamcontrol_remaining_manifest','official_other_tasks_assets_manifest','dreamcontrol_paper_task_matrix')) or any('/'+name+'/' in '/'+rel for name in ['configs','sim2sim_mujoco','sim2sim_completed_20261002','mujoco_robust_contact'])
            permitted = permitted and 'i1' not in rel.lower() and 'smoke' not in rel.lower()
        if not permitted:continue
        target=PACK/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
        digest=sha(p)
        assert sha(target)==digest
        manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=digest,tracked_in_source_git=rel in tracked))
snapshot={'captured_utc':'2026-10-02T08:04:37.094717+00:00','source':'read-only SSH monitoring log and state inspection',
    'dreamcontrol_task':'Ground-Pick','iteration':1733,'target_iterations':2000,
    'log_mtime_utc':'2026-10-02T08:04:30.627598+00:00','state':'RUNNING train',
    'remaining':['Ground-Pick-Top','Button-Press','Open-Drawer','Open-Door'],
    'closd_formal3072':'BLOCKED local_scene_confirmation_pending'}
(PACK/'results/report_progress_snapshot_20261002.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')

# Main results are read from completed, audited records; diagnostic archives stay separate.
kick=read('dreamcontrol_r3_completed.json')['experiments'][0]
(PACK/'results/DreamControl_Kick_official_evidence.json').write_text(json.dumps(kick,ensure_ascii=False,indent=2),encoding='utf-8')
humans=read('../media/original_human_generation_20261002/manifest.json')
assert humans['status']=='COMPLETE' and len(humans['videos'])==6
for v in humans['videos']:assert sha(PACK/'media/original_human_generation_20261002'/v['file'])==v['sha256']
punch=read('r3_dreamcontrol_official_punch_8192_eval.json')
tasks=[dict(task='Kick',score=kick['success_percent'],refs=66,video='r3_official_kick_8192_eval.mp4',hash=kick['video_sha256'],io='默认路径',state='原官方预算'),
       dict(task='Punch',score=punch['evaluation']['success_rate_percent'],refs=100,video='r3_official_punch_8192_eval.mp4',hash=punch['evaluation']['video_sha256'],io='默认路径',state='第1600轮恢复后完成')]
for t,n,v in [('Jump',1,'jump'),('Sit',1,'sit'),('Pick',67,'pick'),('Bimanual-Pick',1,'bimanual_pick')]:
    j=read(f'r3_dreamcontrol_official_{t}_8192_eval.json')
    tasks.append(dict(task=t,score=j['success_percent'],refs=n,video=f'r3_official_{v}_8192_eval.mp4',hash=j['video_sha256'],io='USD显示读写适配' if t in ['Sit','Bimanual-Pick'] else '默认路径',state='训练与评估正常结束'))
for j in tasks:
    assert sha(PACK/'media'/j['video'])==j['hash'],j['task']
matched=read('r3_i1_matched_completed.json')
assert matched['science_gate_passed'] is False
joint=read('r3_i1_full_no_ik_summary.json')['seeds']
ik=read('r3_i1_fixed_three_seed_summary.json')['seeds']
ext=read('r3_i1_extended_validation_summary.json')
closd=read('r3_closd_official_eval4096_20261001_summary.json')
cl512a=read('r3_closd_pretrained_eval_512.json');cl512b=read('r3_closd_finetuned_eval_512.json')
robust=read('mujoco_robust_contact/summary.json')
new=read('sim2sim_completed_20261002/summary.json')
assert new['reference_count']==70 and len(new['videos'])==5
for v in new['videos']:
    assert sha(PACK/'media/sim2sim_completed_20261002'/Path(v['file']).name)==v['sha256']

hero='''<header id="top"><div class="eyebrow">研究汇报 · 已完成实验档案</div><h1>通用条件人体运动生成</h1><p class="subtitle">人体动作生成方法、论文控制基线与跨仿真验证</p><p>数据截至 <b>2026年10月2日 16:04（北京时间）</b> · 文件夹沿用用户指定的20260929名称</p><div class="hero-grid"><div><strong>6 / 11</strong><span>DreamControl控制任务已完成</span></div><div><strong>2组</strong><span>CLoSD官方权重评估已完成</span></div><div><strong>6段</strong><span>原始扩散生成人体动作展示</span></div><div><strong>6项</strong><span>控制器已有MuJoCo测试记录</span></div></div><p class="hero-note">复现控制任务并不等于完成整篇论文；执行测试结束并不等于可靠迁移通过。</p></header>'''
overview=note('<b>本次汇报重点：</b>先说明人体动作怎样由文本与空间条件生成，再观看原始人体视频，最后查看机器人控制基线与跨仿真结果。六项官方控制任务已有结果；跨仿真测试仍存在稳定性和接触失败。','conclusion')
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

basebody='<p>六项均采用官方原参考库、8192个训练环境、2000次PPO迭代、seed42；完成至第1999轮后，用官方1000环境、500帧评估。Punch曾中断，从第1600轮恢复。</p>'
baserows=[]
for t in tasks:
    bar=f'<span class="metric">{pct(t["score"])}</span><div class="bar"><i style="width:{t["score"]}%"></i></div>'
    baserows.append([t['task'],str(t['refs']),bar,esc(t['io']),badge('完成'),f'<a href="#videos" data-select="baseline">查看视频</a>'])
basebody+=table(['任务','原参考数','官方任务成功率','评估运行路径','状态','展示'],baserows)
basebody+=note('<b>分数的实际含义：</b>Sit主要判断骨盆接近参考目标；Pick类主要使用物体高度等任务条件。高分本身不单独验收自然坐姿、持续坐垫接触、稳定双手抓握或动作语义。Jump、Sit与Bimanual各只有一条原参考，100%或99.9%不表示未见动作泛化。Sit和Bimanual默认视频初始化失败，后改用官方 <code>--disable_fabric</code> 的USD显示读写适配；PPO、奖励、参考和物理预算未改，但不能称完全相同运行配置。')
basebody+='<h3>未完成任务（状态快照，不是最终成绩）</h3>'+table(['任务','状态','可核查进度'],[['Ground-Pick',badge('训练中','running'),'1733 / 2000轮；2026-10-02 16:04北京时间'],*[[t,badge('排队','pending'),'未完成正式训练与评估'] for t in snapshot['remaining']]])
basebody+=source('DreamControl_Kick_official_evidence.json','r3_dreamcontrol_official_punch_8192_eval.json','r3_dreamcontrol_official_Jump_8192_eval.json','r3_dreamcontrol_official_Sit_8192_eval.json','r3_dreamcontrol_official_Pick_8192_eval.json','r3_dreamcontrol_official_Bimanual-Pick_8192_eval.json')

clbody='<h3>官方已发布权重：4096环境评估</h3><p>每组500帧、CFG7.5、seed0，每状态至少1000次重置；两组正常退出并完成清理。数字按原日志两位小数记录。</p>'
noft=next(x for x in closd if x['model']=='CLoSD_no_finetune');ft=next(x for x in closd if x['model']=='CLoSD_multitask_finetune')
clbody+=table(['任务状态','官方未微调权重','官方已微调权重','差值（百分点）'],[[name,pct(noft['success_rate_rounded_2dp'][key]*100),pct(ft['success_rate_rounded_2dp'][key]*100),f"{(ft['success_rate_rounded_2dp'][key]-noft['success_rate_rounded_2dp'][key])*100:+.0f}"] for name,key in [('触达','REACH'),('打击（踢/拳共用统计）','STRIKE_KICK'),('坐下','SIT'),('起身','GET_UP')]])
clbody+=note('这是<b>官方权重的评估复现</b>，不是我们重新训练后的成绩。Kick与Punch共用STRIKE计数，不能当成两项独立成绩，也不作简单均分。两次评估未录视频；不能用实时监控画面冒充保存的评估视频。')
clbody+='<h3>本地512环境缩容微调：保留退化结果</h3><p>从epoch62000继续到66000，最终模型已保存，但训练在保存后以退出码139异常退出。相同512环境评估正常完成。</p>'
clbody+=table(['任务状态','微调前官方模型','本地微调后','差值（百分点）'],[[name,pct(cl512a['success_rate_rounded_as_logged'][key]*100),pct(cl512b['success_rate_rounded_as_logged'][key]*100),f"{(cl512b['success_rate_rounded_as_logged'][key]-cl512a['success_rate_rounded_as_logged'][key])*100:+.0f}"] for name,key in [('触达','REACH'),('打击（共用统计）','STRIKE_KICK'),('坐下','SIT'),('起身','GET_UP')]])
clbody+=note('<b>结论：</b>除起身外明显退化，不重复同配置训练，不称论文同配置复现。原退出139的具体根因仍未证实。3072环境三次更新诊断保存epoch62003并正常退出，只是容量/环境检查；新的4000轮正式微调仍待当前实际场景的本地可视确认，不能计入完成。','warning')
clbody+=source('r3_closd_official_eval4096_20261001_summary.json','r3_closd_pretrained_eval_512.json','r3_closd_finetuned_eval_512.json','r3_closd_capacity3072_20261001.json')

i1body='<h3>早期数据增强：没有显示留出集收益</h3>'+table(['实验','训练数据与预算','测试参考','结果'],[['原66条策略','1024环境、2000轮','独立生成的32条留出；1000环境500帧','0%'],['98条增强组','原66＋新增32；1024环境、2000轮、seed123','同一32条留出；1000环境500帧','0%']])
i1body+=note('这两组0%与官方8192环境原参考库85.3%不属于同一测试集与训练预算，不能用85.3→0推断数据增强的因果效果。两组训练种子亦不完全一致。')
i1body+='<h3>额外IK后处理：几何变好，固定控制器跟踪变差</h3><p>同一100个Kick空间输入，采样种子42/43/44。人体生成器不变，只改重定向后的右腿关节；已按用户要求停用该额外IK。</p>'
i1body+=table(['种子','成功重定向','几何误差 原始→IK（米）','全部100输入：15cm阈值通过','加变化速度筛查后通过','固定策略仿真成功率 原始→IK'],[[str(x['seed']),str(x['retargeted']),f"{x['retargeted_mean_before_m']:.3f} → {x['retargeted_mean_after_m']:.3f}",f"{x['retargeted_success_15cm_count']}/100",f"{x['retargeted_success_and_speed_screen_count']}/100",f"{y['official_goal_proximity_success_percent_raw']:.2f}% → {y['official_goal_proximity_success_percent_corrected']:.2f}%"] for x,y in zip(ik,ext['paired_seed_rows'])])
i1body+='<p>另有seed42的56条留出运动学组件检查：平均误差0.420→0.125米，41/56达15cm阈值，加启发式速度筛查后32/56；不属于动力学或语义验证。</p>'
i1body+=table(['固定原始轨迹派生目标：seed42','原始参考','IK≤0.25rad','IK≤0.50rad','IK≤1.00rad'],[['目标邻近成功率',*[f"{ext['frozen_raw_goal_seed42'][k]['official_goal_proximity_success_percent']:.2f}%" for k in ['raw','cap025','cap050','cap100']]],['关节跟踪平均误差（rad）',*[f"{ext['frozen_raw_goal_seed42'][k]['joint_mae_rad']:.3f}" for k in ['raw','cap025','cap050','cap100']]]])
i1body+=note('固定目标后仍退化，不能全部归因于目标重新计算；但固定策略没有在IK新分布上训练，也不能由此断言机器人硬件绝无可能执行。上述固定目标仍由原参考轨迹派生，不是独立外部真值。')
i1body+='<h3>语义、接触与生成效率：完成的是有限代理检查</h3>'+table(['指标','已测结果','边界'],[['踢腿事件幅度/时序保留','58/64、61/64、55/63','只有一个Kick文本，不是多文本语义匹配基准'],['足部接触','右踝接触帧比例由约0.75降至0.27–0.33','只统计身体连杆接触，未测目标撞击力'],['扩散采样耗时','每100条、1000去噪步：413/415/415秒','不含加载、重定向和仿真'],['额外IK开销','每条平均0.747/0.755/0.763秒，80次Adam更新','增加成本，未证明生成效率收益']])
i1body+='<h3>无额外IK联合生成：人体改善未保持到G1目标</h3><p>使用相同预训练OmniControl权重、文本和原右脚目标；候选在扩散采样中加入空间、全身协调、支撑脚代理和时序引导。两组经过同一原始官方G1重定向，未微调扩散权重。</p>'
jointrows=[]
for seed,j in joint.items():
    a=j['all_input_human_metrics']['baseline'];b=j['all_input_human_metrics']['joint_v2']
    jointrows.append([seed,f"{a['original_right_foot_hint_error_m']:.3f} → {b['original_right_foot_hint_error_m']:.3f}",f"{a['left_foot_stance_speed_m_per_frame']:.5f} → {b['left_foot_stance_speed_m_per_frame']:.5f}",f"{j['baseline_retargeted']} → {j['candidate_retargeted']} /100",str(j['same_id_retarget_count']),f"{j['same_id_robot_mean_error_m']['baseline']:.3f} → {j['same_id_robot_mean_error_m']['joint_v2']:.3f}"])
i1body+=table(['采样种子','人体原目标误差（米）','支撑脚帧间位移（米/帧）','重定向接受数','共同编号数','共同编号G1世界目标误差（米）'],jointrows)
hm_a=[j['all_input_human_metrics']['baseline']['original_right_foot_hint_error_m'] for j in joint.values()]
hm_b=[j['all_input_human_metrics']['joint_v2']['original_right_foot_hint_error_m'] for j in joint.values()]
i1body+=f'<p>三个采样种子的人体原目标平均误差：Base {statistics.mean(hm_a):.4f}±{statistics.stdev(hm_a):.4f}米，候选 {statistics.mean(hm_b):.4f}±{statistics.stdev(hm_b):.4f}米（种子间样本标准差，仅描述采样波动，不是跨训练种子显著性）。</p>'
i1body+=note('重定向接受率提升是流程产出率，不是仿真成功率。两臂三种子在原外部人类世界坐标目标15cm阈值下均为0/100；坐标/形态映射尚未解决。支撑脚稳定只是接触代理，Kick原场景没有实体目标。首版8条密集条件试验使右脚误差0.161→0.413米而失败；第二版开发8条改善至0.011米，但独立机器人目标仍未改善，失败版本也保存在明细档案。')
i1body+='<h3>最终同预算PPO：候选在四个相同参考库上均低于Base</h3><p>seed42共同61条训练参考，各8192环境、2000轮，从零训练，同奖励/网络/PPO。测试生成种子43/44共同62/59条，每策略交叉测试两种参考，共8次1000环境500帧评估；均核验模型和视频。</p>'
matchedrows=[]
for seed in [43,44]:
    for arm in ['baseline','candidate']:
        es=[x for x in matched['evaluations'] if x['test_seed']==seed and x['reference_arm']==arm]
        aa=next(x for x in es if x['policy']=='baseline')['success_percent'];bb=next(x for x in es if x['policy']=='candidate')['success_percent']
        matchedrows.append([str(seed),'Base参考' if arm=='baseline' else '候选参考',pct(aa),pct(bb),f'<b class="negative">{bb-aa:+.1f}</b>'])
i1body+=table(['留出生成种子','相同测试参考库','Base策略','候选策略','候选−Base（百分点）'],matchedrows)
i1body+=note('<b>I1完整验收未通过。</b>只有一个控制训练seed、一个Kick文本，生成种子43/44不是两次独立PPO训练；不作跨训练seed显著性推断。成功率使用参考派生虚拟目标，不证明外部条件保持或真实接触。候选人体几何改善不能替代下游收益。本轮缺完整模块消融、跨文本、生成效率提升；I2/I3正式阶段未完成。','warning')
i1body+=source('dreamcontrol_r3_completed.json','r3_i1_fixed_three_seed_summary.json','r3_i1_extended_validation_summary.json','r3_i1_holdout56_summary.json','r3_i1_full_no_ik_summary.json','r3_i1_matched_completed.json')

simbody='<p>将自己的已训练策略与观测归一化器冻结，转到MuJoCo真实物理执行；没有重新训练、额外IK或参考轨迹回放。物理200Hz、策略50Hz，每条原参考独立从0秒执行，最多500控制步/10秒，不自动重置，失败保留。</p>'
simrows=[]
for t in ['Kick','Punch']:
    j=read(f'sim2sim_mujoco/{t}_zero_shot.json');n=j['num_references']
    simrows.append([t,str(n),f"{j['success_count']}/{n}",f"{j['survive_full_horizon_count']}/{n}",f"{j['success_and_survive_full_horizon_count']}/{n}",f"{j['fall_count']}/{n}",'到虚拟目标＋完整10秒不摔倒'])
for j in new['tasks']:
    n=j['num_references'];explanation={'Jump':'初始着地后再次腾空＋稳定落地1秒','Sit':'骨盆/躯干与座椅接触1秒','Pick':'右手接触且相对抬物≥5cm、持续1秒','Bimanual-Pick':'双手接触且相对抬物≥5cm、持续1秒'}[j['task']]
    simrows.append([j['task'],str(n),f"{j['official_proxy_success_count']}/{n}",f"{j['full_horizon_survival_count']}/{n}",f"{j['additional_task_check_success_count']}/{n}",f"{j['fall_count']}/{n}",explanation])
simbody+=table(['任务','试验数','原任务位置/高度代理达标','完整10秒存活','附加任务检查通过','摔倒','附加检查定义'],simrows)
simbody+=note('<b>这些列不能横向当作同一成功标准：</b>踢/拳的附加检查是到虚拟目标并站稳；其他任务检查实际接触或腾空落地。与Isaac1000环境自动重置口径亦不同，不能直接计算跨平台下降百分比。Jump/Sit/Bimanual只有一次参考试验，没有总体泛化成功率。')
simbody+='<h3>新增四项的失败表现</h3>'+table(['任务','观察','解释边界'],[['Jump','约2.9秒摔倒，未完成稳定落地','需要定位驱动、动力学或分布差异；不能仅由一次试验确定根因'],['Sit','达参考目标但无持续坐垫接触','沿用训练座椅中心(0,0,0)，与参考目标位置不对齐，未为提高分数移动座椅'],['Pick','67条均运行10秒，部分有手部接触，但物体未相对抬升5cm','不把身体站稳当作抓取成功；原绝对高度判据也有局限'],['Bimanual-Pick','运行10秒、出现手部接触，未抬升物体','单参考负结果，不推广为所有双手技能均失败']])
simbody+=note('接口核验通过：冻结actor/normalizer与原检查点逐张量一致；每任务30个真实Isaac观测/动作快照误差&lt;0.0002；新增70条参考插值/阶段零错配；接触检测正负夹具通过。<b>这些是实现核验，不是任务通过证明。</b>MuJoCo使用CPU物理、CUDA策略和NVIDIA EGL渲染；USD/MJCF惯量/碰撞形状尚未逐项证明相同，Isaac隐式驱动与显式PD不同。')
simbody+=source('sim2sim_mujoco/Kick_zero_shot.json','sim2sim_mujoco/Punch_zero_shot.json','sim2sim_completed_20261002/summary.json','sim2sim_completed_20261002/contact_detector_fixtures.json')

robbody='<p>冻结Kick/Punch策略，14个条件×(66＋100)条参考＝<b>2324次试验</b>。包括实际摩擦0.5/1.0/1.5、机器人质量±10%、力矩上限−10%、左右20/40N推力，以及原位置/侧移5cm的固定实体靶。均为10秒上限、无自动重置的单参考测试。</p>'
condition_names={'nominal':'标称','friction_low':'低摩擦0.5','friction_high':'高摩擦1.5','nominal_mass_constants_refresh':'质量对照（刷新常数）','mass_low':'质量−10%','mass_high':'质量＋10%','strength_low':'力矩上限−10%','push_left_20':'左推20N','push_right_20':'右推20N','push_left_40':'左推40N','push_right_40':'右推40N','pad_nominal':'原位置实体靶','pad_y_plus_5cm':'实体靶＋5cm','pad_y_minus_5cm':'实体靶−5cm'}
robrows=[]
for task,j in robust['tasks'].items():
    for c in j['conditions']:
        n=c['n'];pad=c['condition'].startswith('pad_')
        robrows.append([task,condition_names.get(c['condition'],c['condition']),f"{c['virtual_goal_reached']}/{n}",f"{c['stable_goal_success']}/{n}",f"{c['valid_strike_10n']}/{n}" if pad else '—',f"{c['stable_strike_success']}/{n}" if pad else '—',f"{c['fallen']}/{n}"])
robbody+=table(['任务','测试条件','到达虚拟目标','到达＋全程站稳','合格实体打击','打击＋全程站稳','摔倒'],robrows)
robbody+=note('合格打击需在事件±0.3秒窗口内，正确脚/手碰撞靶面、方向对齐≥0.5、法向力≥10N、碰撞前运动速度代理≥0.1m/s。是固定刚体靶，不是对手或标定沙袋；不等于实机打击。原位置靶：Kick39/66合格、9/66打击且站稳；Punch42/100、33/100。高摩擦下明显退化，不能宣称鲁棒性通过。','warning')
robbody+='<p>首版仅改地面摩擦，实际合并接触摩擦不等于设定值；保留旧诊断后完成V2全量重跑，同时修改地面/机器人摩擦并逐接触核验。质量对照使用刷新常数的标称条件，不能误配原标称比较。</p>'
robbody+=source('mujoco_robust_contact/summary.json','mujoco_robust_contact/execution_manifest.json','mujoco_robust_contact/v2_configuration_witness.json')

failurebody=table(['记录','现象','处理及边界'],[
 ['Sit/Bimanual视频初始化','默认路径native malloc异常','保留模型/失败日志，官方USD显示读写适配后正常评估；底层根因未证实'],
 ['CLoSD旧512微调','保存66000后退出139、部分任务退化','同口径前后评估已保存，不视为论文3072环境完整训练'],
 ['直播曾影响训练','耦合WebRTC/GLFW停滞','现已分离训练与实际状态监控，显示故障不阻塞PPO'],
 ['MuJoCo接触/动作失败','新增任务没有可靠迁移通过','保留场景/驱动差异与实际负结果，不把站稳当成抓取成功'],
 ['摩擦测试首版','实际接触摩擦与设定不符','V2完整重跑并逐接触验证，汇报使用最终V2结果']])
failurebody+=note('当前DreamControl只完成6/11控制任务，不等于整篇论文完成；CLoSD官方权重评估完成不等于本地重新训练完成。人体生成骨架、机器人参考、Isaac执行、MuJoCo执行是不同层次，报告分别标注。没有实机验证。')
failurebody+='<p>后续先继续既有论文复现队列、补齐CLoSD正式训练；对跨仿真失败核对任务场景、惯量/碰撞与驱动映射，再按独立协议验证，不覆盖本轮负结果。</p>'

# All copied videos are playable inside the one main report.
categories=[]
def addvideo(path,label,group,note_text):
    assert (PACK/'media'/path).is_file()
    categories.append((path,label,group,note_text))
for t in tasks:addvideo(t['video'],t['task']+'｜官方Isaac评估','baseline',pct(t['score'])+'；1000环境500帧，官方原参考库')

for folder,group,label in [('sim2sim_mujoco','sim','踢/拳MuJoCo'),('sim2sim_completed_20261002','sim','新增四项MuJoCo'),('mujoco_robust_contact','robust','MuJoCo扰动/接触')]:
    for p in sorted((PACK/'media'/folder).glob('*.mp4')):addvideo(f'{folder}/{p.name}',label+'｜'+p.stem,group,'固定参考编号展示，未按成绩挑选；完整数值请看上方表格')
video_body='<p>所有视频已复制到汇报文件夹，离线可播放。点击分组筛选，播放时可全屏。CLoSD官方两组评估没有录视频，未用其他画面替代。</p><div class="filters">'+''.join(f'<button data-filter="{k}" class="{"active" if k=="all" else ""}">{v}</button>' for k,v in [('all','全部机器人执行'),('baseline','论文控制基线'),('sim','MuJoCo迁移'),('robust','扰动与实体接触')])+'</div><div class="video-grid">'
for p,title,group,n in categories:
    poster=(PACK/'media'/p).with_suffix('.jpg');attr=f' poster="{url(PACK.name+"/media/"+str(Path(p).with_suffix(".jpg")).replace(chr(92),"/"))}"' if poster.is_file() else ''
    video_body+=f'<article class="video-card" data-group="{group}"><h3>{esc(title)}</h3><video controls preload="none"{attr}><source src="{url(PACK.name+"/media/"+p)}" type="video/mp4"></video><p>{esc(n)}</p><a href="{url(PACK.name+"/media/"+p)}">单独打开视频</a></article>'
video_body+='</div>'

# Lightweight local-only Markdown renderer for readable full-report appendix.
def inline(s):
    s=esc(s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    def a(m):
        href=html.unescape(m[2]);display=m[1]
        if not re.match(r'^(https?://|#)',href):href=url(PACK.name+'/reports/'+href)
        return f'<a href="{esc(href)}">{display}</a>'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',a,s)
def markdown(txt):
    lines=txt.splitlines();out=[];i=0;in_code=False
    while i<len(lines):
        s=lines[i]
        if s.startswith('```'):
            out.append('<pre>' if not in_code else '</pre>');in_code=not in_code;i+=1;continue
        if in_code:out.append(esc(s)+'\n');i+=1;continue
        if s.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:|\-]+$',lines[i+1]):
            h=[inline(x.strip()) for x in s.strip('|').split('|')];i+=2;rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                rows.append([inline(x.strip()) for x in lines[i].strip('|').split('|')]);i+=1
            out.append('<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>');continue
        m=re.match(r'^(#{1,6})\s+(.+)',s)
        if m:out.append(f'<h{min(5,len(m[1])+2)}>{inline(m[2])}</h{min(5,len(m[1])+2)}>')
        elif s.strip() in ['---','***']:out.append('<hr>')
        elif s.strip():out.append('<p>'+inline(s)+'</p>')
        i+=1
    return ''.join(out)
appendix='<p>本节只收录论文复现和跨仿真证据。历史进度以本汇报最新结构化结果为准，可搜索文件名/主题。</p><input id="search" placeholder="搜索报告或数据文件，例如 contact、seed42、CLoSD">'
for p in sorted((ROOT/'reports').glob('*.md')):
    if not p.name.startswith(('10_','11_','13_','14_')):continue
    rel=p.relative_to(ROOT).as_posix();draft=rel not in tracked
    appendix+=f'<details class="searchable"><summary>{esc(p.stem)} {badge("探索草稿，未作为汇报结论依据","pending") if draft else ""}</summary><div class="full-report">{markdown(p.read_text(encoding="utf-8-sig"))}</div><p>{link(rel,"查看原始Markdown")}</p></details>'
appendix+='<details class="searchable"><summary>本次论文复现结果文件索引（包括环境见证与失败记录）</summary><p>文件索引仅覆盖本次汇报的复现资料，不把文件数量当成实验完成数。</p>'+table(['文件','来源状态','大小'],[[link(x['path']), '已纳入来源仓库' if x['tracked_in_source_git'] else '未纳入来源仓库／探索记录',f"{x['bytes']:,} B"] for x in manifest if x['path'].startswith('results/')])+'</details>'
appendix+=f'<p>来源仓库：<a href="https://github.com/Pendragon08/conditional-human-motion-generation">conditional-human-motion-generation</a>；报告读取时来源提交 <code>{head[:7]}</code>。完整模型/日志路径和SHA见结果JSON；权重与受许可资产不重复打包。</p><p>{link("results/report_progress_snapshot_20261002.json","只读进度快照")} · <a href="{url(PACK.name+"/归档清单.json")}">归档清单及SHA256</a></p>'

css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f6;color:#19314a;font-family:system-ui,"Microsoft YaHei",sans-serif;font-size:16px;line-height:1.75}a{color:#126498;text-decoration:none}a:hover{text-decoration:underline}nav{position:fixed;inset:0 auto 0 0;width:225px;padding:32px 22px;background:#10273d;color:#c9d8e7;overflow:auto}nav b{font-size:18px;color:white;display:block;margin-bottom:22px}nav a{display:block;color:#c9d8e7;padding:9px 0;font-size:14px}nav small{display:block;margin-top:24px;color:#92a9be}.main{margin-left:225px;padding:32px;max-width:1600px}header{padding:42px;background:linear-gradient(130deg,#123650,#116f7d);color:#fff;border-radius:20px}h1{font-size:40px;margin:8px 0;letter-spacing:1px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;text-transform:uppercase;color:#16828d}.subtitle{font-size:20px;opacity:.94}header .eyebrow{color:#b9e9e6}.hero-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.hero-grid div{padding:20px;background:#ffffff12;border:1px solid #ffffff26;border-radius:12px}.hero-grid strong{font-size:32px;display:block}.hero-grid span{font-size:13px}.hero-note{font-size:14px;opacity:.85}section{margin:26px 0;padding:32px;background:#fff;border-radius:16px;box-shadow:0 5px 18px #122b4210;scroll-margin-top:20px}h2{font-size:27px;margin:4px 0 20px}h3{font-size:19px;margin:26px 0 12px}.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th{text-align:left;background:#eaf1f5;color:#36546c;padding:12px;border-bottom:2px solid #c8d9e3;white-space:nowrap}td{padding:12px;border-bottom:1px solid #e2eaf0;vertical-align:top}tr:nth-child(even) td{background:#f8fafc}.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;white-space:nowrap}.done{background:#dcf3e9;color:#1c7553}.running{background:#dfeefd;color:#245b8c}.pending{background:#edf0f3;color:#596775}.fail{background:#fde7df;color:#a14531}.note,.warning,.conclusion{padding:18px 22px;margin:20px 0;border-radius:10px;border-left:4px solid #168b91;background:#ecf6f7}.warning{border-color:#bd6842;background:#fff0e7}.conclusion{background:#f0f6e9;border-color:#7d964d;font-size:18px}.source{font-size:12px;color:#7b8b96;overflow-wrap:anywhere}.source a{margin-right:5px}code{background:#eef2f6;padding:2px 5px;border-radius:4px;font-size:12px;overflow-wrap:anywhere}.negative{color:#b04935}.metric{font-weight:800;font-size:18px}.bar{height:5px;background:#e7eef4;border-radius:4px;width:120px;margin-top:6px}.bar i{display:block;height:5px;background:#18898d;border-radius:4px}.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}button{border:1px solid #cbd9e2;background:#fff;padding:9px 14px;border-radius:30px;cursor:pointer;color:#2d516b}button.active{background:#126477;color:#fff;border-color:#126477}.video-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.video-card{padding:18px;border:1px solid #dce5ed;border-radius:12px;background:#f8fafb}.video-card h3{margin:0 0 12px;font-size:16px}.video-card p{font-size:12px;color:#607585}video{width:100%;aspect-ratio:16/9;background:#102231;border-radius:8px}details{border:1px solid #d7e2ea;border-radius:8px;margin:12px 0;padding:14px}summary{cursor:pointer;font-weight:700;overflow-wrap:anywhere}.full-report{font-size:14px}.full-report p{margin:7px 0}.full-report h3{font-size:20px}.full-report h4{font-size:17px}.full-report table{font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f7;padding:15px;font-size:12px}input{width:100%;border:1px solid #c2d4e0;padding:12px;border-radius:9px}footer{font-size:12px;color:#758894;padding:15px}.toolbar{display:flex;gap:10px;align-items:center;margin:0 0 16px}.toolbar span{font-size:12px;color:#647987}.hidden{display:none!important}@media(max-width:1050px){nav{position:relative;width:auto;display:flex;flex-wrap:wrap;gap:16px;padding:18px}nav b,nav small{display:none}nav a{padding:0}.main{margin:0;padding:15px}.hero-grid{grid-template-columns:repeat(2,1fr)}header,section{padding:24px}h1{font-size:30px}.video-grid{grid-template-columns:1fr}}@media print{nav,.toolbar,.filters,video,input{display:none}.main{margin:0;padding:0}header,section{box-shadow:none;border:1px solid #ddd;break-inside:avoid;margin:15px 0}header{color:#19314a;background:#eef4f7}.hero-grid div{border:1px solid #ccd9e0}.hero-note{color:#36556b}.video-grid{display:none}a{color:inherit}details:not([open]){display:none}th,td{font-size:10px;padding:7px}}'''
nav=[('overview','01 汇报概况'),('methods','02 人体运动怎样生成'),('generation','03 先观看人体生成视频'),('baseline','04 DreamControl控制基线'),('closd','05 CLoSD复现'),('sim','06 MuJoCo跨仿真'),('robust','07 扰动与实体接触'),('failures','08 运行问题与后续'),('videos','09 机器人执行视频'),('appendix','10 复现明细与证据')]
body=hero+section('overview','论文复现汇报概况',overview,'先看结论')+section('methods','人体运动生成方法：从条件到执行',methods,'方法说明')+section('generation','原始人体生成：固定六条样本',generation,'供老师观察动作质量')+section('baseline','DreamControl：六项官方控制基线',basebody)+section('closd','CLoSD：权重评估与本地微调',clbody)+section('sim','Isaac Lab → MuJoCo：六项冻结策略迁移',simbody)+section('robust','Kick/Punch：扰动鲁棒性与实体打击',robbody)+section('failures','运行问题、证据边界与后续',failurebody)+section('videos','机器人执行视频',video_body,f'{len(categories)} 段执行视频')+section('appendix','论文复现明细与来源',appendix,'可复查证据')
css+='.method-flow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#f0f6f8;padding:18px;border-radius:12px}.method-flow span{padding:12px;background:white;border:1px solid #cadde6;border-radius:8px;font-size:14px}.human-preview{border:2px solid #9acac3}.human-preview h3{color:#146d66}'

js='''function choose(group){document.querySelectorAll('.video-card[data-group]').forEach(c=>{let hide=group!=='all'&&c.dataset.group!==group;c.classList.toggle('hidden',hide);if(hide)c.querySelector('video').pause()});document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter===group))}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>choose(b.dataset.filter));document.querySelectorAll('[data-select]').forEach(a=>a.onclick=()=>choose(a.dataset.select));document.querySelector('#search').oninput=e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('.searchable').forEach(d=>d.classList.toggle('hidden',!d.textContent.toLowerCase().includes(q)))};document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(o=>{if(o!==v)o.pause()})));'''
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>通用条件人体运动生成｜论文复现与人体生成展示</title><style>'+css+'</style></head><body><nav><b>实验汇报</b>'+''.join(f'<a href="#{k}">{v}</a>' for k,v in nav)+'<small>截至2026-10-02<br>本地离线档案<br>人体生成与控制执行分别展示</small></nav><main class="main"><div class="toolbar"><button onclick="window.print()">打印 / 保存PDF</button><span>浏览器可离线打开；汇报正文、明细和视频均在本文件夹。</span></div>'+body+'<footer>本报告统计完成的实验，不将排队、容量检查、显示预检、文件数或高代理分数等同于论文验收。来源代码与完整模型路径见原始证据。</footer></main><script>'+js+'</script></body></html>'
page=page.replace('没有使用原来的 98 条增强组或 I1 候选策略。','').replace('没有替换成中途恢复策略或创新策略。','')
REPORT.write_text(page,encoding='utf-8')
archive={'report':'实验结果汇报.html','generated_at_local':datetime.datetime.now().isoformat(),'source_git_commit':head,'data_cutoff_beijing':'2026-10-02 16:04',
    'baseline_tasks_completed':6,'baseline_tasks_total':11,'human_motion_videos':6,'new_sim2sim_references':70,'robust_contact_trials':2324,
    'video_count':len(categories)+6,'files':manifest,'report_scope':'baseline reproduction and original human generation; research comparison experiments excluded','copy_hashes_all_verified':True}
(PACK/'归档清单.json').write_text(json.dumps(archive,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'打开汇报说明.txt').write_text('请双击“实验结果汇报.html”。报告数据截至2026年10月2日16:04，文件夹名沿用20260929。\n离线视频和原始证据在“论文复现汇报资料_20261002”内；移动时请连同该文件夹一起移动。原完整档案版另存备份；本次正文仅汇报论文复现。\n',encoding='utf-8')
print(json.dumps({'report':str(REPORT),'html_bytes':REPORT.stat().st_size,'copied_files':len(manifest),'copied_bytes':sum(x['bytes'] for x in manifest),'videos':len(categories),'source_commit':head},ensure_ascii=False))
