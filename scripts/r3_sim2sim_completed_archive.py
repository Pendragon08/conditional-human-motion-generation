"""Validate finished transfer outputs and build a concise gallery and audit report."""
import json
from pathlib import Path
import subprocess
import html
from r3_sim2sim_completed_mujoco import ROOT,RESULT,MEDIA,TASKS,base

labels={'Jump':'跳跃','Sit':'坐下','Pick':'单手拾取','Bimanual-Pick':'双手拾取'}
allrows=[];videos=[]
for task in TASKS:
    p=RESULT/f'{task}_zero_shot.json';j=json.loads(p.read_text())
    assert j['status']=='COMPLETE' and not j['learning'] and j['policy_frozen']
    state=ROOT/'artifacts/monitoring'/f'r3_sim2sim_completed_20261002_{task}_eval.state'
    assert state.read_text().startswith('COMPLETE')
    assert j['num_references']==len(j['trials'])
    assert j['num_references']==(67 if task=='Pick' else 1)
    for trial in j['trials']:
        v=trial.get('video')
        if not v:continue
        v=Path(v)
        f=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(v)],text=True))
        assert float(f['format']['duration'])>0 and v.stat().st_size>0
        frame=MEDIA/(v.stem+'.jpg')
        subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-y','-ss',str(float(f['format']['duration'])/2),
            '-i',str(v),'-frames:v','1',str(frame)],check=True)
        assert frame.is_file() and frame.stat().st_size>1000
        videos.append({'task':task,'reference_index':trial['reference_index'],'file':str(v),
            'sha256':base.sha(v),'bytes':v.stat().st_size,'ffprobe':f,'representative_frame':str(frame)})
    parity=json.loads((RESULT/f'{task}_interface_parity.json').read_text())
    assert parity['passed']
    j['archive_verification']={'eval_exit':0,'model_hash_verified':True,'interface_parity_passed':True,
        'video_verified':True,'local_video_selection':'fixed reference indices0 and1; no outcome selection'}
    allrows.append(j)
audit=json.loads((RESULT/'reference_interpolation_audit.json').read_text());assert audit['passed']
contacts=json.loads((RESULT/'contact_detector_fixtures.json').read_text());assert contacts['passed']
summary={'status':'COMPLETE','date':'2026-10-02','scope':'four newly completed frozen baselines; Kick/Punch excluded from rerun',
    'reference_count':sum(x['num_references'] for x in allrows),'learning':False,
    'physics':'native MuJoCo3.2.3 CPU; CUDA policy/NVIDIA EGL GPU0',
    'tasks':allrows,'videos':videos,'reference_audit':audit,'contact_detector_fixtures':contacts,'evaluation_video_verified':True}
(RESULT/'summary.json').write_text(json.dumps(summary,indent=2))
rows=[];md=[];sections=[]
for j in allrows:
    t=j['task'];n=j['num_references'];proxy=j['official_proxy_success_count'];extra=j['additional_task_check_success_count']
    survive=j['full_horizon_survival_count'];falls=j['fall_count']
    rows.append(f'<tr><td>{t}｜{labels[t]}</td><td>{n}</td><td>{proxy}/{n}</td><td>{survive}/{n}</td><td>{extra}/{n}</td><td>{falls}/{n}</td></tr>')
    md.append(f'| {t} | {n} | {proxy}/{n} | {survive}/{n} | {extra}/{n} | {falls}/{n} |')
    clips=[]
    for v in videos:
        if v['task']!=t:continue
        name=Path(v['file']).name
        clips.append(f'<p>固定参考编号 {v["reference_index"]}，{float(v["ffprobe"]["format"]["duration"]):.2f} 秒</p><video controls preload="metadata" poster="{Path(v["representative_frame"]).name}"><source src="{html.escape(name)}" type="video/mp4"></video>')
    sections.append(f'<section><h2>{t}｜{labels[t]}</h2>'+''.join(clips)+'</section>')
table='<table><thead><tr><th>任务</th><th>参考／试验数</th><th>原判据代理</th><th>完整10秒不摔倒</th><th>额外接触／时序检查</th><th>摔倒</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四项已完成策略｜MuJoCo迁移验证</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;background:#f4f6f8;color:#16273c;line-height:1.7}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd5df;padding:10px}section{background:white;padding:20px;margin:22px 0;border-radius:12px}video{width:100%;max-width:800px}aside{padding:18px;background:#fff0ce;border-radius:10px}a{color:#155abc}</style><h1>已完成策略的 MuJoCo sim-to-sim 验证</h1><p>2026-10-02｜使用自己训练的最终策略，冻结权重；新引擎真实物理执行，未重新训练、未加IK、未播放参考轨迹。</p>'''+table+'''<aside><b>如何读这些结果：</b>“原判据代理”沿用任务的空间／绝对高度条件，不代表完整语义动作。拾取物体初始高度可能已超过原阈值，需同时看实际相对抬升与接触。Jump/Sit/Bimanual每项只有一条参考，不能当成总体成功率。此处是每参考一次无自动重置的10秒试验，与Isaac1000环境自动重置成绩不能直接比较。</aside><p>额外检查：跳跃在首次着地后再腾空，并落地稳定1秒；坐下要求座椅与骨盆／躯干接触1秒；单手／双手拾取要求对应手接触物体、相对抬升至少5厘米且持续1秒。这些是追加的物理检查，尚不足以证明自然动作质量或实机可用性。</p>'''+''.join(sections)+'''<p><a href="../../reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md">完整报告</a> · <a href="../../results/sim2sim_completed_20261002/summary.json">逐条指标及哈希</a> · <a href="../sim2sim_mujoco/results.html">已有Kick/Punch迁移视频</a></p></html>'''
(MEDIA/'results.html').write_text(page,encoding='utf-8')
report='''# Jump、Sit、Pick、Bimanual-Pick：冻结策略的 MuJoCo 跨仿真验证

2026-10-02。按用户授权补做四项已完成控制子任务。原Kick/Punch跨仿真结果不重跑；GPU1现有官方基线训练未中断。四项使用各自model_1999.pt的冻结actor和原观测归一化器，没有训练、额外IK、重定向修改或参考回放。

## 已执行结果

| 任务 | 原参考／试验数 | 原任务判据代理成功 | 完整10秒不摔倒 | 追加接触／时序检查通过 | 摔倒 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(md)+'''

70条参考全部计入分母，每条从0秒独立启动，500控制步上限，摔倒立即结束，不自动重置或删失败。固定参考编号0和1录视频，单参考任务只录0，没有按结果挑选。每项正常退出0；视频已核对正时长、帧数、哈希及代表截图。

**这些计数不是原Isaac1000环境自动重置评估的同口径成功率，不能直接计算跨平台下降百分比。单参考任务n=1，没有统计泛化结论。**

Jump在约2.9秒摔倒；Sit完整运行10秒并接近参考目标，但未持续接触座椅；Pick的67条与Bimanual的1条均完整运行10秒，但没有达到相对抬升5厘米的要求。因此四项迁移测试已经执行完毕，不能称四项任务可靠迁移通过。

座椅沿用实际训练配置：中心(0,0,0)、尺寸(0.35,0.35,0.42)米，与本条参考的坐下目标位置不对齐。这会影响真实座椅接触判据，不能仅凭零接触断定策略或物理引擎是唯一原因，亦未为了提高分数移动座椅。独立正负接触夹具检查已通过：重叠时能检测坐垫/手部接触，远离时无误检，拾取物体具有自由关节、真实采样质量和地面碰撞；这些检查没有修改已评分试验。实际拾取试验有手部接触，但物体未抬升。

## 判据解释

- Jump原判据代理：事件时间窗内躯干距目标小于0.05米；追加检查是在初始着地后双脚再次离地至少0.04秒，随后落地稳定至少1秒且不摔倒，避免初始增加0.15米高度后下落被误认成跳跃。
- Sit原判据代理：事件时间窗内骨盆距目标小于0.05米；追加检查为座椅与骨盆或躯干接触连续1秒且不摔倒。并未独立证明自然坐姿或人体工学。
- Pick原判据代理：closed阶段物体中心高于0.95米；Bimanual为0.7米。Pick重置高度为1.0米，原绝对高度条件可能在真正抬物前就满足。因此另外要求相对重置位置抬升至少0.05米，并持续右手／双手接触1秒，且不摔倒。不得把原判据高分写成成功抓握搬运。

## 冻结策略与接口证据

策略对应已归档的官方预算基线：8192环境、2000迭代、训练seed42，各任务最终1999检查点。加载其TorchScript策略，逐张量核对actor和normalizer等于原检查点。没有替换成上游发布策略或创新组策略。

真实Isaac单环境导出原始参考张量、默认关节位置、驱动增益/限幅/armature、二值手指开闭命令、实际启动物体质量和30个观测—动作快照。Jump/Sit输入633维、Pick654维、Bimanual655维，均输出29维。各项保留不同的未来参考时距和任务额外项。原函数数值AST直接执行，避免手写不同的姿态目标。

30个真实快照的直接观测、策略输出和注入MuJoCo状态后的观测，全部最大绝对误差小于0.0002。全部70条参考另在16个时刻（含负时间与未来偏移）与未修改的官方MotionLibBase函数对照，误差小于0.0002且事件阶段零错配。独立代理严格按照执行文档验收CUDA推理、变化且有限的浮动底座物理状态、NVIDIA EGL非空画面。环境见证通过不等于动作任务通过。

## 场景、控制与限制

使用官方浮动底座G1、27身体和14手部关节。物理200Hz、策略50Hz，重力−9.81、地面摩擦1，关节默认值/增益/力矩限幅/armature来自真实Isaac配置；显式受限PD。MuJoCo3.2.3原生CPU物理，GPU0做CUDA策略与NVIDIA EGL渲染，不称为全GPU物理。

坐垫、桌面、平台采用训练配置的实际cuboid尺寸和位置；Pick/Bimanual物体是可自由运动、可碰撞的刚体，质量为seed0启动随机化的真实采样值，所有试验固定此值。手指动作依据原二值开闭命令，不焊接物体、不脚本带动物体；机器人自碰撞关闭，机器人—地面／桌面／物体碰撞启用。场景参数、模型哈希及原检查点/代码哈希在结构化结果中。

尚未逐项证明USD/MJCF惯量和机器人碰撞网格一致；Isaac隐式驱动与MuJoCo显式PD不同；使用训练几何而非照片级play场景资产；仅一个启动物体质量，没有覆盖完整随机化分布。接触／时序附加检查仍是有限代理，不等于实机、安全部署或完整论文复现。所有负结果保留，不能用接口通过掩盖任务迁移失败。

## 保存位置

- [MuJoCo视频与结果](../media/sim2sim_completed_20261002/results.html)
- `results/sim2sim_completed_20261002/summary.json`：四项逐参考结果、视频FFprobe/SHA256、原策略和场景哈希、接口与参考审计。
- `scripts/r3_sim2sim_completed_{export,mujoco,queue,reference_audit,contact_audit,archive}.py`；执行文档`docs/completed_tasks_sim2sim_20261002.md`。
- 服务器`artifacts/monitoring/r3_sim2sim_completed_20261002*`保留所有日志/状态；首次导出因当前ObservationManager接口不同而失败，原记录保留，已审计修正导出API后一次完成。不是正式MuJoCo试验失败，不记0分。
- 原始参考张量、快照、上游资产和模型仅留服务器；本地及私有GitHub归档指标、报告、脚本和视频。
'''
(ROOT/'reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md').write_text(report,encoding='utf-8')
print('COMPLETED_SIM2SIM_ARCHIVE',len(allrows),sum(x['num_references'] for x in allrows),'videos',len(videos),flush=True)
