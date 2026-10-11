"""Render source-grounded diagnostic status; no paper or Overleaf writes."""
from pathlib import Path
import hashlib
import html
import json
from replay_report_certificate_v30_20261012 import certify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/contact_measurement_review_20261011'
OUT.mkdir(exist_ok=True)
def read(path):
    file=ROOT/path
    return json.loads(file.read_text(encoding='utf8')) if file.exists() else None

v8=read('results/contact_sampling_v8_20261011/sampling_summary.json')
box=read('results/contact_force_fixture_20261011/repeat_comparison.json')
v10=read('results/contact_sensor_v10_20261011/sampling_summary.json')
box_v2=read('results/contact_force_fixture_v2_20261011/repeat_comparison.json')
box_v3=read('results/contact_force_fixture_v3_20261012/repeat_comparison.json')
v11=read('results/contact_cpu_export_v11_20261012/sampling_summary.json')
quality11=read('results/contact_cpu_export_v11_20261012/quality_diagnostic.json')
validation12=read('results/contact_cpu_validation_v12_20261012/validation_summary.json')
slip13=read('results/contact_slip_diagnostic_v13_20261012/diagnostic.json')
point16=read('results/contact_point_fixture_v16_20261012/first_attempt/summary.json')
point17=read('results/contact_point_human_v17_20261012/state.json')
velocity19=read('results/velocity_reference_fixture_v19_20261012/first_attempt/summary.json')
point20=read('results/contact_point_human_v20_20261012/state.json')
com21=read('results/contact_com_metric_v21_20261012/diagnostic.json')
articulated22=read('results/articulated_contact_fixture_v22_20261012/first_attempt/summary.json')
validation24=read('results/contact_com_validation_v24_20261012/validation_summary.json')
numerics25=read('results/metric_runtime_diagnostic_v25_20261012/diagnostic.json')
runtime26=read('results/contact_com_runtime_v26_20261012/runtime_check.json')
replay27=read('results/fixed_reference_com_v27_20261012/replay_summary.json')
formal28=read('results/condition_feedback_formal_v28_20261012/queue_state.json')
progress_paths=sorted((ROOT/'results/condition_feedback_formal_v28_20261012').glob('progress_snapshot_*.json'))
formal_progress=read(str(progress_paths[-1].relative_to(ROOT))) if progress_paths else None
if formal_progress:
    assert formal_progress['source_sha256']==hashlib.sha256((ROOT/'scripts/snapshot_condition_feedback_v28_20261012.py').read_bytes()).hexdigest()
    assert formal_progress['final_gate'] is False and formal_progress['innovation_confirmed'] is False
component_paths=sorted((ROOT/'results/condition_feedback_components_v32_20261012').glob('diagnostic_*.json'))
components=read(str(component_paths[-1].relative_to(ROOT))) if component_paths else None
if components:
    assert components['source_sha256']==hashlib.sha256((ROOT/'scripts/diagnose_condition_feedback_components_v32_20261012.py').read_bytes()).hexdigest()
    assert components['live_pipeline_mutated'] is False and components['thresholds_changed'] is False
    assert components['simulation_calls']==0 and components['generation_calls']==0
    assert components['innovation_confirmed'] is False
replay_certificate=certify(ROOT,replay27)
cached_guard=read('results/fixed_reference_com_v27_20261012/cached_guard_recheck_v30_20261012.json')
cached_guard_pass=(cached_guard is not None and cached_guard.get('status')=='PASS' and len(cached_guard.get('entries',[]))==8 and all(row.get('pass') is True for row in cached_guard['entries']) and cached_guard.get('source_sha256')==hashlib.sha256((ROOT/'scripts/verify_cached_measurement_guards_v30_20261012.py').read_bytes()).hexdigest())
replay_interface_pass=(replay_certificate['status']=='PASS' and cached_guard_pass and runtime26 is not None and runtime26.get('status')=='PASS' and velocity19 is not None and velocity19.get('status')=='PASS' and articulated22 is not None and articulated22.get('status')=='PASS')
probe=read('results/condition_feedback_probe_v2_20261011/probe_summary.json')
inputs=[ROOT/name for name in ('results/contact_sampling_v8_20261011/sampling_summary.json',
 'results/contact_force_fixture_20261011/repeat_comparison.json',
 'results/contact_sensor_v10_20261011/sampling_summary.json',
 'results/contact_force_fixture_v2_20261011/repeat_comparison.json',
 'results/contact_force_fixture_v3_20261012/repeat_comparison.json',
 'results/contact_cpu_export_v11_20261012/sampling_summary.json',
 'results/contact_cpu_export_v11_20261012/quality_diagnostic.json',
 'results/contact_cpu_validation_v12_20261012/validation_summary.json',
 'results/contact_slip_diagnostic_v13_20261012/diagnostic.json',
 'results/contact_point_fixture_v16_20261012/first_attempt/summary.json',
 'results/contact_point_human_v17_20261012/state.json',
 'results/velocity_reference_fixture_v19_20261012/first_attempt/summary.json',
 'results/contact_point_human_v20_20261012/state.json',
 'results/contact_com_metric_v21_20261012/diagnostic.json',
 'results/articulated_contact_fixture_v22_20261012/first_attempt/summary.json',
 'results/contact_com_validation_v24_20261012/validation_summary.json',
 'results/metric_runtime_diagnostic_v25_20261012/diagnostic.json',
 'results/contact_com_runtime_v26_20261012/runtime_check.json',
 'results/fixed_reference_com_v27_20261012/replay_summary.json',
 'results/fixed_reference_com_v27_20261012/cached_guard_recheck_v30_20261012.json',
 'results/condition_feedback_formal_v28_20261012/queue_state.json',
 'results/condition_feedback_probe_v2_20261011/probe_summary.json') if (ROOT/name).exists()]
if component_paths: inputs.append(component_paths[-1])
status=dict(innovation_candidates=3,preliminarily_tested=1,not_yet_experimentally_tested=2,
 confirmed_innovations=0,measurement_gate='CALIBRATED_PROXY_AND_FIXED_REPLAY_PASS' if replay_interface_pass else 'FIXED_REPLAY_NOT_CERTIFIED',g1='NOT_STARTED_BY_GATE',
 replay_export_certificate=replay_certificate,
 cached_guard_recheck=cached_guard,
 formal_comparison_snapshot=formal28,
 formal_completed_arm_snapshot=formal_progress,
 failure_component_diagnostic=components,
 downstream='NOT_STARTED_BY_GATE',overleaf_results_sync=False,
 slip13=slip13,point16=point16,point17=point17,velocity19=velocity19,point20=point20,com21=com21,articulated22=articulated22,validation24=validation24,numerics25=numerics25,runtime26=runtime26,replay27=replay27,
 v8=v8,box=box,v10=v10,box_v2=box_v2,box_v3=box_v3,v11=v11,quality11=quality11,validation12=validation12,
 input_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
(OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf8')

rows=[('V11 CPU导出人体重复','新设置重复性通过，质量尚不通过',f"重复运动、接触力和动作均一致；与旧设置实际位置最大差{v11['cpu_export_vs_v10_state_max']['actual']:.3f}m，不混合旧结果。12输入脚滑代理通过0。" if v11 else '未同步'),
 ('V6','保留错误调用','队列实际调用了V5；不能作为完成等待机制的验证'),
 ('V7','重置一致性失败','等待放在重置末尾时，导出刚体状态与官方重置缓存不一致；未启动轨迹采集'),
 ('V8','记录完整，接触测量不通过',f"两次各12输入×120帧，运动学完全一致；净接触力最大差{v8['contact_force']:.2f}N" if v8 else '结果未同步'),
 ('解析几何测试','仅几何计算通过','5个已知几何测试，不是人体物理执行成功证明'),
 ('物理箱体测试','两次校准通过，接口重复性有差异',f"状态与约束传感器完全一致；净接触力最大差{box['repeat_maximum_differences']['net_contact_force']:.2f}N" if box else '结果未同步'),
 ('箱体V2缓冲区清空','未解决重复性',f"状态、传感器不变，净接触力仍差{box_v2['repeat_maximum_differences']['net_contact_force']:.2f}N；修复假设不获支持" if box_v2 else '结果未同步'),
 ('箱体V3 CPU导出','箱体校验和重复性通过',f"状态和传感器相对原版本不变，重复净接触力差{box_v3['repeat_maximum_differences']['net_contact_force']:.6f}N；尚未验证人体环境" if box_v3 else '结果未同步'),
 ('V9','保留首次失败','适配器字符串断言失败，仿真和传感器校验没有执行'),
 ('V10','人体被动传感器记录复核',('传感器重复性：'+str(v10.get('sensor_repeatable_at_1mN'))+'；不等于真实接触力') if v10 else '尚未得到完整两次复核结果')]
rows.extend([
 ('V13 432次离线几何诊断','不支持最低角点是主要原因','原指标精确重算；诊断性选择附近最慢角点也仅1/432过线，不替代评分。'),
 ('V14/V15 接触接口适配','保留两次失败','V14变量覆盖；V15字段命名不匹配。失败并不证明API不可用。'),
 ('V16 接触接口物体校准','已通过' if point16 else '未同步','3312条原生接触、0错误；已知箱体状态/力/传感器与V3完全一致。'),
 ('V17 人体接触记录','采集完成，校验程序失败' if point17 else '未同步','新增绑定键与旧校验器不兼容；独立核查53764条接触0错误、所有状态/动作与V11一致。'),
 ('V19 速度参考位置','独立校准通过' if velocity19 else '未同步','线速度对应质心；6cm偏置、3rad/s旋转时，原点解释误差0.18006m/s，质心解释误差0.0000703m/s。'),
 ('V20 人体记录接口修复',point20['status'] if point20 else '未同步','保留三键绑定；密集接触文件另由元信息哈希绑定；原阈值与物理不变。'),
 ('V21 质心速度修正','离线计算修正完成' if com21 else '未同步','修正旋转杠杆臂；12输入仍脚滑代理0/12、联合0/12。旧评分保留，不算创新。'),
 ('V22 已知关节体支撑/滑动/腾空',articulated22['status'] if articulated22 else '待复核','使用原判据验证正确静止、明确滑动和腾空，结果不能推广为所有滚动接触或真实力。'),
 ('V25 数值环境差异','原因确认' if numerics25 else '待同步','保存时使用NumPy 1.23.5，旧复算使用1.24.4；统一到采集环境后，72条指标差异为0，分类不变。'),
 ('V26 复算环境修复',runtime26['status'] if runtime26 else '待同步','独立见证保留旧失败及11个输出绑定，复用已完成72条，无新仿真、阈值或容差修改。'),
 ('V27 固定参考回放',replay27['status'] if replay27 else '待完成','回放相同输入的原生成参考及物理轨迹参考，所有来源与构建失败保留；非候选效力实验。')])
table=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in row)+'</tr>' for row in rows)
p=probe['summary'] if probe else {}
probe_table=''.join(f'<tr><td>{html.escape(name)}</td><td>{row["attempts"]}</td><td>{row["endpoint_error_m"]*100:.2f}</td><td>{row["condition_pass"]}</td><td>{row["physical_pass"]}</td><td>{row["joint_pass"]}</td></tr>' for name,row in p.items())
validation_text='质心速度修正后的全24验证来源、3生成种子、两种前缀验证尚未完成。'
if validation24:
    validation_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[key]))+'</td>' for key in ('mode','seed','inputs','completed','condition_pass','physical_pass','joint_pass'))+'</tr>' for row in validation24['entries'])
    validation_text='<p>质心速度修正后已完成432次验证尝试：24来源×3偏移×3种子×2前缀。它是验证集诊断，不能称为每方法432测试输入的正式实验。原目标、阈值不变，失败保留。</p><table><tr><th>前缀</th><th>种子</th><th>尝试</th><th>完成</th><th>条件合格</th><th>物理代理合格</th><th>联合合格</th></tr>'+validation_rows+'</table><p>阶段决定：'+html.escape(validation24['decision'])+'。物理质量未通过与测量程序未通过是不同问题；此表不证明候选机制无效，更不证明创新有效。</p>'
replay_text='<p>固定参考回放尚未完成。</p>'
if replay27:
    replay_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[key]))+'</td>' for key in ('reference','attempted','completed','condition_pass','physical_pass','tracking_pass','combined_construction_replay_joint_pass','construction_terminated'))+'</tr>' for row in replay27['entries'])
    replay_text='<table><tr><th>固定参考</th><th>尝试</th><th>完成</th><th>条件合格</th><th>物理代理合格</th><th>跟踪合格</th><th>构建＋回放联合合格</th><th>构建终止</th></tr>'+replay_rows+'</table><p>所有72输入都保留。构建终止后补齐的序列不能因回放成功而恢复联合成功资格。这是单种子接口与质量诊断，不是新算法对比。</p>'
measurement_text='已知正负对照通过，固定参考回放尚待完成。'
formal_text='<p>C1完整对比尚未启动。</p>'
if formal28:
    formal_text='<p>服务器采集状态快照：'+html.escape(formal28['status'])+'；批次：'+html.escape(formal28.get('label','未提供'))+'。完整协议包含10个对比与消融配置，每配置432测试输入。状态快照不代替实时进程检查，也不是完成结果；尚无新的效力结论。</p>'
if formal_progress:
    snapshot_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in
        ('arm','seed','attempts','condition_pass','physical_pass','joint_pass','early_terminated'))+'</tr>'
        for row in formal_progress['entries'])
    formal_text='<p>已完成回放批次：'+str(formal_progress['complete_replay_arm_seed_batches'])+'/'+str(formal_progress['expected_replay_arm_seed_batches'])+'。快照采集时已核实原进程存在，正在运行 '+html.escape(formal_progress['queue_state'].get('label','未提供'))+'。这是带时间戳的部分结果，不表示打开报告时仍在该批次，也不是正式结论。</p><table><tr><th>方法</th><th>种子</th><th>输入</th><th>条件合格</th><th>物理代理合格</th><th>联合合格</th><th>提前终止</th></tr>'+snapshot_rows+'</table><p>当前已完成的第一个种子中，误差补偿的条件合格为34/144，native_best为24/144；但二者联合合格均为0/144。因此条件改善不能当作物理质量或训练价值改善。其余种子尚待完成；所有判据保持不变。</p>'
component_text='<p>完整来源失败分量尚未同步。</p>'
if components:
    selected_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (row['arm'],row['seed'],row['attempts'],row['component_pass_counts']['condition'],
         row['component_pass_counts']['physical'],row['complete_construction_and_replay_physical_pass'],
         row['physical_pass_but_construction_failed'],row['component_pass_counts']['joint']))+'</tr>'
        for row in components['entries'] if row['arm'] in ('native_best','proposed','planned_prefix'))
    component_text='<p>只读核对了'+str(components['completed_arm_seed_batches'])+'/30个完成回放批次和'+str(len(components['construction_batches']))+'个候选构建批次；各批次均绑定输入、选择记录及轨迹哈希，不改阈值或运行中的队列。</p><table><tr><th>方法</th><th>种子</th><th>输入</th><th>条件合格</th><th>回放物理代理合格</th><th>构建与回放均完成且物理合格</th><th>物理合格但构建失败</th><th>联合合格</th></tr>'+selected_rows+'</table><p>物理代理合格数可能包含构建失败后补齐的参考，不能把这类样本计为有效生成。首个种子的三个物理代理合格案例均属于构建失败；第二个种子的计划前缀消融出现一个构建与回放均完成的物理合格案例，但未同时满足全部条件与跟踪判据。</p><p>前两个种子中，误差补偿的条件合格数分别为34/144和33/144，native_best分别为24/144和23/144；计划前缀消融分别为33/144和33/144。所有观察到的联合合格数仍为零。因此仅有终点改善不足以支持创新，且实际执行前缀的额外作用目前没有得到支持。第三个种子与最终统计尚待完成。</p><p>另已准备原始MoCap参考直接回放对照，用于检查跟踪器、表示转换与滑移代理是否也对真实参考产生质量下限；该对照尚未运行，不作为第四个创新点。</p>'
if replay_interface_pass:
    measurement_text='重置、坐标、时间、数值复算、已知正负对照及固定参考回放接口已通过。足部指标仍是碰撞表面滑移代理，不能称为真实接触力或所有滚动接触的真值。人体物理质量与候选效力另行判断。'
document=f'''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>条件动作生成：接触测量复核与候选机制</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.75 system-ui;color:#172635}}h1,h2{{color:#153c5a}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #cbd5df;padding:10px;text-align:left}}th{{background:#eaf1f7}}.notice{{padding:16px;background:#fff2d8;border-left:5px solid #b67500}}a{{color:#126496}}@media print{{body{{max-width:none;margin:0}}tr{{break-inside:avoid}}}}</style>
<h1>条件动作生成：测量复核与候选机制</h1><p>2026-10-12 · 冻结生成模型和人体跟踪器 · 仅仿真 · 实验结果暂不写入Overleaf</p>
<div class="notice"><strong>3个候选、1个已初测、2个未测试、0个确认创新。</strong>{measurement_text}程序修复、传感器记录、文档执行见证均不作为论文创新。</div>
<h2>C1完整对比进度</h2>{formal_text}<h2>失败分量与消融证据</h2>{component_text}<h2>本轮测量证据</h2><table><tr><th>试验</th><th>状态</th><th>实际证据与边界</th></tr>{table}</table>
<p>简单物体测试采用5公斤箱体、平面、已知重力和无摩擦滑动。稳定支撑力接近49.05N，初始腾空接触力接近零。约束力传感器重复一致，但关节系统中的传感器还可能包含内部约束反力，因此不能将箱体结果推广为人体接触测量通过。</p>
<p>V11只证明固定新设置内的重复性，不能将其解释为行为不变的修复。初始七项状态与旧设置一致，第一步微小数值差异随后由闭环放大。原阈值保持不变；小样本穿透12/12通过、连续性7/12通过、脚滑代理0/12通过，联合0/12。</p><h2>全来源验证诊断</h2>{validation_text}<h2>速度计算修正与验证边界</h2><p>V19确认原计算将质心线速度当作刚体原点线速度，脚底材料点速度需要扣除旋转后的质心偏移。已单独修正并完成已知静止/滑动/腾空箱体和旋转材料点校验。V21的12条人体输入仍未达到原物理判据，因此计算错误不能解释全部质量问题。历史V12的432次评分没有覆盖正确速度约定，暂不作为正式有效性结论；V24修正后的结果见上表。所有阈值与原始目标保持不变。</p><h2>固定参考回放</h2>{replay_text}<h2>C1的既有初步结果（旧导出设置）</h2><table><tr><th>方法</th><th>输入数</th><th>平均目标误差(cm)</th><th>条件合格</th><th>物理代理合格</th><th>联合合格</th></tr>{probe_table}</table>
<p>仅4个动作来源、3种偏移、1个生成种子，每臂最多3候选。条件改善是探索性信号；物理代理尚不可靠，联合结果不能证明方法有效或普遍无效。原始目标、失败和阈值均保留。并非原计划432输入的正式验证。</p>
<h2>三个候选机制</h2><ol><li><strong>C1：有界条件误差补偿。</strong>按实际执行误差有限调整下一轮生成条件。已初测，正式对比正在运行，尚未通过效力及新颖性门。</li><li><strong>C2：可靠反馈与质量守护。</strong>估计反馈可靠性，避免不可靠分量驱动更新，拒绝使条件或物理质量恶化的候选。核心、仿真入口和配对分析已实现，数值检查通过；尚未完成独立运行见证或仿真效力试验。重复执行计入三次预算；若C1最终全部联合合格为零，先诊断，不自动运行。</li><li><strong>C3：按修复价值分配预算。</strong>区分参考缺陷与控制器困难，按单位成本的预计合格增益与来源覆盖分配候选。尚未实验。</li></ol>
<p>物理精化、执行反馈、闭环数据扩增已有前人路线。以上具体机制还需最近邻方法核查和等预算消融；不能直接写成论文贡献。</p>
<h2>进入正式验证的条件</h2><p>接触和时间对齐测量校验通过后，补齐独立来源、3生成种子和缺失消融。联合合格率相对最强等预算基线至少提高10个百分点，来源聚类95%置信区间排除零，并满足语义与多样性护栏，才能进入G1验证。下游收益还需相同训练预算、3训练种子与独立测试集。</p>
<h2>记录与复现</h2><ul><li><a href="../contact_force_fixture_20261011/repeat_comparison.json">物理箱体重复比较</a> · <a href="../contact_force_fixture_20261011/DOCUMENT_WITNESS.md">独立文档见证</a></li><li><a href="../contact_sampling_v8_20261011/sampling_summary.json">V8结果</a> · <a href="../contact_sampling_v8_20261011/EXPERIMENT_AUDIT.md">V8独立审计</a></li><li><a href="../../docs/candidate_mechanisms_20261011.md">候选机制与验证边界</a></li><li><a href="status.json">当前状态及证据哈希</a> · <a href="https://github.com/Pendragon08/conditional-human-motion-generation">代码与实验记录仓库</a></li></ul></html>'''
(OUT/'measurement_review.html').write_text(document,encoding='utf8')
print(OUT/'measurement_review.html')
