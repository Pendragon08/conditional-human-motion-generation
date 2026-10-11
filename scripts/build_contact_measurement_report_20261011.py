"""Render source-grounded diagnostic status; no paper or Overleaf writes."""
from pathlib import Path
import hashlib
import html
import json

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
probe=read('results/condition_feedback_probe_v2_20261011/probe_summary.json')
inputs=[ROOT/name for name in ('results/contact_sampling_v8_20261011/sampling_summary.json',
 'results/contact_force_fixture_20261011/repeat_comparison.json',
 'results/contact_sensor_v10_20261011/sampling_summary.json',
 'results/contact_force_fixture_v2_20261011/repeat_comparison.json',
 'results/condition_feedback_probe_v2_20261011/probe_summary.json') if (ROOT/name).exists()]
status=dict(innovation_candidates=3,preliminarily_tested=1,not_yet_experimentally_tested=2,
 confirmed_innovations=0,measurement_gate='NOT_PASSED',g1='NOT_STARTED_BY_GATE',
 downstream='NOT_STARTED_BY_GATE',overleaf_results_sync=False,
 v8=v8,box=box,v10=v10,box_v2=box_v2,
 input_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
(OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf8')

rows=[('V6','保留错误调用','队列实际调用了V5；不能作为完成等待机制的验证'),
 ('V7','重置一致性失败','等待放在重置末尾时，导出刚体状态与官方重置缓存不一致；未启动轨迹采集'),
 ('V8','记录完整，接触测量不通过',f"两次各12输入×120帧，运动学完全一致；净接触力最大差{v8['contact_force']:.2f}N" if v8 else '结果未同步'),
 ('解析几何测试','仅几何计算通过','5个已知几何测试，不是人体物理执行成功证明'),
 ('物理箱体测试','两次校准通过，接口重复性有差异',f"状态与约束传感器完全一致；净接触力最大差{box['repeat_maximum_differences']['net_contact_force']:.2f}N" if box else '结果未同步'),
 ('箱体V2缓冲区清空','未解决重复性',f"状态、传感器不变，净接触力仍差{box_v2['repeat_maximum_differences']['net_contact_force']:.2f}N；修复假设不获支持" if box_v2 else '结果未同步'),
 ('V9','保留首次失败','适配器字符串断言失败，仿真和传感器校验没有执行'),
 ('V10','人体被动传感器记录复核',('传感器重复性：'+str(v10.get('sensor_repeatable_at_1mN'))+'；不等于真实接触力') if v10 else '尚未得到完整两次复核结果')]
table=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in row)+'</tr>' for row in rows)
p=probe['summary'] if probe else {}
probe_table=''.join(f'<tr><td>{html.escape(name)}</td><td>{row["attempts"]}</td><td>{row["endpoint_error_m"]*100:.2f}</td><td>{row["condition_pass"]}</td><td>{row["physical_pass"]}</td><td>{row["joint_pass"]}</td></tr>' for name,row in p.items())
document=f'''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>条件动作生成：接触测量复核与候选机制</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.75 system-ui;color:#172635}}h1,h2{{color:#153c5a}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #cbd5df;padding:10px;text-align:left}}th{{background:#eaf1f7}}.notice{{padding:16px;background:#fff2d8;border-left:5px solid #b67500}}a{{color:#126496}}@media print{{body{{max-width:none;margin:0}}tr{{break-inside:avoid}}}}</style>
<h1>条件动作生成：测量复核与候选机制</h1><p>2026-10-12 · 冻结生成模型和人体跟踪器 · 仅仿真 · 实验结果暂不写入Overleaf</p>
<div class="notice"><strong>3个候选、1个已初测、2个未测试、0个确认创新。</strong>完整接触测量门尚未通过。程序修复、传感器记录、文档执行见证均不作为论文创新。</div>
<h2>本轮测量证据</h2><table><tr><th>试验</th><th>状态</th><th>实际证据与边界</th></tr>{table}</table>
<p>简单物体测试采用5公斤箱体、平面、已知重力和无摩擦滑动。稳定支撑力接近49.05N，初始腾空接触力接近零。约束力传感器重复一致，但关节系统中的传感器还可能包含内部约束反力，因此不能将箱体结果推广为人体接触测量通过。</p>
<h2>C1的既有初步结果</h2><table><tr><th>方法</th><th>输入数</th><th>平均目标误差(cm)</th><th>条件合格</th><th>物理代理合格</th><th>联合合格</th></tr>{probe_table}</table>
<p>仅4个动作来源、3种偏移、1个生成种子，每臂最多3候选。条件改善是探索性信号；物理代理尚不可靠，联合结果不能证明方法有效或普遍无效。原始目标、失败和阈值均保留。并非原计划432输入的正式验证。</p>
<h2>三个候选机制</h2><ol><li><strong>C1：有界条件误差补偿。</strong>按实际执行误差有限调整下一轮生成条件。已初测，尚未通过正式效力及新颖性门。</li><li><strong>C2：可靠反馈与质量守护。</strong>估计反馈可靠性，避免不可靠分量驱动更新，拒绝使条件或物理质量恶化的候选。尚未实验。</li><li><strong>C3：按修复价值分配预算。</strong>区分参考缺陷与控制器困难，按单位成本的预计合格增益与来源覆盖分配候选。尚未实验。</li></ol>
<p>物理精化、执行反馈、闭环数据扩增已有前人路线。以上具体机制还需最近邻方法核查和等预算消融；不能直接写成论文贡献。</p>
<h2>进入正式验证的条件</h2><p>接触和时间对齐测量校验通过后，补齐独立来源、3生成种子和缺失消融。联合合格率相对最强等预算基线至少提高10个百分点，来源聚类95%置信区间排除零，并满足语义与多样性护栏，才能进入G1验证。下游收益还需相同训练预算、3训练种子与独立测试集。</p>
<h2>记录与复现</h2><ul><li><a href="../contact_force_fixture_20261011/repeat_comparison.json">物理箱体重复比较</a> · <a href="../contact_force_fixture_20261011/DOCUMENT_WITNESS.md">独立文档见证</a></li><li><a href="../contact_sampling_v8_20261011/sampling_summary.json">V8结果</a> · <a href="../contact_sampling_v8_20261011/EXPERIMENT_AUDIT.md">V8独立审计</a></li><li><a href="../../docs/candidate_mechanisms_20261011.md">候选机制与验证边界</a></li><li><a href="status.json">当前状态及证据哈希</a> · <a href="https://github.com/Pendragon08/conditional-human-motion-generation">代码与实验记录仓库</a></li></ul></html>'''
(OUT/'measurement_review.html').write_text(document,encoding='utf8')
print(OUT/'measurement_review.html')
