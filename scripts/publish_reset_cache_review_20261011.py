"""Small public numerical audit package; never export licensed motion arrays."""
import csv
from collections import Counter
import hashlib
import html
import json
import numpy as np
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'results/reset_alignment_v4_20261011'
FAILED_PROBE = ROOT / 'results/condition_feedback_probe_20261011'
PROBE = ROOT / 'results/condition_feedback_probe_v2_20261011'
CONTACT = ROOT / 'results/contact_sampling_v5_20261011'


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def main():
    measurement = json.loads((FIX / 'measurement_repair_summary.json').read_text())
    thresholds = json.loads((ROOT / 'results/physics_guided_20261011/frozen_thresholds.json').read_text())['thresholds']
    split_counts = {}
    for name, manifest_path in (('full_custom_validation', ROOT/'results/physics_guided_20261011/source_manifest.json'), ('small_custom_validation', FIX/'measurement_manifest.json')):
        validation_rows = [row for row in json.loads(manifest_path.read_text())['rows'] if row['split'] == 'val']
        split_counts[name] = dict(Counter(row['official_split'] for row in validation_rows))
    save(FIX/'dataset_scope.json', dict(counts=split_counts, classification='custom validation diagnostic, not official held-out efficacy', pretrained_training_overlap_present=True))
    gate_breakdown = {}
    for mode in ('planned', 'native'):
        rows = json.loads((FIX / ('full_val_' + mode + '_s11') / 'metrics.json').read_text())
        gate_breakdown[mode] = {}
        for key, bound in (('surface_slip_proxy_fraction', thresholds['sliding_fraction']), ('penetration_fraction', thresholds['penetration_fraction']), ('jerk_p95_m_s3', thresholds['continuity_jerk_p95_m_s3'])):
            values = np.array([row[key] for row in rows if row.get(key) is not None])
            gate_breakdown[mode][key] = dict(attempts=len(rows), failed=sum(row.get(key) is None or row[key] > bound for row in rows), frozen_bound=bound, minimum=float(values.min()), median=float(np.median(values)), maximum=float(values.max()))
    save(FIX / 'physical_gate_diagnosis.json', dict(status='DIAGNOSTIC_NOT_THRESHOLD_CALIBRATION', modes=gate_breakdown, thresholds_changed=False, proxy_contact_point_truth_verified=False))
    records = []
    public_files = []
    for directory in (FIX, FAILED_PROBE, PROBE, CONTACT):
        if not directory.exists():
            continue
        public = directory / 'public'
        public.mkdir(exist_ok=True)
        batches = sorted(directory.glob('*/receipt.json')) if directory == FIX else sorted((directory / 'raw').glob('*/receipt.json'))
        attempt_rows = []
        for receipt_path in batches:
            path = receipt_path.parent
            receipt = json.loads(receipt_path.read_text())
            meta = json.loads((path / 'metadata.json').read_text())
            metrics = json.loads((path / 'metrics.json').read_text())
            record = dict(label=path.name, directory=directory.name, attempts=len(metrics),
                completed=sum(row['completed'] for row in metrics), condition_pass=sum(row['condition_pass'] for row in metrics),
                physical_pass=sum(row['physical_pass'] for row in metrics), tracking_pass=sum(row['tracking_pass'] for row in metrics),
                joint_pass=sum(row['joint_pass'] for row in metrics), reset_gate=meta['initial_reset_gate_pass'],
                initial_state_difference=meta['manual_reset_vs_first_action_state_max_difference'],
                initial_xy_max_m=meta['initial_actual_reference_xy_max_m'], mode=meta['mode'], seed=meta['seed'],
                round=meta['round'], original_goal_sha256=meta['original_goal_sha256'],
                generation_calls=meta['generation_batch_calls'], seconds=receipt['elapsed_seconds'],
                outputs={name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in
                         ('metadata.json', 'metrics.json', 'receipt.json', 'lineage.json', 'initial_states.npz', 'trajectories.npz')})
            if directory == FIX:
                records.append(record)
            numeric_path = path / 'metric_numerics.json'
            if numeric_path.exists():
                record['metric_numeric_recomputation'] = json.loads(numeric_path.read_text())
            save(public / (path.name + '_evidence.json'), record)
            for row in metrics:
                # Source identities are salted only by a fixed cryptographic hash; not scores.
                entry = {key: value for key, value in row.items() if key not in ('source_id', 'original_goal', 'endpoint_error_vector')}
                entry.update(batch=path.name, source_sha256=hashlib.sha256(row['source_id'].encode()).hexdigest())
                attempt_rows.append(entry)
        if attempt_rows:
            columns = sorted(set().union(*(row.keys() for row in attempt_rows)))
            with (public / 'all_attempts.csv').open('w', newline='', encoding='utf8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
                writer.writeheader()
                writer.writerows(attempt_rows)
        summary_files = ('state.json', 'budget_ledger.json', 'measurement_repair_summary.json', 'versions_before_execution.json') if directory == FIX else (
            'state.json', 'budget_ledger.json', 'probe_summary.json', 'versions.json', 'frozen_thresholds.json', 'arms.json', 'sampling_summary.json', 'repeatability_summary.json', 'analysis_provenance_posthoc.json', 'EXPERIMENT_AUDIT.json')
        public_files.extend(path for path in directory.rglob('*.json') if path.parent == public or path.parent.name == 'requests')
        public_files += [public / 'all_attempts.csv'] if attempt_rows else []
        public_files.extend(directory / name for name in summary_files if (directory / name).exists())
        public_files.extend(directory.glob('*_execution_request.json'))
    quality_rows = [[row['label'], row['attempts'], row['completed'], row['condition_pass'], row['physical_pass'],
                     row['tracking_pass'], row['joint_pass'], row['initial_xy_max_m']] for row in records]
    if (PROBE / 'arms.json').exists() and (PROBE / 'probe_summary.json').exists():
        ledger = {row['label']:row for row in json.loads((PROBE/'budget_ledger.json').read_text())}
        costs = {}
        for arm in ('native_best', 'proposed', 'no_feedback', 'shuffled'):
            labels = ['val_native_s11_r0'] + (['val_native_s11_r1', 'val_native_s11_r2'] if arm == 'native_best' else ['val_%s_s11_r%d' % (arm,i) for i in (1,2)]) + ['val_eval_%s_s11' % arm]
            metas = [json.loads((PROBE/'raw'/label/'metadata.json').read_text()) for label in labels]
            scores = json.loads((PROBE/'raw'/labels[-1]/'metrics.json').read_text())
            complete_selections = json.loads((PROBE/'selected'/(arm+'_s11.json')).read_text())['choices']
            qualified = sum(row['joint_pass'] and choice['construction_complete'] for row,choice in zip(scores,complete_selections))
            seconds = sum(ledger[label]['seconds'] for label in labels)
            costs[arm] = dict(labels=labels, input_count=12, candidate_count_per_input=3,
                equivalent_standalone_seconds=seconds, generator_batch_calls=sum(meta['generation_batch_calls'] for meta in metas),
                construction_first_attempts=36, selected_replay_first_attempts=12,
                qualified_selected_references=qualified, seconds_per_qualified_reference=seconds/qualified if qualified else None,
                sharing='First native batch is charged once in total ledger and once per arm in standalone-equivalent cost.')
        save(PROBE/'cost_by_arm.json', costs)
        public_files.append(PROBE/'cost_by_arm.json')
    probe_path = PROBE / 'probe_summary.json'
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    content = '<p>条件纠偏诊断尚未完成，不报告尚未产生的比较结果。</p>'
    if probe:
        content = table(['方法', '独立回放数', '条件合格', '物理代理合格', '联合合格', '平均终点误差(m)', '平均滑移代理比例'], [
            [name, value['attempts'], value['condition_pass'], value['physical_pass'], value['joint_pass'],
             round(value['endpoint_error_m'], 4), round(value['surface_slip_proxy_fraction'], 4)] for name, value in probe['summary'].items()])
        content += '<p>每方法3个候选，均经过独立参考回放。去除反馈仍实际消耗3次生成与仿真；打乱反馈使用其他原始来源。'
        content += '原始目标、文本、时刻、权重及阈值固定。仅4个验证来源×3个偏移、种子11，是诊断，不是正式三种子测试。这4个来源包含2个HumanML3D官方train来源与2个官方test来源，不能作为独立held-out效能证据。</p>'
        content += table(['对照', '联合率差(百分点)', '来源聚类95%CI', '平均终点误差减少(m)'], [
            [name, round(value['joint_gain_pp'], 3), value['source_cluster_ci95_pp'], round(value['endpoint_error_reduction_m'], 4)]
            for name, value in probe['differences'].items()])
        content += '<p><strong>创新结论：</strong>' + html.escape(probe['innovation_status']) + '。不能将次要条件指标改善替代联合指标；'
        content += '物理代理的鉴别能力、语义与多样性仍未完成独立确认，不宣称机制普遍无效或论文创新成立。</p>'
    contact_note = ''
    if (CONTACT/'repeatability_summary.json').exists():
        repeat = json.loads((CONTACT/'repeatability_summary.json').read_text())
        contact_note = '<h2 id="contact">接触测量复核：尚未修复</h2>' + table(['采样方式','运动学最大差','接触力最大差(N)','支撑标签Jaccard','脚滑比例最大差'], [[name, value['deltas']['actual'], round(value['deltas']['contact_force'], 2), round(value['support_jaccard'],4), round(value['max_slip_fraction_difference'],4)] for name,value in repeat.items() if isinstance(value,dict)])
        contact_note += '<p>同输入、同模型、同噪声、同实际前缀的两次完整运行均保留120帧。V5仅在复制张量前增加GPU生产者同步；仍未消除接触力和支撑标签差异。该试验排除了简单同步操作作为完整修复，未定位PhysX隐藏历史/接触采样差异的确切原因。</p><div class="box"><strong>测量校验尚未整体通过。</strong>导出重置缓存一致性通过，不等于完整PhysX状态或接触测量可靠。主要联合指标的物理代理不具备当前比较所需的有效性；停止正式测试、G1和下游训练。目标误差改善仅作探索记录，不能作为创新成立或机制普遍无效的证据。</div>'
    doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重置状态修复与条件反馈诊断</title><style>body{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#203247}h1{font-size:29px}h2{margin-top:36px}a{color:#156594}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #c7d5df;padding:8px;text-align:left}th{background:#edf4f8}.box{background:#fff4d7;padding:16px;border-left:5px solid #ac7400}pre{white-space:pre-wrap;font-size:13px}@media print{body{max-width:none;margin:0}tr{break-inside:avoid}}</style></head><body>
<h1>重置状态修复与条件反馈诊断</h1><p>2026-10-11 · 冻结生成器/跟踪器 · 人体物理仿真 · 本轮不更新Overleaf实验结果</p>
<nav><a href="#reset">重置修复</a> · <a href="#measurement">216次校验</a> · <a href="#probe">创新候选诊断</a> · <a href="#contact">接触测量复核</a> · <a href="#limits">证据边界</a></nav>
<div class="box">216次导出重置缓存/坐标检查完成；接触重复性校验失败，完整测量门尚未通过。四臂诊断全部联合零合格。不能将工程修复当作生成创新。</div>
<h2 id="reset">修复了什么</h2><p>上游重置将关节状态写入仿真，并恢复一次刚体状态缓存。旧收集器在重置返回后再次刷新，读回了前一次仿真的刚体快照。
本次仅移除这次多余刷新，保留上游重置、物理步骤、策略和生成权重。新增初始状态数组保存与首次动作前逐项检查；未用模型输出伪造实际执行状态。</p>
<p>七组导出张量包括根节点、关节位置/速度、刚体位置/四元数/线速度/角速度。修复后，bootstrap重置和首次动作前的七组张量均逐元素一致，四组刚体缓存也一致。
这不等于验证了PhysX内部隐藏状态、接触求解器全部状态或真实机器人。</p>
<h2 id="measurement">测量校验结果与未通过项</h2><p>六个12输入批次覆盖生成种子11/12/13与计划/实际前缀；另两个72输入批次覆盖全部24个验证来源，生成种子11。
合计216个首次尝试，未删除提前终止。坐标逆变换、参考帧时钟、三种世界目标偏移、来源分组和输入/输出绑定保留。
未改原始目标和成功阈值，不与历史失效测试混合。这里的验证为自定义划分：24来源中19个来自HumanML3D官方train，5个来自官方test，不能作独立held-out效能测试。完整验证的两种模式均72/72不满足脚滑代理比例界限；连续性代理分别41/72和48/72不满足界限，穿透代理均0/72失败。该拆解不用于放宽判据，不能替代真实接触点校准。</p>''' + table(
        ['批次', '尝试', '完成', '条件合格', '物理代理合格', '跟踪合格', '联合合格', '初始XY误差(m)'], quality_rows) + '''
<h2 id="probe">条件误差反馈：等预算对比与消融</h2><p>V1第二批仿真完成后，浮点重算出现4.44e-16弧度差异，被旧逐字节检查拒绝。V1全部记录保留；V2在新目录重跑，仅允许浮点舍入误差，分类、阈值与分数不变。</p>''' + content + '''
''' + contact_note + '''
<h2 id="limits">证据边界与下一阶段</h2><ul><li>足部滑移是刚体接触力与碰撞表面几何的代理，不是实测接触点真值；腾空不直接计为漂浮。</li>
<li>216例只校验初始化与接口。完整验证集的物理/联合零合格率及接触重复性失败仍是测量瓶颈，不等于创新机制已经被证明无效。</li>
<li>动作生成模型及控制器均冻结，未增加机器人IK、音乐条件或接触力生成。</li><li>未进入原计划432输入/方法的正式测试；未运行G1迁移、下游训练或实机部署。</li>
<li>当前重置修复是工程正确性修复；生成贡献仍需独立来源、三种子、语义/多样性护栏及下游训练支持。</li>
<li>旧失败全部保留；本轮不清理旧实验或其他项目。本地及GitHub同步数值与代码，许可动作数组保留服务器。</li></ul>
<p><a href="public/all_attempts.csv">216次校验逐次指标</a> · <a href="measurement_repair_summary.json">完整校验结论</a> ·
<a href="../condition_feedback_probe_v2_20261011/probe_summary.json">反馈诊断结论</a> · <a href="../condition_feedback_probe_v2_20261011/public/all_attempts.csv">全部诊断尝试</a> ·
<a href="budget_ledger.json">校验预算</a> · <a href="../condition_feedback_probe_v2_20261011/budget_ledger.json">诊断预算</a> ·
<a href="EXPERIMENT_AUDIT.md">重置审计</a> · <a href="../condition_feedback_probe_v2_20261011/EXPERIMENT_AUDIT.md">四臂诊断审计</a> ·
<a href="../contact_sampling_v5_20261011/EXPERIMENT_AUDIT.md">接触复核审计</a> · <a href="combined_budget.json">累计预算</a> ·
<a href="final_round_status.json">停止状态</a></p></body></html>'''
    report = FIX / 'measurement_report.html'
    report.write_text(doc, encoding='utf8')
    public_files.append(report)
    public_files.append(FIX / 'physical_gate_diagnosis.json')
    public_files.append(FIX / 'dataset_scope.json')
    budget_rows = [row for directory in ([ROOT / 'results/physics_guided_20261011'] + [ROOT / ('results/' + name) for name in ('reset_alignment_20261011', 'reset_alignment_v2_20261011', 'reset_alignment_v3_20261011')] + [FIX, FAILED_PROBE, PROBE, CONTACT]) if (directory / 'budget_ledger.json').exists() for row in json.loads((directory / 'budget_ledger.json').read_text())]
    save(FIX / 'combined_budget.json', dict(charged_gpu_hours=sum(row['seconds'] for row in budget_rows)/3600, includes_failed_attempts=True, records=budget_rows, hard_limit_gpu_hours=72))
    public_files.append(FIX / 'combined_budget.json')
    status = dict(status='STOP_CONTACT_MEASUREMENT_INVALID', exported_reset_cache_coordinate_attempts=216, construction_attempts=108, independent_selected_reference_replays=48, synchronized_contact_repeat_attempts=24, formal_efficacy_test_run=False, g1_run=False, downstream_training_run=False, overleaf_result_writes=False, novelty='not_established', conclusion='Immutable goal endpoint improvement is exploratory; force-gated physical validity is not supported. No threshold relaxation.')
    save(FIX/'final_round_status.json',status)
    public_files.append(FIX/'final_round_status.json')
    for directory in (FIX,PROBE,CONTACT):
        public_files.extend(directory/name for name in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','resource_at_completion.json') if (directory/name).exists())
    archive = ROOT / 'results/reset_cache_review_public_20261011.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(set(public_files)):
            bundle.add(path, arcname=str(path.relative_to(ROOT / 'results')))
    print('RESET_CACHE_PUBLIC_PACKAGE', archive, archive.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
