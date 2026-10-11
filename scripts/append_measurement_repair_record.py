"""Publish separate reset-repair evidence without pooling historical results."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def main(repo):
    old=repo/'results/physics_guided_20261011'
    repair=repo/'results/reset_alignment_v3_20261011'
    result=json.loads((repair/'measurement_repair_summary.json').read_text(encoding='utf8'))
    assert result['next_stage'].startswith('STOP_'), 'Continue valid experiments before publishing a stopped workflow'
    rows=result['full_validation_quality'] or result.get('available_fixture_quality',{})
    label={'full_val_planned_s11':'计划前缀（完整验证）','full_val_native_s11':'执行前缀（完整验证）',
           'planned_s11':'计划前缀（部分测量）'}
    table=''.join('<tr><td>'+label[key]+'</td>'+''.join('<td>'+str(row[name])+'</td>' for name in
        ('attempts','completed','condition_pass','physical_pass','tracking_pass','joint_pass'))+'</tr>' for key,row in rows.items())
    xy=max(r['initial_xy_max_m'] for r in result['initial_state_checks'])
    body_position=max(r['full_state_max_difference']['_rigid_body_pos'] for r in result['initial_state_checks'])
    goal=max((r['target_max_difference_m'] for r in result['cross_seed_and_mode_checks']),default=None)
    offset=max(r['max_absolute_offset_error_m'] for r in result['offset_checks'])
    coordinate='通过' if result['coordinate_gate_pass'] else '未通过'
    text=('独立初始化复核计划216例，实际启动%d例、完成%d例，测量校验%s。'
          '首动作前实际与参考根节点的最大XY分量误差为 %.8g 米，'
          '引导重置与首动作前的刚体位置最大分量差为 %.8g 米；其余导出状态逐项差异见摘要，不能将不同单位的差异合为一个误差指标。'
          '已运行来源的三个目标偏移最大误差为 %.8g 米。'
          %(result['attempts'],result.get('completed_cases',result['attempts']),coordinate,xy,body_position,offset))
    if goal is None:
        text+='跨种子、跨模式和完整验证未运行。外层进程在下一批前触发保护其他工作负载的资源断言，退出码为1；原始状态文件残留RUNNING，不代表仍在运行。'
    if result['coordinate_gate_pass']:
        text+='三种子目标、计划前缀与初态一致，生成噪声不同；请求与输出均有执行时哈希绑定。'
    else:
        text+='具体失败项见独立测量摘要，不据此开展机制有效性比较。'
    if result['joint_gate_floor']:
        text+='修复坐标后的全部验证来源中，两种前缀模式仍均为零联合合格。这表明当前验收与跟踪设置尚无可用的进入条件，不能据此认定反馈机制无效。'
    text+='本复核不改变质量阈值，不与旧轮合并，不包含候选反馈的完整对比或下游训练；G1及九组策略训练按停止条件未启动。'
    start='<!-- RESET_REPAIR_20261011_START -->'; end='<!-- RESET_REPAIR_20261011_END -->'
    document=(old/'experiment_report.html').read_text(encoding='utf8')
    if start in document:
        a=document.index(start); b=document.index(end)+len(end)
        document=document[:a]+document[b:]
    addition=start+'<section id="reset-repair"><h2>独立初始化修复与停止依据</h2><p>'+html.escape(text)+'</p>'
    addition+='<table><thead><tr><th>模式</th><th>尝试</th><th>完成</th><th>条件合格</th><th>物理代理合格</th><th>跟踪合格</th><th>联合合格</th></tr></thead><tbody>'+table+'</tbody></table>'
    addition+='<p>前两次初始化修复均在控制开始前失败，全部日志、请求和耗时保留。V3在原目标冻结前统一参考坐标，保留相对动作；XY锚点改善不等于完整初始状态校验通过。首批7项导出运动学状态中刚体位置、旋转及速度仍不同，后续批次没有启动。旧轮与三次复核保守合计 %.3f GPU小时。此范围不包括真实机器人或G1验证。</p>'%result['combined_charged_gpu_hours']
    addition+='<p><a href="../reset_alignment_v3_20261011/measurement_repair_summary.json">独立测量摘要</a> · <a href="../reset_alignment_v3_20261011/EXPERIMENT_AUDIT.md">独立审计</a></p></section>'+end
    document=document.replace('</body>',addition+'\n</body>')
    (old/'experiment_report.html').write_text(document,encoding='utf8')
    tex=(old/'overleaf_experiment_record.tex').read_text(encoding='utf8')
    marker='% RESET_REPAIR_RECORD_20261011'
    if marker in tex: tex=tex.split(marker)[0]
    tex+='\n'+marker+'\n'+r'\subsection{独立初始化修复与停止依据}'+'\n'+text+'\n'
    tex+=r'\begin{center}\small\begin{tabular}{lrrrrrr}\hline 模式 & 尝试 & 完成 & 条件 & 物理代理 & 跟踪 & 联合 \\ \hline'+'\n'
    for key,row in rows.items():
        tex+=label[key]+' & '+' & '.join(str(row[name]) for name in ('attempts','completed','condition_pass','physical_pass','tracking_pass','joint_pass'))+r' \\'+'\n'
    tex+=r'\hline\end{tabular}\end{center}'+'\n'
    tex+='前两次修复在控制前分别因目标未初始化及坐标断言失败，原失败保留。V3的XY锚点改善不等于完整初态通过；首批导出刚体状态仍有差异，后续批次未启动。所有质量阈值保持不变。保守合计 %.3f GPU小时。\n'%result['combined_charged_gpu_hours']
    (old/'overleaf_experiment_record.tex').write_text(tex,encoding='utf8')
    publication=json.loads((old/'publication_versions.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    publication.update(report_sha256=sha(old/'experiment_report.html'),tex_sha256=sha(old/'overleaf_experiment_record.tex'),
        measurement_appendix_publisher_sha256=sha(Path(__file__)),independent_measurement_summary_sha256=sha(repair/'measurement_repair_summary.json'),
        initialization_repair='separate; not pooled with historical run',
        combined_charged_gpu_hours=result['combined_charged_gpu_hours'])
    (old/'publication_versions.json').write_text(json.dumps(publication,indent=2),encoding='utf8')
    print('SEPARATE_MEASUREMENT_RECORD_APPENDED',result['next_stage'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);main(Path(p.parse_args().repo))
