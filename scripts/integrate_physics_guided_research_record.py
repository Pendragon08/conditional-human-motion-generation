"""Incrementally integrate completed diagnostics into the existing R46 records."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(repo, paper):
    out=repo/'results/physics_guided_20261011'
    summary=json.loads((out/'summary.json').read_text(encoding='utf8'))
    repair_path=repo/'results/reset_alignment_v3_20261011/measurement_repair_summary.json'
    repair=json.loads(repair_path.read_text(encoding='utf8'))
    assert repair['next_stage'].startswith('STOP_'), 'Do not finalize before a required controlled continuation'
    assert json.loads((out/'queue_state.json').read_text())['status']=='COMPLETE_GATE_NOT_MET', \
        'This negative-result record must not be used for a passed gate'
    ordinary=summary['summary']['ordinary']; random=summary['summary']['random_best']; proposed=summary['summary']['proposed']
    assert all(r['attempts']==432 for r in (ordinary,random,proposed))
    source=paper/'content_cloud_before_physics_experiment_20261011.tex'
    report_source=paper/'literature_review_report_cloud_before_physics_experiment_20261011.tex'
    body=source.read_text(encoding='utf8'); review=report_source.read_text(encoding='utf8')
    assert body.count(r'\section{Conclusion}')==1
    assert review.count(r'\section{记录与一致性}')==1
    assert r'\input{physics_guided_paper_status_20261011}' not in body
    assert r'\input{overleaf_experiment_record}' not in review
    status=r'''% Measured diagnostic status, separate from proposed full method.
\subsection{条件反馈机制的诊断验证}
\label{sec:physics-guided-diagnostic-status}
为检验执行后条件误差能否改善参考构建，本轮冻结DiP生成器与CLoSD人体物理跟踪器，比较未经精化的生成参考、普通物理精化、等预算候选筛选、执行前缀反馈及有界条件纠偏，并分别去除、打乱反馈或更换前缀与筛选依据。每方法在48个原始MoCap来源、三个名义水平目标偏移和三个种子下进行432次独立回放；同一案例和种子内的原始目标固定，所有失败均保留。本轮是人体仿真诊断，不是目标人形机器人的部署实验，也不是Morph、PhysDiff或CLAIMS的完整复现。
'''
    status+=('普通精化、等预算随机筛选与候选条件纠偏的联合合格次数分别为%d、%d和%d（各432次）。'
             %(ordinary['joint_pass'],random['joint_pass'],proposed['joint_pass']))
    status+=r'''独立复核发现，参考和目标锚点在角色创建时构造，而实际角色随后重置，二者没有同步平移。种子变化因而同时改变生成噪声和目标相对实际角色的位置，名义目标偏移也混入了随机锚点差异。因此，初始化坐标测量校验没有通过；不能把这些数值视为受控偏移实验或纯随机种子重复，也不能据此证明条件纠偏有效或无效。
人体进入门槛未通过，因此按停止规则未开展G1迁移及下游策略训练。验证集还出现联合判据地板效应，主要受足部滑移代理与连续性限制；数值阈值在验证前固定，但没有按验证分布进行校准，这是与原计划的程序偏差。运行中还发生过实现核验与版本修订，早期输出的历史请求绑定不足。因此，本轮仅作为经过修订的诊断记录，不能作为确认性创新验证。
\begin{itemize}
\item 有效且有技术差异：本轮尚未建立此类结论。
\item 有效但属于已有机制：物理精化、执行反馈及闭环数据构建已有文献依据\citep{closd,parc,claims}；本轮不能仅凭回放完成率将其解释为联合质量改善。
\item 未获支持：本轮尚未取得可用于证明有界条件纠偏收益的有效实验依据；合成数据的下游训练收益尚未检验。
\end{itemize}
完整对比、消融、指标分解、尝试登记与版本记录保存在本项目的\texttt{literature\_review\_report.tex}研究记录及\href{https://github.com/Pendragon08/conditional-human-motion-generation}{公开实验仓库}中。后续须先独立核验任务窗口与验收判据的区分能力，再冻结新协议开展验证。
'''
    quality=repair['full_validation_quality']
    if repair['coordinate_gate_pass']:
        status+='\n独立初始化修复后，216次测量尝试完成，跨种子原始目标、计划前缀与初始导出运动学状态一致，名义目标偏移得到核验。'
    else:
        status+='\n独立初始化复核仍未通过，不能据此展开机制有效性比较。'
    if repair['joint_gate_floor']:
        status+='两种前缀模式在完整验证集各72次尝试中均没有联合合格动作；因此当前设置没有通过后续比较的进入条件。该结果提示需先校准接触代理和跟踪设置，不能将其当作候选反馈机制无效的证据。'
    status+='独立复核不与旧轮合并，质量阈值未改变；首次修复在控制前的失败亦完整保留。\n'
    (paper/'physics_guided_paper_status_20261011.tex').write_text(status,encoding='utf8')
    shutil.copy2(out/'overleaf_experiment_record.tex',paper/'overleaf_experiment_record.tex')
    for name in ('content.tex','literature_review_report.tex','research_gap_review.html'):
        old=paper/name; backup=paper/(old.stem+'_local_before_physics_experiment_20261011'+old.suffix)
        if old.exists() and not backup.exists(): shutil.copy2(old,backup)
    (paper/'content.tex').write_text(body.replace(r'\section{Conclusion}',r'\input{physics_guided_paper_status_20261011}'+'\n\n'+r'\section{Conclusion}'),encoding='utf8')
    (paper/'literature_review_report.tex').write_text(review.replace(r'\section{记录与一致性}',r'\input{overleaf_experiment_record}'+'\n\n'+r'\section{记录与一致性}'),encoding='utf8')
    report=paper/'research_gap_review.html'
    original=report.read_text(encoding='utf8')
    start='<!-- PHYSICS_GUIDED_DIAGNOSTIC_20261011_START -->'; end='<!-- PHYSICS_GUIDED_DIAGNOSTIC_20261011_END -->'
    if start in original:
        assert end in original
        a=original.index(start);b=original.index(end)+len(end);original=original[:a]+original[b:]
    rows=''.join('<tr><td>'+html.escape(label)+'</td><td>432</td><td>'+str(r['joint_pass'])+'</td><td>'+str(r['completed'])+'</td></tr>'
                 for label,r in [('普通精化',ordinary),('等预算随机筛选',random),('有界条件反馈',proposed)])
    addition=start+'''<section id="physics-guided-experiment-20261011"><h2>物理引导条件动作生成：本轮对比与消融结果</h2>
<p><strong>诊断实验，人体门槛未通过；未开展G1与下游训练。</strong>本轮冻结生成器和跟踪器，每方法432次独立回放。结果未支持将有界条件反馈认定为已验证创新。</p>
<table><thead><tr><th>方法</th><th>尝试</th><th>联合合格</th><th>回放完成</th></tr></thead><tbody>'''+rows+'''</tbody></table>
<p><strong>初始化测量校验未通过：</strong>参考和目标使用创建时的随机锚点，实际角色随后重置且未共同平移；三个种子同时改变生成噪声和任务几何，名义±15厘米条件也受到混杂。不能据此判定机制有效或无效。</p>
<p>回放完成不是联合合格。验证集还存在联合判据地板效应；足部滑移采用表面几何与支撑状态代理，不能当作真实接触点测量。阈值未按验证分布校准，运行中有实现与溯源修订，因此结果限定为经过修订的诊断记录。</p>
<p>“有效且有技术差异”尚未建立；物理精化与执行反馈属于已有机制；本轮条件纠偏“未获支持”。合成数据能否改善机器人学习尚未验证。</p>
<p><a href="../conditional-human-motion-generation/results/physics_guided_20261011/experiment_report.html">完整对比、消融、语义代理、资源消耗与失败记录</a> · <a href="https://github.com/Pendragon08/conditional-human-motion-generation">现有GitHub实验仓库</a></p></section>'''+end
    repair_text=('独立初始化复核计划216例，实际启动'+str(repair['attempts'])+'例、完成'+str(repair.get('completed_cases',repair['attempts']))+'例，测量校验'+('通过' if repair['coordinate_gate_pass'] else '未通过')+'。')
    if quality:
        repair_text+='完整验证集中计划前缀与执行前缀的联合合格次数分别为 '+str(quality['full_val_planned_s11']['joint_pass'])+' 和 '+str(quality['full_val_native_s11']['joint_pass'])+'（各72次）。'
    else:
        repair_text+='首批XY误差小于1毫米，但导出刚体状态不一致；下一批前因保护其他工作负载的资源断言停止。跨种子、跨模式和完整验证未运行，原RUNNING状态是异常退出留下的旧状态。'
    repair_text+='复核不与旧轮合并，阈值未放宽；当前停止状态为 '+repair['next_stage']+'，未进入G1与下游训练。'
    addition=addition.replace('</section>'+end,'<p>'+html.escape(repair_text)+'</p></section>'+end)
    assert '</body>' in original
    report.write_text(original.replace('</body>',addition+'\n</body>'),encoding='utf8')
    names=['content.tex','literature_review_report.tex','physics_guided_paper_status_20261011.tex','overleaf_experiment_record.tex','research_gap_review.html']
    manifest=dict(authoritative_cloud_backup_sha256={p.name:digest(p) for p in (source,report_source)},
                  updated_files_sha256={name:digest(paper/name) for name in names},
                  bibliography='Reuse existing closd/parc/claims keys; no new or duplicate entries',
                  independent_measurement_summary_sha256=digest(repair_path),
                  cloud_upload_status='pending UI upload and verification')
    (paper/'physics_guided_integration_manifest_20261011.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    print('R46_INCREMENTAL_RECORDS_READY',len(names))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--paper',required=True)
    a=p.parse_args();main(Path(a.repo),Path(a.paper))
