"""Post-run publication; leaves frozen execution source and version record intact."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import time

from physics_guided_publish import main as publish_base, LABELS, table
from physics_guided_lineage import file_hash


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def main(out):
    state=json.loads((out/'queue_state.json').read_text())
    assert state['status'] in ('COMPLETE_GATE_NOT_MET','HUMAN_GATE_PASSED_PENDING_VALIDATED_G1_AND_TRAINING_ADAPTERS')
    publish_base(out)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/physics_guided_20261011.json').read_text())
    summary=json.loads((out/'summary.json').read_text()); gate=summary['gate']
    formal=json.loads((out/'attempt_register.json').read_text())
    assert len(formal)==74 and sum(r.get('cases',0) for r in formal)==10512
    assert len(summary['summary'])==10 and all(r['attempts']==432 and r['sources']==48 for r in summary['summary'].values())
    arms=json.loads((out/'arms.json').read_text())
    assert len(arms)==30
    ancillary=[]; ancillary_rows=[]
    for directory in sorted(out.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(('development','witness')):
            continue
        receipt_path=directory/'receipt.json'
        log=out/(directory.name+'.log')
        record=dict(label=directory.name,scope='development_or_document_witness',
                    included_in_formal_comparison=False,charged_in_initial_reserved_debit=True)
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text()); meta=json.loads((directory/'metadata.json').read_text())
            rows=json.loads((directory/'metrics.json').read_text())
            record.update(status=receipt['status'],cases=len(rows),completed=sum(r['completed'] for r in rows),
                          elapsed_seconds=receipt['elapsed_seconds'],receipt_sha256=file_hash(receipt_path),
                          metadata_sha256=file_hash(directory/'metadata.json'),metrics_sha256=file_hash(directory/'metrics.json'),
                          trajectories_sha256=file_hash(directory/'trajectories.npz'))
            for row in rows:
                ancillary_rows.append(dict(label=directory.name,scope=record['scope'],case_id=row['case_id'],
                    family=row['family'],seed=row['seed'],length=row['length'],completed=row['completed'],
                    joint_pass=row['joint_pass'],metric_version='historical diagnostic; not comparable to final formal scores'))
        else:
            record.update(status='failed_or_incomplete_retained',cases=None,
                          log_sha256=file_hash(log) if log.exists() else None,
                          count_note='No complete receipt; do not invent per-sample completion or timing')
        ancillary.append(record)
    cpu_failures=[]
    for line in (out/'queue.log').read_text().splitlines():
        marker='PHYSICS_GUIDED_QUEUE '
        if line.startswith(marker):
            record=json.loads(line[len(marker):])
            if record['status']=='FAILED_RETAINED': cpu_failures.append(record)
    amendments={p.name:json.loads(p.read_text()) for p in out.glob('*amendment*.json')}
    reconstruction=out/'intermediate_version_reconstruction.json'
    if reconstruction.exists(): amendments[reconstruction.name]=json.loads(reconstruction.read_text())
    pause_history=json.loads((out/'queue_pause_history.json').read_text())
    register=dict(formal_batches=formal,development_and_document_witnesses=ancillary,
                  cpu_queue_failures=cpu_failures,queue_pause_history=pause_history,amendments=amendments,
                  version_snapshot_hashes={p.name:file_hash(p) for p in out.glob('versions*.json')},
                  formal_first_attempts=10512,formal_replay_attempts=4320,
                  other_attempt_count_known=sum(r.get('cases') or 0 for r in ancillary),
                  unknown_failed_attempt_counts=sum(r.get('cases') is None for r in ancillary),
                  budget_policy='All development/witness failures covered by conservative initial 1800s debit; formal tasks and semantic evaluation use wall-clock ledger',
                  study_status='amended exploratory diagnostic, not clean confirmatory experiment')
    write_json(out/'complete_attempt_register.json',register)
    if ancillary_rows:
        with (out/'development_and_witness_attempts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(ancillary_rows[0]));w.writeheader();w.writerows(ancillary_rows)
    lineage=json.loads((out/'lineage_register.json').read_text())
    for record in lineage:
        directory=out/'raw'/record['label']
        compatibility=directory/'compatibility_after_amendment.json'
        history=directory/'lineage_history.json'
        if compatibility.exists():
            record['compatibility_after_amendment']=json.loads(compatibility.read_text())
            record['request_role']='original input-bound execution request preserved; compatibility with later nonexecution version separately recorded'
        if history.exists(): record['prior_lineage_records']=json.loads(history.read_text())
    write_json(out/'lineage_register.json',lineage)
    families={}
    components=config['base_thresholds']
    for arm in LABELS:
        rows=[]
        for seed in config['seeds']:
            path=Path(arms[arm+':'+str(seed)])
            package=json.loads((path/'metrics.json').read_text())
            choices=json.loads((out/'selected'/(arm+'_s%d.json'%seed)).read_text())['choices']
            for r,c in zip(package,choices):
                r['joint_pass']=bool(r['joint_pass'] and c['construction_complete']); rows.append(r)
        families[arm]={}
        for family in config['families']:
            group=[r for r in rows if r['family']==family]
            assert len(group)==216
            mean=lambda key:sum(r[key] for r in group)/len(group)
            families[arm][family]=dict(attempts=len(group),joint_pass=sum(r['joint_pass'] for r in group),
                completed=sum(r['completed'] for r in group),endpoint_error_m=mean('endpoint_error_m'),
                heading_error_rad=mean('heading_error_rad'),tracking_mean_m=mean('tracking_mean_m'),
                condition_pass=sum(r['condition_pass'] for r in group),physical_pass=sum(r['physical_pass'] for r in group))
    write_json(out/'per_family_summary.json',families)
    resource=json.loads((out/'resource_summary.json').read_text())
    resource.update(formal_batch_count=74,formal_first_attempt_count=10512,formal_replay_count=4320,
                    development_receipt_count=sum(r.get('cases') is not None for r in ancillary),
                    development_and_witness_first_attempts_known=register['other_attempt_count_known'],
                    failed_development_batch_counts_unknown=register['unknown_failed_attempt_counts'],
                    queue_cpu_failure_count=len(cpu_failures))
    write_json(out/'resource_summary.json',resource)
    extra='<h2 id="records">完整尝试与版本修订</h2><p>正式实验共74个批次、10512次构建或回放首轮尝试，其中独立回放4320次。'
    extra+='此外单独登记开发、文档见证和失败的队列校验；无完成凭据的早期失败不补造样本数量或成绩。</p>'
    extra+=table(['开发/见证批次','状态','已知尝试','正式比较'],[[r['label'],r['status'],r.get('cases') if r.get('cases') is not None else '不可恢复','否'] for r in ancillary])
    extra+='<p>运行中修正包括：参考时钟与状态读取、候选前缀、旧输出兼容性、NPZ解压性能、评价器版本绑定和单精度输入校验。'
    extra+='原执行请求、版本快照与修正历史保留。它们不构成事先注册的确认性证据。</p>'
    extra+='<p><a href="complete_attempt_register.json">完整尝试登记</a> · <a href="per_family_summary.json">按动作族分解</a> · <a href="lineage_register.json">输入与输出溯源</a></p>'
    extra+='<p>数值进入门槛对每个等预算对照分别检查，避免最高分对照的并列顺序影响判定。展示的“最强对照”是观察后选出的描述性对象；'
    extra+='其单个置信区间不解释为经过对照选择校正的确认性区间。条件与语义保护采用预先固定的点估计容差。</p>'
    doc=(out/'experiment_report.html').read_text()
    doc=doc.replace('进入后续阶段需超过最强等预算对照至少10个百分点','进入后续阶段需相对每个等预算对照至少提高10个百分点')
    doc=doc.replace('</body></html>',extra+'</body></html>')
    if gate['pass']:
        doc=doc.replace('未通过人体门槛，未运行G1与下游训练','人体门槛已通过，但G1与下游训练仍待经验证接口执行')
        doc=doc.replace('“未获支持”</strong>适用于本轮候选条件反馈机制','“有效但创新性尚待核实”</strong>适用于本轮人体数值结果')
    (out/'experiment_report.html').write_text(doc,encoding='utf8')
    tex=r'''% Experimental research record; does not claim successful novelty.
\section{物理引导条件动作生成：受控对比与消融诊断}
\label{sec:physics-guided-diagnostic-20261011}
\subsection{目的、机制与证据边界}
本轮检验物理处理是否改善参考动作的可用性，同时保留原始条件。冻结DiP生成权重和CLoSD人体物理跟踪器，使用有限次条件纠偏构造候选：根据执行终点相对于固定目标的位置与朝向误差，修正下一轮生成条件，最多三个候选。每次位置调整不超过5厘米、累计不超过15厘米，朝向调整不超过5度；文本、动作长度和目标时刻不变。
物理精化、执行状态反馈和闭环数据扩展已有直接先例\cite{closd,parc,claims}，不能将这些机制本身作为创新。本轮是CLoSD发布组件的适配诊断，不是Morph、PhysDiff或CLAIMS完整复现。
\subsection{数据、对照与验收}
跳跃与舞蹈各选64个训练来源、12个验证来源、24个测试来源。按官方原始MoCap来源分组，镜像与切片不跨本轮划分；发布权重的预训练暴露并未排除。测试48个来源、三个目标水平偏移和三个生成种子，每方法432次输入。任务仅由文本标签预先筛选，4秒窗口不保证包含完整技能事件，因此不能称为完整舞蹈或跳跃专项基准。
对照包括未经精化的生成参考、普通固定跟踪精化、等预算随机筛选及执行前缀反馈。消融分别去除条件误差反馈、跨原始来源打乱反馈、仅按跟踪表现选候选，以及同时改用计划前缀和计划坐标锚点。所有参考经同一冻结跟踪器独立回放；构建失败、提前终止及不合格候选保留。
联合验收同时要求条件、物理代理、跟踪与完整执行合格。足部滑移按支撑状态、足部碰撞表面及角速度计算，是代理而非真实接触点测量；腾空本身不判为漂浮。验证阶段用于检查测量有效性，数值阈值已在验证前写入，未按验证分布重新校准。这与原计划的阈值校准程序存在偏差；本轮不在看到测试结果后放宽阈值。
\subsection{独立回放结果}
\begin{center}\small
\begin{tabular}{p{6.5cm}rrr}\hline
方法 & 尝试 & 联合合格 & 完成回放 \\\hline
'''
    for arm in LABELS:
        row=summary['summary'][arm]
        tex+=LABELS[arm]+(' & %d & %d & %d'%(row['attempts'],row['joint_pass'],row['completed']))+chr(92)*2+'\n'
    tex+=r'\hline\end{tabular}\end{center}'+'\n'
    tex+='\n回放完成表示达到120帧且未触发提前终止，不等同于条件与物理联合合格；提前终止也不能全部称为跌倒。\n'
    tex+=r'\subsection{配对比较与消融结论}'+'\n'
    for name,c in summary['comparisons'].items():
        tex+=LABELS[name]+('：候选方法提升 %.2f 个百分点，来源聚类95'%(c['gain_pp']))+chr(92)+'%'+('置信区间[%.2f, %.2f]。\n'%tuple(c['ci95_pp']))
    tex+='\n进入后续阶段要求候选方法对每个等预算对照均提高至少10个百分点、来源聚类置信区间排除零、三个种子同向，并通过条件、语义与多样性保护。\n'
    tex+=r'\subsection{阶段判定与论文贡献分类}'+'\n'
    if not gate['pass']:
        tex+=r'''本轮人体门槛未通过，因此按停止规则不开展G1与下游九组训练，不补造未执行结果。结论分为：
\begin{itemize}
\item 有效且有技术差异：本轮未建立此类结论。
\item 有效但属于已有机制：物理精化与执行反馈是已有路线；本轮不能由回放完成率将其直接解释为联合质量收益。
\item 未获支持：本轮有界条件纠偏未达到预定联合验收提升门槛，不能作为已验证的论文创新。
\end{itemize}
验证集已出现零联合合格率，存在判据交集的地板效应风险。否定结果仅针对本轮固定代理、动作窗口及冻结控制设置，不能推及所有反馈方法。后续若修改任务窗口或判据，须先独立验证并冻结，再重新测试。
'''
    else:
        tex+='人体门槛通过，但技术差异、G1迁移及下游训练收益仍须分别验证；目前不声称论文创新已成立。\n'
    tex+=r'''本轮在实现复核中发生过修正，并逐条核验复用的早期输出。因此归类为经过修订的诊断实验，不能称为无修订的确认性实验。原执行请求、轨迹散列、版本快照、完整尝试与失败记录均保留，缺失的历史执行绑定不追溯补造。语义与多样性采用冻结HumanML3D文本--动作嵌入评价器，是学习型代理而非人工真值。
'''
    tex+='\n正式记录含74个批次、10512次构建或回放首轮尝试，其中4320次独立回放。保守计费GPU时间为 %.3f 小时，低于72小时上限；本轮输出限制15GiB，旧实验与其他项目未删除。\n'%resource['total_gpu_hours']
    tex+=r'完整数值、对照、消融、版本与失败记录见本地实验报告和\href{https://github.com/Pendragon08/conditional-human-motion-generation}{现有GitHub项目}。'+'\n'
    (out/'overleaf_experiment_record.tex').write_text(tex,encoding='utf8')
    write_json(out/'publication_versions.json',dict(time=time.time(),publisher_sha256=file_hash(Path(__file__)),
               execution_versions_sha256=file_hash(out/'versions.json'),summary_sha256=file_hash(out/'summary.json'),
               report_sha256=file_hash(out/'experiment_report.html'),tex_sha256=file_hash(out/'overleaf_experiment_record.tex')))
    print('POST_RUN_PUBLICATION_COMPLETE',len(formal),resource['total_gpu_hours'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    main(Path(p.parse_args().output).resolve())
