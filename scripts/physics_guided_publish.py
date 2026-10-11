"""Publish measured diagnostics and all attempt counts, without motion redistribution."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import time
import numpy as np
from physics_guided_lineage import file_hash

ROOT=Path(__file__).resolve().parents[1]
LABELS={'raw':'未精化生成参考','ordinary':'普通物理精化','random_best':'等预算随机候选筛选',
        'native_single':'原生执行前缀反馈（单候选）','native_best':'原生执行前缀反馈（等预算筛选）',
        'proposed':'候选方法：执行前缀＋有界条件反馈','no_feedback':'消融：去除条件误差反馈',
        'shuffled':'消融：打乱原始来源反馈','planned_prefix':'消融：计划前缀＋计划锚点',
        'tracking_only':'消融：仅按跟踪表现筛选'}


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+\
           '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main(out):
    data=json.loads((out/'summary.json').read_text()); s=data['summary']; gate=data['gate']
    config=json.loads((ROOT/'configs/physics_guided_20261011.json').read_text())
    arms=json.loads((out/'arms.json').read_text()); thresholds=config['base_thresholds']
    attempts=[]; lineage=[]; raw_rows=[]
    for directory in sorted((out/'raw').iterdir()):
        if not (directory/'receipt.json').exists():
            attempts.append(dict(label=directory.name,status='incomplete_retained')); continue
        receipt=json.loads((directory/'receipt.json').read_text()); meta=json.loads((directory/'metadata.json').read_text())
        rows=json.loads((directory/'metrics.json').read_text())
        record=dict(label=directory.name,status=receipt['status'],mode=meta['mode'],seed=meta['seed'],round=meta['round'],
                    cases=len(rows),completed=sum(r['completed'] for r in rows),joint_pass=sum(r['joint_pass'] for r in rows),
                    generation_batch_calls=meta['generation_batch_calls'],elapsed_seconds=receipt['elapsed_seconds'],
                    original_goal_sha256=meta['original_goal_sha256'],trajectories_sha256=receipt['trajectories_sha256'],
                    metadata_sha256=file_hash(directory/'metadata.json'),metrics_sha256=file_hash(directory/'metrics.json'))
        binding=directory/'lineage.json'
        if binding.exists():
            b=json.loads(binding.read_text()); record['certification']=b['certification']
            lineage.append(dict(label=directory.name,lineage=b,request_role=('retrospective compatibility target, not historical execution request'
                           if b['certification'].startswith('retrospective') else 'input-bound execution request')))
        attempts.append(record)
        for r in rows:
            raw_rows.append(dict(label=directory.name,source_sha256=hashlib.sha256(r['source_id'].encode()).hexdigest(),
                **{k:r.get(k) for k in ('case_id','clip_id','family','offset_m','seed','round','mode','length','completed','terminated',
                 'joint_pass','condition_pass','physical_pass','tracking_pass','endpoint_error_m','heading_error_rad',
                 'endpoint_time_error_s','tracking_mean_m','tracking_p95_m','surface_slip_proxy_fraction','support_observations',
                 'penetration_fraction','jerk_p95_m_s3')}))
    (out/'attempt_register.json').write_text(json.dumps(attempts,indent=2))
    (out/'lineage_register.json').write_text(json.dumps(lineage,indent=2))
    with (out/'all_construction_and_replay_attempts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw_rows[0]));writer.writeheader();writer.writerows(raw_rows)
    component_rows=[]; components={}; costs={}
    for arm in LABELS:
        rows=[]; construction_seconds=0.; replay_seconds=0.; calls=0
        for seed in config['seeds']:
            directory=Path(arms[arm+':'+str(seed)])
            rows+=json.loads((directory/'metrics.json').read_text())
            replay_seconds+=json.loads((directory/'receipt.json').read_text())['elapsed_seconds']
            choices=json.loads((out/'selected'/(arm+'_s%d.json'%seed)).read_text())
            # Candidate paths include candidates not selected; count each full construction batch.
            binding=json.loads((out/'selected'/(arm+'_s%d.lineage.json'%seed)).read_text())
            if arm=='raw' or arm=='ordinary': paths=[out/'raw'/('test_planned_s%d_r0'%seed)]
            elif arm=='random_best': paths=[out/'raw'/('test_planned_s%d_r%d'%(seed,r)) for r in range(3)]
            elif arm.startswith('native'): paths=[out/'raw'/('test_native_s%d_r%d'%(seed,r)) for r in range(1 if arm=='native_single' else 3)]
            else:
                family='proposed' if arm=='tracking_only' else arm
                paths=[out/'raw'/('test_'+('planned' if arm=='planned_prefix' else 'native')+'_s%d_r0'%seed)]+\
                      [out/'raw'/('test_%s_s%d_r%d'%(family,seed,r)) for r in (1,2)]
            assert len(paths)==choices['construction_candidates']==len(binding['candidate_outputs'])
            for path in paths:
                m=json.loads((path/'metadata.json').read_text());construction_seconds+=m['elapsed_seconds'];calls+=m['generation_batch_calls']
        component={}
        checks={'条件合格':'condition_pass','物理代理合格':'physical_pass','跟踪合格':'tracking_pass'}
        for name,key in checks.items(): component[name]=sum(r.get(key,False) for r in rows)
        component.update(完成=sum(r['completed'] for r in rows),位置=sum(r['endpoint_error_m']<=thresholds['goal_position_m'] for r in rows),
            朝向=sum(r['heading_error_rad']<=thresholds['heading_rad'] for r in rows),
            时刻=sum(r['endpoint_time_error_s']<=thresholds['goal_time_s']+1e-6 for r in rows),
            地面穿透=sum(r['penetration_fraction']<=thresholds['penetration_fraction'] for r in rows),
            滑移代理=sum(r['surface_slip_proxy_fraction'] is not None and r['surface_slip_proxy_fraction']<=thresholds['sliding_fraction'] for r in rows),
            连续性=sum(r['jerk_p95_m_s3'] is not None and r['jerk_p95_m_s3']<=thresholds['continuity_jerk_p95_m_s3'] for r in rows))
        components[arm]=component
        component_rows.append([LABELS[arm]]+[component[key] for key in ('完成','条件合格','物理代理合格','跟踪合格','位置','朝向','时刻','地面穿透','滑移代理','连续性')])
        costs[arm]=dict(construction_gpu_wall_seconds=construction_seconds,replay_gpu_wall_seconds=replay_seconds,
                       generation_batch_calls=calls,candidates_per_input=choices['construction_candidates'],
                       seconds_per_joint_qualified=None if not s[arm]['joint_pass'] else (construction_seconds+replay_seconds)/s[arm]['joint_pass'],
                       accounting='per-arm equivalent standalone cost; shared parent batches counted per arm, not additive')
    (out/'component_metrics.json').write_text(json.dumps(components,indent=2,ensure_ascii=False))
    ledger=json.loads((out/'budget_ledger.json').read_text())
    budget=dict(total_gpu_hours=sum(r['seconds'] for r in ledger)/3600,
                by_stage_gpu_hours={stage:sum(r['seconds'] for r in ledger if r['stage']==stage)/3600 for stage in config['budget_gpu_hours']},
                conservative_initial_debit_seconds=1800,hard_limit_gpu_hours=72,new_output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                data_free_bytes=shutil.disk_usage('/data').free,per_arm_equivalent_costs=costs,
                exact_task_count=len(attempts),first_attempts=sum(r.get('cases',0) for r in attempts))
    (out/'resource_summary.json').write_text(json.dumps(budget,indent=2))
    assert budget['total_gpu_hours']<=72 and budget['new_output_bytes']<=config['max_output_bytes']
    summary_rows=[[LABELS[a],v['attempts'],v['joint_pass'],'%.2f%%'%(v['joint_rate']*100),v['completed'],
       '%.3f'%v['endpoint_error_m'],'%.1f'%(np.degrees(v['heading_error_rad'])),'%.3f'%v['tracking_mean_m'],v['early_terminated']]
       for a,v in s.items()]
    comparison_rows=[[LABELS[a],'%.2f'%c['gain_pp'],'[%.2f, %.2f]'%tuple(c['ci95_pp']),str(c['seed_gains_pp'])] for a,c in data['comparisons'].items()]
    semantic=gate['semantic_diversity_gate']; semantic_rows=[]
    for a,v in semantic.get('per_arm',{}).items():
        semantic_rows.append([LABELS[a],*['%.4f'%v[k] for k in ('matching_distance','diversity','multimodality')]])
    audit_path=out/'EXPERIMENT_AUDIT.md'
    audit_text=html.escape(audit_path.read_text()) if audit_path.exists() else '独立审计尚待完成，不能标为审计通过。'
    previews=out/'previews/manifest.json'
    media=''
    if previews.exists():
        for item in json.loads(previews.read_text()):
            media+='<figure><video controls preload="metadata" src="previews/'+item['video']+'"></video><figcaption>'+html.escape(item['family']+' / '+item['case_id'])+'：预先确定的首个来源，种子11，零偏移；记录状态的骨架回放。</figcaption></figure>'
    doc='''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>物理引导条件动作生成：受控诊断实验</title><style>
:root{--ink:#203449;--blue:#185f89;--soft:#edf4f8}body{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:var(--ink);max-width:1320px;margin:30px auto;padding:0 24px}
h1{font-size:30px}h2{margin-top:36px;border-bottom:2px solid var(--soft);padding-bottom:6px}a{color:var(--blue)}.notice{background:#fff4d9;border-left:5px solid #bd7900;padding:16px 22px}
nav{background:var(--soft);padding:16px 22px}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}td,th{border:1px solid #cbd6df;padding:7px;text-align:left}th{background:var(--soft)}
.scroll{overflow-x:auto}pre{white-space:pre-wrap;background:var(--soft);padding:16px;font-size:13px}video{max-width:100%;width:900px}code{font-size:13px}
@media print{body{margin:0;max-width:none;padding:0}nav,video{display:none}h2{break-after:avoid}tr,figure{break-inside:avoid}a{color:inherit}details>div{display:block}}
</style></head><body><h1>物理引导条件动作生成：对比与消融诊断实验</h1><p>2026-10-11 · 冻结DiP生成器与CLoSD人体跟踪器 · 仅物理仿真</p>
<nav>目录：<a href="#result">结论</a> · <a href="#method">方法与公平设置</a> · <a href="#comparison">主对比</a> · <a href="#ablation">消融</a> · <a href="#components">失败分解</a> · <a href="#cost">成本</a> · <a href="#limits">证据边界</a> · <a href="#audit">审计</a></nav>
<h2 id="result">实验结论</h2><div class="notice">'''+html.escape(gate['reason'])+'''。<strong>当前不支持将候选机制作为论文创新。</strong>
本轮是在实现复核中发生过修正、保留并逐条核验早期有效输出的诊断实验，不能称为未经调整的确认性实验。</div>
<p>人体门槛'''+('已通过，后续接口与训练仍须实证。' if gate['pass'] else '未通过，因此依照停止规则不开展G1验证和下游九组训练。没有填写未执行的分数。')+'''</p>
<p>结论分类：<strong>“未获支持”</strong>适用于本轮候选条件反馈机制。CLoSD执行状态反馈、物理精化与数据回流属于已有机制；
本轮未建立“有效且有技术差异”的机制，也不能由零联合合格率断言这些已有路线在所有场景均无效。</p>
<h2 id="method">方法与公平设置</h2><p>跳跃和舞蹈各64个训练来源、12个验证来源、24个测试来源；镜像与切片按官方原始MoCap source_path分组。
本轮测试48个原始来源×3个水平目标偏移（−15/0/+15cm）×3个种子（11/12/13），每方法432次首轮尝试，所有失败保留。
发布权重可能预训练见过这些人体来源，本轮分组独立不等于预训练未暴露。</p>
<p>4秒动作：2秒已有动作前缀，DiP接收最后1秒上下文并生成2秒动作；人体物理控制30Hz。
跳跃条件为骨盆终点，舞蹈条件为右手腕终点，使用发布模型的目标位置与朝向接口。
只有片段文本标签用于任务筛选，未证实4秒截取片段一定包含完整跳跃或舞蹈事件；这不是完整舞蹈/跳跃专项基准。</p>
<ol><li>在生成前固定数据集终点、偏移目标、朝向与目标时刻。</li><li>冻结生成权重与跟踪器，构造最多三个候选；候选方法使用实际执行前缀与实际坐标锚点。
根据相对于原始目标的误差修正下一轮生成条件，位置每次≤5cm、累计≤15cm；朝向每次≤5°，保持文本、长度与目标时刻不变。</li>
<li>随机筛选、原生状态反馈与消融采用相同三候选构建预算。原始评价目标不随修正变化。</li>
<li>所有选出的参考动作重新交给同一冻结策略独立回放；构建失败与回放失败均纳入分母。</li></ol>
<h2 id="comparison">独立回放主对比</h2><div class="scroll">'''+table(['方法','尝试','联合合格','合格率','回放完成','目标位置误差(m)','朝向误差(°)','跟踪均值(m)','提前终止'],summary_rows)+'''</div>
<p>回放完成表示达到120帧且未触发提前终止，并不等于动作合格；提前终止受跟踪失败等终止条件影响，不能全部称为跌倒。
目标时刻是固定终点的几何接近时刻代理，并非真实接触事件时刻。未设定音乐条件、接触力目标或任意关键帧接口。</p>
<h2 id="ablation">配对比较与消融</h2>'''+table(['对照','候选方法提升(百分点)','原始来源聚类95%CI','三个种子提升'],comparison_rows)+'''
<p>每个原始来源包含三个偏移和三个种子。按来源聚类、跳跃/舞蹈分层进行5000次自助抽样。
进入后续阶段需超过最强等预算对照至少10个百分点、置信区间排除零、三个种子同向，且条件、语义和多样性未超过5%退化。</p>
<ul><li>无反馈：保留实际前缀与计算预算，条件误差修正置零。</li><li>打乱反馈：整个原始来源组之间置换，保持目标偏移对应；不是同一来源内部三个目标互换。</li>
<li>计划前缀：同时替换前缀和坐标锚点，避免残留实际执行信息。</li><li>仅跟踪筛选：采用同组候选，仅按完成/跟踪表现选择，不用原始条件误差排序。</li></ul>
<h2 id="components">联合门槛的失败分解</h2><p>以下为432次回放中的单项合格次数；即使单项通过，构建失败、提前终止或其他单项失败仍使联合判据不通过。
足部滑移按支撑状态、足部碰撞表面和角速度计算，是滑移代理，不能解释为接触点真值。
生成窗口全程无支撑时，足部滑移记为不可测量，不能通过支撑质量验收；腾空本身不判为漂浮。</p>
<div class="scroll">'''+table(['方法','完成','条件','物理代理','跟踪','位置','朝向','时刻','穿透','滑移代理','连续性'],component_rows)+'''</div>
<p><strong>阈值地板风险：</strong>验证集已有零联合合格率，提示多个判据的交集可能过严或任务适配不足。
数值判据在验证之前写入文件，验证用于测量有效性而未按分布重新校准，这与原计划存在程序偏差。
测试后不放宽阈值；本轮只能报告在该固定判据下未通过，不能据此普遍否定反馈方法。后续若重设判据，须先独立验证并冻结，再开始新测试。</p>
<h2>语义与多样性保护</h2><p>状态：'''+html.escape(semantic.get('status','not_verified'))+'''。
采用缓存HumanML3D文本—动作评价模型的嵌入距离、跨来源多样性和跨种子多模态性；是学习型代理，不是人工语义真值。失败动作补最后有效姿态，保留全部尝试。</p>
'''+(table(['方法','匹配距离↓','跨来源多样性','跨种子多模态性'],semantic_rows) if semantic_rows else '<p>未获得有效代理评价，不能声称语义保护通过。</p>')+'''
<h2>代表回放</h2>'''+(media or '<p>无代表视频，数值记录仍完整。</p>')+'''
<h2 id="cost">资源与生成成本</h2><p>实际保守计费GPU时间：<strong>%.3f小时</strong> / 72小时；本轮输出：%.3fGiB；数据盘剩余：%.2fGiB。
已保留失败运行与队列修正记录，没有删除旧项目，也没有付费模型调用。</p>'''%(budget['total_gpu_hours'],budget['new_output_bytes']/2**30,budget['data_free_bytes']/2**30)+\
table(['方法','每输入构建候选','构建GPU墙钟秒','独立回放秒','每条联合合格参考成本'],[[LABELS[a],v['candidates_per_input'],'%.1f'%v['construction_gpu_wall_seconds'],
       '%.1f'%v['replay_gpu_wall_seconds'],'不可估计（无合格动作）' if v['seconds_per_joint_qualified'] is None else '%.1fs'%v['seconds_per_joint_qualified']] for a,v in costs.items()])+'''
<p>单方法成本为等价独立运行成本，包含未选候选；共享的初始候选在各方法中分别计入，不可将各行相加当作服务器总消耗。总消耗以预算账本为准。</p>
<h2 id="limits">证据边界与后续处理</h2><ul><li>本轮是冻结的CLoSD组件适配和参考数据诊断，不是PhysDiff、Morph、CLAIMS完整复现，也不将不同控制检查点视为独立算法。</li>
<li>旧有效输出仅通过当前清单、目标、噪声对应关系、完整参考数组及指标重算确认兼容。缺少最初执行时的请求绑定，不能追溯补造。早期协议文本未完整保存，当前协议和修订说明均保留。</li>
<li>物理精化、状态反馈、闭环数据扩增本身已有研究；有限次条件纠偏只有效果、归因与实质技术差异均获支持后，才能作为创新候选。</li>
<li>未通过人体门槛，未运行G1与下游训练，因此尚不能回答改善机器人模仿学习或跨条件泛化的问题。</li>
<li>所有人体原始/派生轨迹留在服务器；公开代码、数值、来源散列和权重散列，许可限制不通过上传动作数组绕过。</li></ul>
<h2 id="audit">独立复核</h2><details><summary>展开审计记录（同模型家族，暂定审核）</summary><pre>'''+audit_text+'''</pre></details>
<h2>文件与参考</h2><p><a href="per_attempt.csv">每方法逐次回放</a> · <a href="all_construction_and_replay_attempts.csv">全部构建与回放尝试</a> ·
<a href="attempt_register.json">尝试登记与散列</a> · <a href="resource_summary.json">成本明细</a> · <a href="versions.json">版本</a> · <a href="protocol_snapshot.md">协议快照</a> · <a href="progression_gate.json">后续阶段门槛</a></p>
<p>相关路线：<a href="https://guytevet.github.io/CLoSD-page/">CLoSD：扩散与执行状态反馈</a>；
<a href="https://xbpeng.github.io/projects/PARC/index.html">PARC：物理动作与数据构建</a>；
<a href="https://github.com/WesleyXu224/CLAIMS_Official">CLAIMS：迭代闭环合成与控制扩展</a>。</p>
<p><a href="https://github.com/Pendragon08/conditional-human-motion-generation">现有GitHub项目</a></p></body></html>'''
    (out/'experiment_report.html').write_text(doc,encoding='utf8')
    print('PHYSICS_GUIDED_PUBLISH_COMPLETE',budget['total_gpu_hours'],budget['first_attempts'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.output).resolve())
