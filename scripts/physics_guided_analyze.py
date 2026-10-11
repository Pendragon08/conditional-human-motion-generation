"""Paired source-clustered analysis, without inventing missing downstream runs."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from physics_guided_lineage import file_hash, motion_hash

ROOT=Path(__file__).resolve().parents[1]


def main(out):
    config=json.loads((ROOT/'configs/physics_guided_20261011.json').read_text())
    arms=json.loads((out/'arms.json').read_text())
    rows=[]
    for key,path in arms.items():
        arm,seed=key.split(':')
        package=json.loads((Path(path)/'metrics.json').read_text())
        choice_path=out/'selected'/(arm+'_s'+seed+'.json')
        choices=json.loads(choice_path.read_text())['choices']
        selection_lineage=json.loads(choice_path.with_suffix('.lineage.json').read_text())
        selected_npz=choice_path.with_suffix('.npz')
        assert selection_lineage['selected_motion']==motion_hash(selected_npz)
        assert selection_lineage['selection_json']==file_hash(choice_path)
        replay_lineage=json.loads((Path(path)/'lineage.json').read_text())
        assert replay_lineage['request']['replay_motion']==motion_hash(selected_npz)
        for name,digest in replay_lineage['outputs'].items(): assert file_hash(Path(path)/name)==digest
        assert [r['case_id'] for r in package]==[r['case_id'] for r in choices]
        for r,c in zip(package,choices):
            r['arm']=arm
            r['construction_complete']=c['construction_complete']
            r['joint_pass']=bool(r['joint_pass'] and c['construction_complete'])
            rows.append(r)
    groups={}
    for r in rows: groups.setdefault(r['arm'],[]).append(r)
    summary={}
    for arm,group in groups.items():
        complete=[r for r in group if r.get('completed')]
        def mean(key):
            vals=[r[key] for r in group if isinstance(r.get(key),(int,float)) and r[key] is not None]
            return float(np.mean(vals)) if vals else None
        summary[arm]=dict(attempts=len(group),sources=len({r['source_id'] for r in group}),
                         completed=sum(r['completed'] for r in group),
                         construction_completed=sum(r['construction_complete'] for r in group),
                         joint_pass=sum(r['joint_pass'] for r in group),
                         joint_rate=sum(r['joint_pass'] for r in group)/len(group),
                         endpoint_error_m=mean('endpoint_error_m'),heading_error_rad=mean('heading_error_rad'),
                         endpoint_time_error_s=mean('endpoint_time_error_s'),tracking_mean_m=mean('tracking_mean_m'),
                         slip_proxy_fraction=mean('surface_slip_proxy_fraction'),penetration_fraction=mean('penetration_fraction'),
                         per_seed={str(s):sum(r['joint_pass'] for r in group if r['seed']==s)/sum(r['seed']==s for r in group)
                                   for s in config['seeds']},
                         early_terminated=sum(r['terminated'] for r in group))
    proposed={}
    for r in groups['proposed']: proposed.setdefault(r['source_id'],[]).append(r)
    rng=np.random.RandomState(20261011)
    comparisons={}
    for baseline in ('ordinary','random_best','native_best','no_feedback','shuffled','planned_prefix','tracking_only'):
        control={(r['source_id'],r['case_id'],r['seed']):r for r in groups[baseline]}
        clusters={}
        for source,source_rows in proposed.items():
            values=[int(r['joint_pass'])-int(control[(source,r['case_id'],r['seed'])]['joint_pass']) for r in source_rows]
            clusters[source]=float(np.mean(values))
        family_clusters={f:np.array([v for k,v in clusters.items() if proposed[k][0]['family']==f]) for f in config['families']}
        samples=[]
        for _ in range(config['bootstrap_repetitions']):
            sampled=[rng.choice(v,size=len(v),replace=True) for v in family_clusters.values()]
            samples.append(float(np.concatenate(sampled).mean()))
        comparisons[baseline]=dict(gain_pp=100*np.mean(list(clusters.values())),
             ci95_pp=(100*np.quantile(samples,[.025,.975])).tolist(),cluster_unit='original MoCap source_path',
             seed_gains_pp={str(s):100*(summary['proposed']['per_seed'][str(s)]-summary[baseline]['per_seed'][str(s)])
                            for s in config['seeds']})
    equal_budget_arms=('random_best','native_best','no_feedback','shuffled','planned_prefix','tracking_only')
    strongest=max(equal_budget_arms,key=lambda name:summary[name]['joint_rate'])
    main=comparisons[strongest]
    # Gate against every equal-budget control, avoiding post-selection or
    # tuple-order tie-breaking advantages from an observed "strongest" arm.
    numeric_checks={name:bool(comparisons[name]['gain_pp']>=config['min_gain_percentage_points'] and
                     comparisons[name]['ci95_pp'][0]>0 and
                     all(v>0 for v in comparisons[name]['seed_gains_pp'].values())) for name in equal_budget_arms}
    numeric=all(numeric_checks.values())
    tolerance=config['base_thresholds']['max_condition_regression_fraction']
    condition_checks={k:dict(proposed=summary['proposed'][k],baseline=summary[strongest][k],
        pass_=summary['proposed'][k]<=summary[strongest][k]*(1+tolerance)+1e-6)
        for k in ('endpoint_error_m','heading_error_rad','endpoint_time_error_s')}
    conditions=all(c['pass_'] for c in condition_checks.values())
    # Matching-score/diversity checks need an independently invoked evaluator.
    # Their absence is not filled with placeholder zero or proxy text overlap.
    semantic_path=out/'semantic_diversity_gate.json'
    semantics=json.loads(semantic_path.read_text()) if semantic_path.exists() else {'status':'not_verified'}
    protected=False
    if semantics.get('status')=='pass' and semantics.get('weights_frozen') is True:
        assert semantics['arms_sha256']==file_hash(out/'arms.json')
        assert semantics['evaluator_weights_sha256']==json.loads((out/'versions.json').read_text())['text_motion_evaluator_weights_sha256']
        assert semantics['semantic_asset_hashes']==json.loads((out/'versions.json').read_text())['external_semantic_inputs']
        for name,digest in semantics['semantic_asset_hashes'].items(): assert file_hash(name)==digest
        for key,path in arms.items():
            assert semantics['input_hashes'][key]=={name:file_hash(Path(path)/name) for name in ('trajectories.npz','metadata.json','metrics.json')}
        semantic_arms=semantics.get('per_arm',{})
        if 'proposed' in semantic_arms and strongest in semantic_arms:
            p,b=semantic_arms['proposed'],semantic_arms[strongest]
            limits=config['base_thresholds']
            protected=(p['matching_distance']<=b['matching_distance']*(1+limits['max_semantic_regression_fraction']) and
                       p['diversity']>=b['diversity']*(1-limits['max_diversity_regression_fraction']) and
                       p['multimodality']>=b['multimodality']*(1-limits['max_diversity_regression_fraction']) and
                       all(semantics['per_seed'][name+':'+str(s)]['attempts']==144
                           for name in ('proposed',strongest) for s in config['seeds']))
    reasons=[]
    if not numeric: reasons.append('Joint-pass gain, confidence interval, or three-seed direction failed')
    if not conditions: reasons.append('Original-condition error non-regression failed')
    if not protected: reasons.append('Independent semantic/diversity non-regression has not passed')
    gate=dict(pass_=bool(numeric and protected and conditions),numeric_gate=bool(numeric),condition_checks=condition_checks,
              strongest_equal_budget_baseline=strongest,
              comparison=main,semantic_diversity_gate=semantics,reason='; '.join(reasons) or 'All human gates passed',
              g1='not_run',downstream_training='not_run',novelty='not_established',
              study_status='amended diagnostic experiment; reused pilot artifacts retrospectively verified',
              equal_budget_arms=list(equal_budget_arms),
              numeric_checks_by_equal_budget_arm=numeric_checks,
              conclusion=('human_gate_supported_technical_difference_not_established' if numeric and protected and conditions
                          else 'unsupported_under_fixed_amended_diagnostic_gate'))
    gate['pass']=gate.pop('pass_')
    (out/'progression_gate.json').write_text(json.dumps(gate,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,comparisons=comparisons,gate=gate),indent=2))
    fields=['arm','seed','case_id','clip_id','family','offset_m','construction_complete','completed','joint_pass',
            'endpoint_error_m','heading_error_rad','endpoint_time_error_s','tracking_mean_m','tracking_p95_m',
            'surface_slip_proxy_fraction','penetration_fraction','jerk_p95_m_s3','terminated']
    with (out/'per_attempt.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    tr=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [arm,s['attempts'],s['joint_pass'],
              '%.2f%%'%(100*s['joint_rate']),s['completed'],'%.3f'%s['endpoint_error_m'],s['early_terminated']])+'</tr>'
              for arm,s in summary.items())
    cr=''.join('<tr><td>'+name+'</td><td>%.2f</td><td>[%.2f, %.2f]</td><td>%s</td></tr>'%
               (c['gain_pp'],c['ci95_pp'][0],c['ci95_pp'][1],html.escape(str(c['seed_gains_pp']))) for name,c in comparisons.items())
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>物理引导条件动作生成实验</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 20px;color:#203040}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd4dd;padding:8px;text-align:left}
th{background:#eef4fa}.note{background:#fff4da;padding:18px;border-left:5px solid #c98200}a{color:#175ea2}
@media print{body{margin:0;max-width:none}table{break-inside:auto}tr{break-inside:avoid}}</style>
<h1>物理引导条件动作生成：对比与消融</h1><p>2026-10-11 · 冻结生成器与控制器 · 仅仿真</p>
<div class="note"><strong>阶段结论：</strong>'''+html.escape(gate['reason'])+'''。当前不能宣称论文创新成立。
G1和下游训练只有通过人体门槛才继续。</div>
<h2>任务与公平设置</h2><p>48个原始MoCap来源，跳跃/舞蹈各24个；三个世界目标偏移、三个生成种子，
每组432个首轮尝试。镜像与切片按原始source_path分组。本轮测试独立于本轮训练来源，但不保证未被发布权重预训练见过。
未提前终止不等于合格；构建失败、回放失败均保留分母。</p>
<h2>独立参考回放结果</h2><table><tr><th>方法</th><th>尝试</th><th>联合合格</th><th>合格率</th><th>回放完成</th>
<th>终点误差(m)</th><th>提前终止</th></tr>'''+tr+'''</table>
<h2>候选方法相对基线的提升</h2><table><tr><th>基线</th><th>提升(百分点)</th><th>按来源聚类95%CI</th><th>三种子提升</th></tr>'''+cr+'''</table>
<h2>消融解释</h2><p>random_best：相同候选预算下随机生成并筛选；native_best：使用实际执行前缀及坐标锚点；
proposed：实际执行前缀及实际锚点、相同噪声、有界条件误差修正；no_feedback：计算预算相同但不修正条件；shuffled：打乱反馈来源；
planned_prefix：同时替换为计划前缀和计划锚点；tracking_only：仅按跟踪表现选择。同一固定外部目标用于所有评价。</p>
<h2>证据边界</h2><ul><li>足部滑移使用碰撞表面与刚体接触力代理，非实测接触点真值。</li>
<li>原始运动、物理精化运动都重新提交固定跟踪器回放，未将精化输出直接当作执行成功。</li>
<li>主表为适配后的CLoSD组件实验，不是Morph、PhysDiff或CLAIMS完整复现。</li>
<li>没有音乐条件、接触力生成、真实机器人实验或未执行的下游训练分数。</li></ul>
<p>参考：<a href="https://guytevet.github.io/CLoSD-page/">CLoSD</a>、
<a href="https://xbpeng.github.io/projects/PARC/index.html">PARC</a>、
<a href="https://github.com/WesleyXu224/CLAIMS_Official">CLAIMS</a>。</p>
<p>完整配置、逐尝试结果与预算见同目录JSON/CSV；原始动作受上游许可约束，保留服务器。</p></html>'''
    (out/'experiment_report.html').write_text(doc,encoding='utf8')
    tex='\\section{物理引导条件动作生成的配对对比与消融实验}\n'
    tex+='本轮固定DiP生成模型与CLoSD跟踪器，采用原始MoCap来源分组的跳跃与舞蹈数据。'
    tex+='每方法432组输入，固定原始世界目标，精化参考经独立回放评价。\n'
    tex+='\\begin{itemize}\n'
    for arm,s in summary.items():
        tex+='\\item '+arm.replace('_',r'\_')+'：联合合格'+str(s['joint_pass'])+'/'+str(s['attempts'])+'。\n'
    tex+='\\end{itemize}\n'
    tex+='候选方法相对于最强等预算基线的联合合格率提升为 %.2f 个百分点，来源聚类95\\%%置信区间为[%.2f, %.2f]。\n'%(main['gain_pp'],*main['ci95_pp'])
    tex+='当前人体门槛'+('已通过' if gate['pass'] else '未通过')+'，不报告尚未执行的G1或下游训练结果。'
    tex+='足部滑移为碰撞表面几何与接触力代理；效果与创新性须分别判断。\n'
    (out/'overleaf_experiment_record.tex').write_text(tex,encoding='utf8')
    print('PHYSICS_GUIDED_ANALYSIS_COMPLETE',json.dumps(gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    main(Path(p.parse_args().output))
