"""Source-paired C2 comparisons; completion and candidate acceptance are not success."""
import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
import numpy as np

from physics_guided_com_lineage_v24_20261012 import file_hash, motion_hash

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/condition_feedback_formal_v28_20261012'
CONTROLS=('random_best','native_best','no_feedback','shuffled','planned_prefix',
          'tracking_only','c1','no_agreement','no_quality_guard','shuffled_agreement')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')


def main(out):
    out=Path(out)
    config=read(ROOT/'configs/physics_guided_20261011.json')
    version=read(out/'versions.json')
    assert file_hash(BASE/'summary.json')==version['c1_summary_sha256']
    arms=read(out/'arms.json'); baselines=read(BASE/'arms.json')
    origins={key:(out,key.split(':')[0]) for key in arms if key.split(':')[0] not in CONTROLS[:7]}
    for name in CONTROLS[:7]:
        original='proposed' if name=='c1' else name
        for seed in config['seeds']:
            key=name+':'+str(seed)
            arms[key]=baselines[original+':'+str(seed)]
            origins[key]=(BASE,original)
    save(out/'arms.json',arms)
    all_rows=[]; goal_by_seed={}; cases_by_seed={}
    for key,path in arms.items():
        name,seed=key.split(':'); package=Path(path)
        source,source_name=origins[key]
        meta=read(package/'metadata.json'); metrics=read(package/'metrics.json')
        receipt=read(package/'receipt.json'); binding=read(package/'lineage.json')
        assert receipt['status']=='complete' and receipt['cases']==len(metrics)==144
        for filename,digest in binding['outputs'].items():
            assert file_hash(package/filename)==digest
        chosen=source/'selected'/(source_name+'_s'+seed+'.json')
        selected=chosen.with_suffix('.npz'); selection=read(chosen); lineage=read(chosen.with_suffix('.lineage.json'))
        assert lineage['selection_json']==file_hash(chosen)
        assert lineage['selected_motion']==motion_hash(selected)==binding['request']['replay_motion']
        assert selection['construction_candidates']==3
        assert meta['original_goal_sha256']==selection['original_goal_sha256']
        assert goal_by_seed.setdefault(seed,meta['original_goal_sha256'])==meta['original_goal_sha256']
        assert cases_by_seed.setdefault(seed,meta['cases'])==meta['cases']
        assert [r['case_id'] for r in metrics]==[r['case_id'] for r in selection['choices']]
        for row,choice in zip(metrics,selection['choices']):
            row=dict(row,arm=name,construction_complete=choice['construction_complete'])
            row['joint_pass']=bool(row['joint_pass'] and choice['construction_complete'])
            all_rows.append(row)
    groups={name:[r for r in all_rows if r['arm']==name] for name in ('proposed',)+CONTROLS}
    assert all(len(rows)==432 and len({r['source_id'] for r in rows})==48 for rows in groups.values())
    summary={}
    numeric_keys=('endpoint_error_m','heading_error_rad','endpoint_time_error_s',
                  'tracking_mean_m','tracking_p95_m','surface_slip_proxy_fraction',
                  'penetration_fraction','jerk_p95_m_s3')
    for name,rows in groups.items():
        s=dict(attempts=len(rows),joint_pass=sum(r['joint_pass'] for r in rows),
            completed=sum(r['completed'] for r in rows),
            construction_completed=sum(r['construction_complete'] for r in rows),
            early_terminated=sum(r['terminated'] for r in rows),
            per_seed={str(seed):sum(r['joint_pass'] for r in rows if r['seed']==seed)/144
                for seed in config['seeds']},nominal_construction_calls_per_input=3,
            independent_replay_calls_per_input=1)
        s['joint_rate']=s['joint_pass']/432
        for key in numeric_keys:
            values=[r[key] for r in rows if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
            s[key]=float(np.mean(values)) if values else None
            s[key+'_measured_count']=len(values)
        summary[name]=s
    proposed=groups['proposed']; comparisons={}
    for name in CONTROLS:
        controls={(r['source_id'],r['case_id'],r['seed']):r for r in groups[name]}
        clusters={}
        family={}
        for row in proposed:
            sid=row['source_id']; family[sid]=row['family']
            clusters.setdefault(sid,[]).append(int(row['joint_pass'])-
                int(controls[(sid,row['case_id'],row['seed'])]['joint_pass']))
        values={sid:float(np.mean(v)) for sid,v in clusters.items()}
        strata=[np.asarray([v for sid,v in values.items() if family[sid]==f]) for f in config['families']]
        rng=np.random.RandomState(20261011)
        boot=[float(np.concatenate([rng.choice(v,len(v),replace=True) for v in strata]).mean())
            for _ in range(config['bootstrap_repetitions'])]
        comparisons[name]=dict(gain_pp=100*float(np.mean(list(values.values()))),
            ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),
            seed_gains_pp={str(seed):100*(summary['proposed']['per_seed'][str(seed)]-
                summary[name]['per_seed'][str(seed)]) for seed in config['seeds']},
            cluster_unit='original source_path',all_failures_in_denominator=True)
    numeric_checks={name:bool(c['gain_pp']>=config['min_gain_percentage_points'] and
        c['ci95_pp'][0]>0 and all(v>0 for v in c['seed_gains_pp'].values()))
        for name,c in comparisons.items()}
    strongest=max(CONTROLS,key=lambda name:summary[name]['joint_rate'])
    limits=config['base_thresholds']; p=summary['proposed']; b=summary[strongest]
    condition_checks={key:bool(p[key] is not None and b[key] is not None and
        p[key]<=b[key]*(1+limits['max_condition_regression_fraction'])+1e-6)
        for key in ('endpoint_error_m','heading_error_rad','endpoint_time_error_s')}
    semantic_path=out/'semantic_diversity_gate.json'
    semantic=read(semantic_path) if semantic_path.exists() else dict(status='not_verified')
    semantic_pass=False
    if semantic.get('status')=='pass' and semantic.get('weights_frozen') is True:
        assert semantic['arms_sha256']==file_hash(out/'arms.json')
        assert semantic['evaluator_weights_sha256']==version['text_motion_evaluator_weights_sha256']
        assert semantic['semantic_asset_hashes']==version['external_semantic_inputs']
        for path,digest in version['external_semantic_inputs'].items(): assert file_hash(path)==digest
        for key,path in arms.items():
            assert semantic['input_hashes'][key]=={n:file_hash(Path(path)/n)
                for n in ('trajectories.npz','metadata.json','metrics.json')}
        sp=semantic['per_arm']['proposed'];sb=semantic['per_arm'][strongest]
        semantic_pass=bool(sp['matching_distance']<=sb['matching_distance']*(1+limits['max_semantic_regression_fraction'])
            and sp['diversity']>=sb['diversity']*(1-limits['max_diversity_regression_fraction'])
            and sp['multimodality']>=sb['multimodality']*(1-limits['max_diversity_regression_fraction'])
            and all(semantic['per_seed'][name+':'+str(seed)]['attempts']==144
                for name in ('proposed',strongest) for seed in config['seeds']))
    gate=dict(pass_=all(numeric_checks.values()) and all(condition_checks.values()) and semantic_pass,
        numeric_checks_by_control=numeric_checks,condition_checks=condition_checks,
        semantic_diversity_pass=semantic_pass,strongest_equal_budget_baseline=strongest,
        confirmed_innovation=False,novelty='not_established',g1='not_run',downstream_training='not_run',
        evidence_scope='amended simulation comparison, shared C1 controls disclosed',
        reason='Every matched three-call control, three-seed direction, source CI and independent guards required')
    gate['pass']=gate.pop('pass_')
    save(out/'summary.json',dict(summary=summary,comparisons=comparisons,gate=gate))
    save(out/'progression_gate.json',gate)
    fields=['arm','seed','source_id','case_id','family','offset_m','construction_complete',
        'completed','terminated','joint_pass']+list(numeric_keys)
    with (out/'per_attempt.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_rows)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (name,s['attempts'],s['joint_pass'],round(100*s['joint_rate'],3),s['early_terminated']))+'</tr>'
        for name,s in summary.items())
    (out/'experiment_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>C2反馈可靠性与质量守护实验</title><style>body{font:16px/1.7 system-ui;margin:40px;max-width:1200px}'
        'table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:8px}</style>'
        '<h1>C2：反馈可靠性与质量守护</h1><p>候选机制，尚非论文创新。所有失败保留；重复调用计入三次预算。'
        'C1控制组来自已完成V28，实际共享调用不重复计算资源费用；不据此声称速度优势。</p>'
        '<p>已校准足部滑移代理并非真实接触真值；当前未开展G1、下游训练或实机测试。</p>'
        '<table><tr><th>方法</th><th>尝试</th><th>联合合格</th><th>合格率%</th><th>提前终止</th></tr>'
        +table+'</table><p>人体门槛：'+str(gate['pass'])+'</p></html>',encoding='utf8')
    print('C2_PAIRED_ANALYSIS_COMPLETE',json.dumps(gate),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.inside:
        import run_reset_alignment_checks_v4 as runtime
        os.execve(str(runtime.PYTHON),[str(runtime.PYTHON),'-u',str(Path(__file__).resolve()),
            '--output',args.output,'--inside'],runtime.environment())
    assert np.__version__=='1.23.5'
    main(args.output)
