"""Archive bounded development receipts, keeping licensed arrays/weights remote."""
from pathlib import Path
import hashlib
import html
import json
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/world_residual_20261007'
RAW=RESULT/'raw'

def main():
    queue=json.loads((RAW/'queue.state').read_text())
    assert not queue['formal_PPO_started'] and queue['retired_test_evaluations']==0
    fits={seed:json.loads(f.read_text()) for seed in [42,43,44]
          if (f:=RAW/f'fit_seed{seed}_summary.json').exists()}
    collect={seed:json.loads(f.read_text()) for seed in [42,43,44]
             if (f:=RAW/f'collect_seed{seed}_summary.json').exists()}
    probe=json.loads((RAW/'probe_summary.json').read_text())
    kernel=json.loads((RAW/'kernel_witness.json').read_text())
    for row in queue.get('completed',[]):
        path=RAW/f"fit_seed{row['seed']}_summary.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['summary_sha256']
    summaries=[]
    for seed,r in fits.items():
        assert r['retired_test_evaluations']==0 and r['development_only'] and not r['formal_prediction_gate_passed']
        summaries.append({'seed':seed,'development_triage':r['development_triage'],
            'H10_recovery_FK_mm':{model:1000*v['recovery']['10']['body_fk_rmse_m'] for model,v in r['metrics'].items()},
            'H10_clean_FK_mm':{model:1000*v['clean']['10']['body_fk_rmse_m'] for model,v in r['metrics'].items()},
            'development_comparisons':r['development_comparisons'],'selection':r['selection'],
            'new_transitions':r['fresh_actual_transitions'],'actual_resets':r['actual_reset_count']})
    result={'status':queue['status'],'queue_stage':queue.get('stage'),'active_seed':queue.get('seed'),
            'queue_updated_unix':queue['updated_unix'],'development_triage':queue.get('development_triage'),
            'formal_PPO_updates':0,'formal_prediction_gate_passed':False,'innovation_validated':False,
            'old_v1_prediction_FAIL_preserved':True,
            'development_wrapper_exit_codes':[0]*6,'independent_actual_child_exit_codes_verified':False,
            'actual_child_exit_receipt_limitation':'Executed wrapper captured rc after assignment; only wrapper0+sentinels+complete artifacts independently verified. No repeat.','old_test_retired_evaluations':0,
            'actual_unique_new_transitions':sum(c['actual_transitions'] for c in collect.values()),
            'planned_new_transitions':11796480,'completed_predictor_seed_count':len(fits),
            'probe_transitions_diagnostic':probe['actual_transitions'],'probe_real_resets':probe['reset_count'],
            'kernel_CUDA_witness':kernel,'rows':summaries,
            'limitations':['Actual development child exit codes not independently retained; wrapper return0, artifacts and sentinels verified.',
                           'Validation selects checkpoint, so improvements and CI are development diagnostics, not independent confirmation.',
                           'Force is scheduled once on torso, ground contact only; no entity strike or general robustness claim.',
                           'Reset-crossing/unmatched windows excluded for all predictors; this is not a control success denominator.',
                           'Only an encoder may later feed PPO; all formal causal control ablations and new sim-to-sim still pending.'],
            'raw_receipt_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in RAW.iterdir() if f.is_file()}}
    (RESULT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    rows=[];md=[]
    for r in summaries:
        metrics=r['H10_recovery_FK_mm'];gain=r['development_comparisons']['ridge_correct']
        ci=gain['development_reference_bootstrap_95pct_ci']
        row=f"<tr><td>{r['seed']}</td><td>{metrics['ridge_correct']:.2f}</td><td>{metrics['direct']:.2f}</td><td>{metrics['residual_correct']:.2f}</td><td>{metrics['residual_state_only']:.2f}</td><td>{metrics['residual_shuffled']:.2f}</td><td>{100*gain['development_relative_gain']:.1f}%</td></tr>"
        rows.append(row)
        md.append(f"|{r['seed']}|{metrics['ridge_correct']:.3f}|{metrics['direct']:.3f}|{metrics['residual_correct']:.3f}|{metrics['residual_state_only']:.3f}|{metrics['residual_shuffled']:.3f}|{100*gain['development_relative_gain']:.2f}% [{100*ci[0]:.2f}%,{100*ci[1]:.2f}%]|{r['development_triage']}|")
    stamp=datetime.fromtimestamp(queue['updated_unix'],timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    status='开发对照已完成' if queue['status']=='COMPLETE' else '开发对照运行中' if queue['status']=='RUNNING' else '开发对照需诊断'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>残差世界模型开发验证</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f4f6f9;color:#1d3045}}main{{max-width:1120px;margin:auto;padding:38px 28px}}h1{{font-size:32px}}.card{{background:#fff;border-radius:12px;padding:22px;margin:18px 0}}p,li{{line-height:1.8}}table{{border-collapse:collapse;width:100%}}th,td{{padding:13px 9px;border-bottom:1px solid #dde5ed;text-align:left}}th{{background:#edf3f8}}.badge{{display:inline-block;background:#fff1d9;color:#855410;padding:10px 16px;border-radius:8px}}.flow{{background:#e9f2fb;padding:18px;border-radius:8px}}a{{color:#176baa}}small{{color:#597085}}</style>
<main><small>独立方法开发记录 · 北京时间 {stamp} · 非论文基线报告</small><h1>残差世界模型＋PPO：开发验证</h1><span class="badge">{status} · 正式PPO尚未启动</span>
<div class="card"><p>上一轮三个种子的原GRU预测未优于Ridge，失败结果完整保留。这一轮固定强线性基线，学习它未解释的残差，检查真实Isaac扰动恢复数据是否提供可用的非线性信号。</p><p>已完成新采集 {result['actual_unique_new_transitions']:,} / 11,796,480 转移；已完成预测拟合 {len(fits)} / 3 个种子。正式PPO更新：0。</p></div>
<div class="card"><div class="flow">扩散参考动作 → 原G1重定向 → 真Isaac执行与扰动 → Ridge＋GRU残差开发 → 新动作集单次确认 → 五臂真实PPO → 冻结MuJoCo验证</div><p>世界模型学习物理响应，参考动作仍由扩散模型提供。actor只读取已有本体历史；未来动作只作监督预测条件。contact是地面接触，Kick没有实体目标。</p></div>
<h2>扰动恢复预测效果</h2><div class="card"><p>H10＝未来0.2秒，根坐标全身FK误差，mm，越低越好。比较同一数据、同一窗口和相同辅助更新预算。mm列是全窗口聚合RMSE；改善及CI则先逐reference计算窗口MSE均值的开方，再对reference等权平均，不由此表mm列直接相除。</p><table><tr><th>seed</th><th>Ridge</th><th>原GRU</th><th>残差GRU</th><th>未来动作置零</th><th>错配残差</th><th>对Ridge改善</th></tr>{''.join(rows)}</table><p>此处validation也用于选择checkpoint，表中结果只能作为开发依据。旧11条test已退休；没有使用它们重新证明改善。</p></div>
<h2>能说明什么</h2><div class="card"><ul><li>真实扰动数据、文件哈希和CUDA更新有独立核验。开发包装器返回0，但实际子进程退出码未独立留存；新确认已修正退出回执。模型不使用未知未来施力。</li><li>错误残差监督精确置换四折OOF残差，contact标签随同置换；模型可选择update0保留线性起点。</li><li>开发失败就停止当前残差方案，不调整门槛追结果；开发有改善也必须新确认集通过。</li><li>尚未完成真实PPO控制消融、扰动恢复收益、实体打击或新的sim-to-sim；不能宣称创新验证成功。</li></ul></div>
<p><a href="World_model_PPO_阶段验证.html">上一轮完整否定结果</a> · <a href="../reports/25_World_model_PPO_residual_development_20261007.md">方法与运行记录</a> · <a href="../results/world_residual_20261007/summary.json">结构化结果</a></p></main></html>'''
    (ROOT/'media/World_model_PPO_残差开发验证.html').write_text(body,encoding='utf-8')
    report=ROOT/'reports/25_World_model_PPO_residual_development_20261007.md'
    original=report.read_text(encoding='utf-8').split('\n<!-- AUTO_DEVELOPMENT_STATUS -->')[0]
    progress=f"\n<!-- AUTO_DEVELOPMENT_STATUS -->\n\n## 自动核对状态（北京时间 {stamp}）\n\n{status}；stage={queue.get('stage')}。新采集去重转移{result['actual_unique_new_transitions']:,}，完成拟合{len(fits)}/3种子。正式PPO与退休test调用均为0。\n\n|seed|Ridge mm|Direct mm|Residual mm|Future-action-zero mm|Shuffled mm|对Ridge开发改善 [CI]|开发triage|\n|---|---|---|---|---|---|---|---|\n"+'\n'.join(md)+'\n\nvalidation选择checkpoint，本表不是独立确认。mm是全窗口聚合RMSE，改善/CI则逐reference的窗口MSE均值开方后等权平均，不能直接混算。\n'
    report.write_text(original+progress,encoding='utf-8')
    print('WORLD_RESIDUAL_LOCAL_ARCHIVE',queue['status'],len(fits),result['actual_unique_new_transitions'])

if __name__=='__main__':main()
