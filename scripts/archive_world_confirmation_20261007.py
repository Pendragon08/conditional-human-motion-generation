"""Archive immutable confirmation metadata, never raw licensed motions/weights."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import shlex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/world_confirmation_20261007'
RAW = OUT / 'raw'


def load(name, default=None):
    f = RAW / name
    if not f.exists():
        return default
    text = f.read_text(encoding='utf-8').strip()
    if name.endswith('.state') and not text.startswith('{'):
        tokens = shlex.split(text)
        result = {'status': tokens[0], 'raw_text': text}
        for token in tokens[1:]:
            if '=' in token:
                key, value = token.split('=',1)
                result[key] = int(value) if value.lstrip('-').isdigit() else value
        return result
    return json.loads(text)


def main():
    manifest = load('selection_manifest.json')
    generation = load('generation.state')
    assert manifest['generated_count'] == 32 and manifest['selected_count'] == 11
    assert manifest['accepted_static_count'] + manifest['rejected_count'] == 32
    assert not manifest['physical_results_used_for_selection']
    state = load('queue.state', {'status': 'NOT_DISPATCHED', 'stage': 'confirmation_only'})
    fits = {seed: load(f'evaluate_seed{seed}_summary.json') for seed in [42,43,44]}
    fits = {seed: r for seed, r in fits.items() if r is not None}
    collects = {seed: load(f'collect_seed{seed}_summary.json') for seed in [42,43,44]}
    collects = {seed: r for seed, r in collects.items() if r is not None}
    for seed, r in fits.items():
        assert r['confirmation_only'] and r['PPO_updates'] == 0 and r['auxiliary_updates'] == 0
        assert r['confirmation_evaluations_per_seed'] == 1
    for r in collects.values():
        assert r['fresh_confirmation_references'] and r['diagnostic_ppo_updates'] == 0
        assert r['actual_transitions'] == 983040
    stages = {}
    witness = load('identity_v2_doc_witness.json', {})
    for seed in [42,43,44]:
        for mode in ['collect','evaluate']:
            receipt = load(f'{mode}_seed{seed}.state')
            if receipt is not None:
                stages[f'{mode}_seed{seed}'] = receipt
    result = {
        'status': state['status'], 'stage': state.get('stage'),
        'active_seed': state.get('seed'), 'formal_PPO_started': False,
        'PPO_updates': 0, 'auxiliary_updates': 0, 'innovation_validated': False,
        'generation_status': generation['status'],
        'generated':32, 'accepted_static':22, 'rejected_static':10,
        'selected':11, 'eligible_not_selected':11,
        'selected_original_ids': [r['original_id'] for r in manifest['selected']],
        'manifest_sha256': hashlib.sha256((RAW/'selection_manifest.json').read_bytes()).hexdigest(),
        'actual_confirmation_transitions': sum(r['actual_transitions'] for r in collects.values()),
        'planned_confirmation_transitions': 2949120,
        'actual_resets': sum(r['reset_count'] for r in collects.values()),
        'completed_confirmation_seeds': len(fits),
        'independent_audit': {'status': witness.get('status'),
            'exactly_reconstructed_window_rows': witness.get('all_window_rows_exactly_reconstructed'),
            'hashed_output_files': witness.get('hashed_output_file_count'),
            'passed': witness.get('all_three_seeds_clean_recovery_gate_passed',False)},
        'prediction_gate_passed': state.get('prediction_gate_passed', False),
        'recovery_domain_all_seed_gate': state.get('recovery_domain_all_seed_gate', False),
        'next': state.get('next'), 'stage_exit_receipts': stages,
        'preserved_transport_and_failed_initializer': ['transport_v1_doc_witness.json','transport_v3_doc_witness.json','v1_queue.state','v1_model_lock.json','failed_initializer_cleanup_termination.json'],
        'per_seed': {str(seed): {
            'domain_gates': r['domain_gates'], 'comparisons': r['comparisons'],
            'metrics': r['metrics'], 'window_counts': r['window_counts'],
            'formal_prediction_gate_passed': r['formal_prediction_gate_passed'],
            'collect_reset_count': collects[seed]['reset_count'],
            'postpush_reset_count': collects[seed]['disturbance_protocol']['reset_60_to_109_pushed_envs'],
        } for seed, r in fits.items()},
        'limitations': [
            'Static acceptance is an interface/kinematic screen, not physics feasibility.',
            'New references use the same Kick text/hint distribution; no cross-task semantic generalization.',
            'Root-relative prediction FK is not world-goal, control success or entity-contact accuracy.',
            'Window exclusion defines reset-free prediction labels. All simulation resets are separately reported.',
            'Confirmation predictions contain no PPO updates. A prediction PASS alone does not prove control benefit.',
            'Old development actual-child exit receipts are unverified; preserved in report25. New wrapper captures rc immediately.',
        ],
        'raw_hashes': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in RAW.iterdir() if f.is_file()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stamp = datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    rows = []
    for seed, r in fits.items():
        for domain in ['clean','recovery']:
            c = r['comparisons'][domain]
            row = [str(seed), domain, 'PASS' if r['domain_gates'][domain] else 'FAIL']
            for comparator in ['ridge_correct','ridge_state_only','residual_state_only','direct','residual_shuffled']:
                gain = c[comparator]
                ci = gain['relative_gain_95pct_CI']
                row.append(f"{100*gain['relative_gain']:.2f}% [{100*ci[0]:.2f}%, {100*ci[1]:.2f}%]")
            rows.append(row)
    headings=['seed','域','预注册门','相对Ridge-A','相对Ridge-0','相对Residual-0','相对Direct','相对Shuffled']
    mdtable='|'+'|'.join(headings)+'|\n|'+'|'.join(['---']*len(headings))+'|\n'
    mdtable+='\n'.join('|'+'|'.join(row)+'|' for row in rows) or '|待完成|—|—|—|—|—|—|—|'
    full = 'PASS' if result['prediction_gate_passed'] else 'FAIL' if state['status']=='COMPLETE' else '尚未完成'
    receipt_ok = len(stages)==6 and all(r.get('status')=='COMPLETE' and r.get('exit')==0 for r in stages.values())
    result['all_six_actual_child_exits_zero_verified_from_wrapper_receipts'] = receipt_ok
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=f'''# 世界模型＋PPO：新生成动作独立预测确认（2026-10-07）

北京时间 {stamp} 核对：**{state['status']}；预测门 {full}；正式PPO更新0，创新点尚未验证。**

开发阶段约22%的改善来自选择模型用的validation，不能当确认结果。本轮重新生成动作并在任何物理结果之前固定选择规则、11参考、六预测模型及归一化；仅执行一次确认，不更新模型或重新选择checkpoint。

## 生成与静态筛查

冻结官方OmniControl权重，原Kick文本/空间hint前32项，seed20261007，1次重复，1000扩散步，196帧/20fps＝9.8秒。原PyRoki retarget/refine源码逐字一致，不追加IK。32条中22条符合接口/限位条件，10条被官方重定向/整理拒绝；前11个接受ID用于确认，其他11条合格但未选，不计失败。选中ID：{result['selected_original_ids']}。manifest SHA256：{result['manifest_sha256']}。

筛查要求215帧有限27关节/根位置/四元数、四元数模误差≤0.001、官方URDF限位超出≤0.0001rad、grab_idx在范围内；按固定215帧对齐，减第20帧root平移并反旋yaw，关节不改，不时间扭曲，排除与原66条/先前接受候选同时qRMSE<0.01rad与rootRMSE<0.001m的近重复。独立见证复核22条、原/副本源码、所有选中哈希及CUDA/JAX实际计算。静态通过不等于动力学可执行。

## 真Isaac确认数据及比较

每seed42/43/44为2048环境×480控制步，固定原随机actor/std、奖励、PD与终止，仅参考换11新动作。半数clean、半数在step50..59受torso世界Y方向±80N/0.2秒推力。采集后六个冻结模型共享同一窗口，0辅助/0PPO更新。已核验新真实转移 {result['actual_confirmation_transitions']:,}/2,949,120、真实reset {result['actual_resets']:,}。

历史10/未来10同reference/episode；恢复窗step60..109必须属于施力开始episode，未来没有新推力。每reference/域至少256合法窗、最多固定随机抽1024窗，覆盖不足必须BLOCKED，不删除难动作。所有reset另报；有效预测窗改善不等于恢复成功。

主统计H10（0.2秒）root-frame URDF全身FK：每窗MSE→每reference内平均再开方→11参考RMSE等权平均，reference配对bootstrap10000次。下表是相对各对照的改善及95%CI；主统计与另存的聚合FK RMSE不能混算。clean/recovery分别要求比两个Ridge及Residual-0改善≥10%且CI下界>0，Direct误差比CI上界≤1.05，Shuffled改善CI下界>0。三个seed两个域均通过才能打开原预测门。

{mdtable}

## 运行回执与结论边界

独立确认完成种子{len(fits)}/3；六项实际子进程退出0回执完整：{receipt_ok}。新wrapper在timeout返回后立即记录rc；旧开发wrapper退出码缺陷单独保留于报告25，不把完成哨兵替代退出证据。

fresh独立见证状态：{witness.get('status','待完成')}；逐项重建{witness.get('all_window_rows_exactly_reconstructed',0):,}窗口，复核{witness.get('hashed_output_file_count',0)}输出哈希，模型/归一化/参考锁不变，bootstrap独立复算差异0。主回执results/world_confirmation_20261007/raw/identity_v2_doc_witness.json。

首个运输因CRLF阻塞、v1科学初始化因新参考身份guard二维/三维不符阻塞，均未采集控制转移或产生预测分数。失败清理卡住后精确SIGTERM保全，退出链0但sentinel缺失，不能当成功。原源文件/model_lock/state/独立见证均归档；v2只把身份concat改为stack及形状assert，采集/模型/指标不变，独立新输出目录。主统计文字在任何确认物理数据前更正，原文本保全，不是结果后换指标。

当前下一步：{state.get('next','确认待完成；程序FAILED时为接口阻塞，不能记成方法FAIL')}。若只恢复域通过，原nominal门仍FAIL，不改变阈值重复test。预测PASS仅允许准备控制消融，还需当前真实仿真本地可视Gate及五臂三seed匹配PPO、实际恢复/任务/跌倒指标，最后冻结MuJoCo验证。Kick无实体目标；ground contact标签不是打击接触，root相对FK也不证明世界空间目标。无跨动作类别语义结论。

代码/协议/结构化结果归档新私有仓库；原人体/重定向数组、权重、确认原始数据和逐窗结果保存在服务器，未上传许可资产。DreamControl/CLoSD教师基线报告保持独立。
'''
    (ROOT/'reports/26_World_model_PPO_new_reference_confirmation_20261007.md').write_text(report,encoding='utf-8')
    th=''.join(f'<th>{html.escape(x)}</th>' for x in headings)
    tr=''.join('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in row)+'</tr>' for row in rows)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>世界模型新动作独立确认</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#183249;background:#f1f5f9}}main{{max-width:1240px;margin:auto;padding:32px}}section{{background:white;padding:24px;border-radius:12px;margin:18px 0}}p,li{{line-height:1.8}}.status{{background:#e8eef8;padding:15px;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:12px 8px;border-bottom:1px solid #d8e4ed}}th{{background:#edf4fa}}.scroll{{overflow:auto}}a{{color:#1665a4}}</style><main>
<small>独立方法验证 · 北京时间 {stamp}</small><h1>世界模型＋PPO：新动作预测确认</h1><div class="status">{html.escape(state['status'])} · 原预测门 {full} · 正式PPO尚未启动</div>
<section><h2>新参考与冻结模型</h2><p>32条新的Kick生成动作，22条静态接受、10条官方重定向拒绝。按预注册ID规则选择11条，另外11条合格未选。原G1重定向，没有额外IK；筛查在物理结果之前完成。</p><p>三个seed各2048环境×480步，冻结原运动跟踪策略。六个预测器、归一化与checkpoint保持不变。实际确认转移 {result['actual_confirmation_transitions']:,}；reset {result['actual_resets']:,}；已完成种子 {len(fits)}/3。</p></section>
<section><h2>独立确认效果</h2><p>0.2秒后全身FK预测误差；正数表示相对对照改善，方括号为reference配对95%CI。干净与受推力恢复两个域分别判定，三个seed都必须满足预注册门。</p><div class="scroll"><table><tr>{th}</tr>{tr}</table></div><p>比较先对每reference的窗口MSE平均再开方，随后等权平均11个reference的RMSE。窗口只用于预测标签，不能把排除reset的窗口当作控制成功样本。</p></section>
<section><h2>结论与后续</h2><p>当前预测门：{full}。预测通过也不能说明PPO收益；仍需匹配PPO控制消融，报告真实恢复AUC、任务成功、跌倒与计算成本，再做冻结策略MuJoCo验证。</p><p>Kick无实体打击对象。地面接触不是打击接触，root相对误差不是世界目标误差。本阶段不验证跨动作语义。</p><p>实际子进程退出0回执完整：{receipt_ok}。开发阶段旧回执缺陷已单独保留，未重跑覆盖。</p></section>
<p><a href="World_model_PPO_残差开发验证.html">开发对照</a> · <a href="../reports/26_World_model_PPO_new_reference_confirmation_20261007.md">完整协议和结论</a> · <a href="../results/world_confirmation_20261007/summary.json">结构化结果</a></p></main></html>'''
    (ROOT/'media/World_model_PPO_新动作预测确认.html').write_text(page,encoding='utf-8')
    print('ARCHIVED_CONFIRMATION',state['status'],len(fits),result['actual_confirmation_transitions'],full)


if __name__ == '__main__':
    main()
