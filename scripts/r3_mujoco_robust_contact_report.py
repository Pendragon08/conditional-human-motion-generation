"""Build the evidence report and local video gallery from completed frozen tests."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'nominal': '无扰动', 'nominal_mass_constants_refresh': '质量测试配套初始化对照',
    'friction_low': '地面摩擦 0.5', 'friction_high': '地面摩擦 1.5',
    'mass_low': '质量 −10%', 'mass_high': '质量 +10%', 'strength_low': '力矩上限 −10%',
    'push_left_20': '左向推力 20 N × 0.2 s', 'push_right_20': '右向推力 20 N × 0.2 s',
    'push_left_40': '左向推力 40 N × 0.2 s', 'push_right_40': '右向推力 40 N × 0.2 s',
    'pad_nominal': '实体靶位于原目标', 'pad_y_plus_5cm': '实体靶横移 +5 cm',
    'pad_y_minus_5cm': '实体靶横移 −5 cm',
}


def fraction(c, key):
    return f"{c[key]}/{c['n']}（{100*c[key]/c['n']:.1f}%）"


def main():
    result = json.loads((ROOT/'results/mujoco_robust_contact/summary.json').read_text())
    assert result['status'] == 'COMPLETE'
    tables = {task: {c['condition']: c for c in data['conditions']} for task, data in result['tasks'].items()}
    lines = ['# Kick/Punch 冻结策略：MuJoCo 扰动与实体接触验证', '',
        '2026-10-01。完成 2,324 次测试，4 个独立分片均正常退出。保持原论文 Kick、Punch 基线策略冻结；没有训练、额外 IK、参考轨迹修改或按结果调整阈值。', '',
        '本报告检验两项已有策略在规定扰动和可碰撞目标下的表现，不是创新点验证，也不是实机打击验收。', '',
        '首轮 V1 发现仅降低地面摩擦时，MuJoCo 取地面和脚部参数的较大值，实际摩擦仍为 1.0。已保留 V1 的代码、记录和视频，修正双方参数及条件间质量缓存恢复后完整重跑 V2。本报告仅使用 V2；实际每个地面接触的摩擦都已核验，未改变策略或成绩阈值。', '',
        '## 结果说明', '',
        f"- 原条件下稳定完成：Kick {fraction(tables['Kick']['nominal'],'stable_goal_success')}；Punch {fraction(tables['Punch']['nominal'],'stable_goal_success')}。",
        f"- 40 N 左／右推力下，Kick 稳定完成分别为 {fraction(tables['Kick']['push_left_40'],'stable_goal_success')}／{fraction(tables['Kick']['push_right_40'],'stable_goal_success')}；Punch 为 {fraction(tables['Punch']['push_left_40'],'stable_goal_success')}／{fraction(tables['Punch']['push_right_40'],'stable_goal_success')}。一个方向偶尔提升命中不代表普遍鲁棒，应结合配对失败与摔倒看。",
        f"- 接触摩擦增至 1.5 时，Kick／Punch 稳定完成为 {fraction(tables['Kick']['friction_high'],'stable_goal_success')}／{fraction(tables['Punch']['friction_high'],'stable_goal_success')}。参数变化会明显影响站立，尚不能判定鲁棒性通过。",
        f"- 靶位于原目标时，Kick 合格打击 {fraction(tables['Kick']['pad_nominal'],'valid_strike_10n')}，打击且站稳 {fraction(tables['Kick']['pad_nominal'],'stable_strike_success')}；Punch 对应 {fraction(tables['Punch']['pad_nominal'],'valid_strike_10n')}、{fraction(tables['Punch']['pad_nominal'],'stable_strike_success')}。到达虚拟目标并不保证实体接触后仍能站稳。", '',
        '## 测试口径', '',
        '每个条件使用全部原参考：Kick 66 条、Punch 100 条。每条从零时刻独立开始，最多 10 秒；摔倒即终止，不自动重置。失败初始化保留在原始分母。表中“稳定完成”指在原攻击事件窗到达虚拟目标并完整 10 秒不摔倒。它与官方 Isaac 1000 环境自动重置的成功率口径不同。', '',
        '这些动作来自已有训练参考库，不是独立动作留出集。当前检验跨仿真和环境条件变化，不能据此声称生成器对未见文本／动作泛化，或证明数据增强创新有效。', '',
        '推力沿世界坐标左右方向施加到骨盆，持续 0.2 秒，开始于参考攻击事件前 0.2 秒（最早 0.2 秒）。20/40 N 对应计划冲量 4/8 N·s。记录实际冲量；提前摔倒而未受到推力的输入仍计入分母。质量、惯量同步缩放 ±10%，力矩上限降低 10%，地面摩擦取 0.5/1.0/1.5。', '',
        '质量改变须刷新 MuJoCo 的常量；这个刷新也影响运行时设置 armature 后的缓存。故质量两项与“质量测试配套初始化对照”比较，其他扰动与“无扰动”比较。不能把刷新差异全部归因于质量变化。', '',
        '## 扰动与站立结果', '',
        '| 任务／条件 | 到达虚拟目标 | 稳定完成 | 摔倒 | 较配套对照损失／新增稳定成功 |',
        '|---|---:|---:|---:|---:|']
    stress_html = ''
    contact_html = ''
    for task, data in result['tasks'].items():
        for c in data['conditions']:
            if c['condition'].startswith('pad_'):
                continue
            cells = [f"{task}／{LABELS[c['condition']]}", fraction(c,'virtual_goal_reached'),
                fraction(c,'stable_goal_success'), fraction(c,'fallen'),
                f"{c['paired_stable_goal_lost']}／{c['paired_stable_goal_gained']}"]
            lines.append('| '+' | '.join(cells)+' |')
            stress_html += '<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>'
    lines += ['', '## 实体接触结果', '',
        '使用固定、可碰撞的 24 × 24 cm 打击靶，厚 5 cm；正面位于原虚拟目标，并额外横移 ±5 cm。靶有真实碰撞响应，但没有沙袋摆动或对手动力学。靶的位移不传给冻结策略；这是位置误差测试，不是对靶自适应控制。', '',
        '分层计分：任意部位碰靶 → 指定右脚／右手碰靶 → 攻击时间窗内碰靶 → 合格打击 → 合格打击后保持站立。合格打击须在事件 ±0.3 秒内、正确手脚、从靶正面接触、法向力至少 10 N，且步前名义打击点的朝靶速度至少 0.1 m/s。这个速度是代理，非精确接触点冲击速度。另报告 5/20 N 敏感性；阈值是预设工程口径，不是已发表标准 benchmark。', '',
        '| 任务／靶位置 | 任意接触 | 正确手脚接触 | 时间窗内正确接触 | 合格打击 ≥10 N | 合格打击且站稳 | 摔倒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for task, data in result['tasks'].items():
        for c in data['conditions']:
            if not c['condition'].startswith('pad_'):
                continue
            cells = [f"{task}／{LABELS[c['condition']]}"] + [fraction(c,k) for k in
                ['any_contact','correct_effector_contact','window_correct_contact','valid_strike_10n','stable_strike_success','fallen']]
            lines.append('| '+' | '.join(cells)+' |')
            contact_html += '<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>'
    lines += ['', '### 力阈值与冲量', '',
        '| 任务／靶位置 | 合格打击 5／10／20 N（次数） | 正确接触样本数 | 峰值力中位数（N） | 总法向冲量中位数（N·s） |',
        '|---|---:|---:|---:|---:|']
    for task, data in result['tasks'].items():
        for c in data['conditions']:
            if c['condition'].startswith('pad_'):
                force = c['correct_contact_peak_force_median_n']
                impulse = c['correct_contact_impulse_median_ns']
                lines.append(f"| {task}／{LABELS[c['condition']]} | {c['valid_strike_5n']}／{c['valid_strike_10n']}／{c['valid_strike_20n']} | {c['force_stat_denominator_correct_contact']} | {force:.1f} | {impulse:.3f} |" if force is not None else f"| {task}／{LABELS[c['condition']]} | 0／0／0 | 0 | 无 | 无 |")
    lines += ['', '力和冲量只来自正确手脚发生接触的样本，未接触样本不混入条件中位数。累计冲量可能包含多次、较长时间接触，不能直接解释为单次拳脚冲击强度。MuJoCo 求解器、靶刚性和碰撞几何影响峰值，不能换算为实机打击力。', '',
        '## 证据与限制', '',
        '- 增加靶但关闭其碰撞后，与旧迁移适配器的 30 步运动、观测与动作完全一致；两项真实 Isaac 快照接口误差仍约 10⁻⁷。正、负碰撞检测夹具通过。刻意重叠的正夹具不计入策略成绩。',
        '- 运行时采用 CPU MuJoCo 3.2.3 物理、CUDA 冻结策略推理与 NVIDIA EGL 渲染。新增测试不改变原策略和原迁移适配器。',
        '- 保留上游碰撞几何。Kick 脚部仅有小型脚底球形碰撞点；手部使用原手指／腕部几何。这是现有 MJCF 的实体接触验证，尚不是精细足背、拳套或人体接触模型。',
        '- 固定靶不是可移动沙袋或机器人对手；未验证实际人体安全性、摩擦组合扰动、多个训练随机种子或无限扰动强度。不能宣布普遍鲁棒性、实机可部署或新算法有效。', '',
        '### 名义结果复核', '']
    for task, data in result['tasks'].items():
        lines.append(f"- {task}：新无扰动测试与旧首轮结果的逐条差异编号：{data['nominal_disagreements_with_previous']}。")
        if data['nominal_disagreements_with_previous']:
            lines.append('  上述长期执行判定差异尚需排查；30 步一致不能证明整段轨迹数值完全一致。本报告各扰动与本轮同包装的名义对照配对，不混用上轮逐条判定。')
    lines += ['', '### 文件', '',
        '- [真实 MuJoCo 场景与视频](../media/mujoco_robust_contact/results.html)。参考编号 0/1 和录制条件在运行前固定，包含失败片段。',
        '- `results/mujoco_robust_contact/summary.json`：全部条件汇总、参数和模型／参考／代码哈希、12 段视频 SHA-256 与 ffprobe 时长。',
        '- `results/mujoco_robust_contact/{Kick,Punch}_shard_{0,1}.json`：2,324 条原始测试，包含接触力、冲量、初始穿透、受扰情况及失败。',
        '- `configs/mujoco_robustness_contact_protocol.json`：运行前固定的协议。执行说明：`docs/mujoco_robust_contact_execution.md`。',
        '- 四项状态与完整日志保存在服务器新项目 `artifacts/monitoring/r3_mujoco_robust_contact_v2_<Task>_shard<0,1>.*`。均应为 COMPLETE、退出 0。',
        '- V1 初始诊断位于服务器 `artifacts/mujoco_robust_contact_diagnostic_v1`，不能混入最终鲁棒性结果。',
        '- 未复制或上传大型参考库、机器人源资产及受许可原始数据。', '',
        '接触读取依据：[MuJoCo mj_contactForce](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-contactforce)，固定 mocap 物体依据：[MuJoCo XML body](https://mujoco.readthedocs.io/en/3.2.3/XMLreference.html#body-mocap)。原控制策略与机器人模型依据：[DreamControl 官方代码](https://github.com/GenRobo/DreamControl)。']
    (ROOT/'reports/11_Kick_Punch_robustness_contact_20261001.md').write_text('\n'.join(lines), encoding='utf-8')
    cards = ''
    for v in result['videos']:
        name = Path(v['path']).name
        task, rest = name.split('_',1)
        condition, ref = rest.rsplit('_ref_',1)
        caption = f"{task} · {LABELS[condition]} · 固定参考 {int(ref[:-4])}"
        cards += f'<article><h3>{html.escape(caption)}</h3><video controls preload="metadata" src="{name}"></video><p>{float(v["ffprobe"]["format"]["duration"]):.2f} 秒；真实 MuJoCo 物理执行，非轨迹回放。</p></article>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kick/Punch 扰动与实体接触</title><style>body{{font:16px/1.65 system-ui,sans-serif;background:#f4f6fa;color:#192739;margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}h1{{font-size:30px}}table{{border-collapse:collapse;background:white;font-size:14px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}article{{background:white;border-radius:10px;padding:16px}}video{{width:100%;background:#111}}.note{{background:#fff3d9;padding:16px;border-left:4px solid #e3a02f}}p{{max-width:1050px}}</style><main>
<h1>Kick / Punch：扰动鲁棒性与实体接触</h1><p>2026-10-01 · 2,324 次冻结策略测试 · 14 条件 · 每条参考最多 10 秒，无自动重置。</p>
<p class="note">表中的“稳定完成”是到达原虚拟目标后完整 10 秒不摔倒。实体打击另行统计：正确手脚、攻击时段、正面接触、≥10 N、朝靶运动。固定靶有真实碰撞响应，但不是实机、沙袋或对手。未重新训练。</p>
<h2>扰动测试</h2><p>推力 20/40 N 持续 0.2 秒；摩擦 0.5/1.5；质量 ±10%；力矩上限 −10%。质量项与配套常量刷新对照比较。</p><div class="scroll"><table><thead><tr><th>任务／条件</th><th>虚拟目标</th><th>稳定完成</th><th>摔倒</th><th>损失／新增</th></tr></thead><tbody>{stress_html}</tbody></table></div>
<h2>实体打击靶</h2><p>固定 24 × 24 × 5 cm 靶，原目标或横移 ±5 cm。接触由物理引擎检测；峰值力不可解释为实机力。脚部保留原小型脚底碰撞点。</p><div class="scroll"><table><thead><tr><th>任务／靶</th><th>任意接触</th><th>正确手脚</th><th>攻击窗</th><th>合格打击</th><th>打击且站稳</th><th>摔倒</th></tr></thead><tbody>{contact_html}</tbody></table></div>
<h2>固定样本视频</h2><p>编号和条件在测试前选定，保留失败，不能用两段视频替代总体统计。橙红色方形是实体靶；无靶视频的橙色标记是虚拟目标。</p><div class="grid">{cards}</div>
<p><a href="../../reports/11_Kick_Punch_robustness_contact_20261001.md">详细报告</a> · <a href="../../results/mujoco_robust_contact/summary.json">完整指标与视频校验</a></p></main></html>'''
    (ROOT/'media/mujoco_robust_contact/results.html').write_text(page, encoding='utf-8')
    print('REPORT_GALLERY_COMPLETE')


if __name__ == '__main__':
    main()
