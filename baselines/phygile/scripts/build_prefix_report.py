"""Build an evidence-bounded report; archived results/reports remain untouched."""
from pathlib import Path
import collections
import hashlib
import html
import json
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/prefix_integrity'
run = OUT/'document_run'
pipeline = json.loads((run/'pipeline.json').read_text())
assert pipeline['status'] == 'COMPLETED'
assert all(x['exit_code'] == 0 for x in pipeline['stages'])
regression = json.loads((run/'regression/verification.json').read_text())
gate_path = run/'feedback_gate/verification.json'
# A failed transport/runtime stage is retained; an explicit later recovery may finish it.
if not gate_path.exists():
    gate_path = OUT/'feedback_gate_recovery/verification.json'
gate = json.loads(gate_path.read_text())
assert regression['status'] == 'PASSED_REGRESSION_CHECKS'
assert gate['status'] == 'COMPLETED_FEEDBACK_GATE_DIAGNOSTIC'
reasons = collections.Counter(reason for row in gate['records'] for reason in row['reasons'])
audit_path = OUT/'audit/EXPERIMENT_AUDIT.json'
audit_text = '独立实验审查尚未完成；此文件仅供审查，不是最终交付。'
if audit_path.exists():
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert str(audit.get('verdict',audit.get('overall_verdict',''))).lower() in ['pass','warn']
    assert not audit.get('blocking_defects')
    audit_text = '独立实验审查：'+str(audit.get('verdict',audit.get('overall_verdict')))+'。同模型家族的独立上下文审查，结论为临时接受；详见 `results/prefix_integrity/audit/EXPERIMENT_AUDIT.md`。'
text = f'''# PhyGile 课程修复与物理前缀推进报告

日期：2026-10-17。**本轮完成了课程路径的局部完整性修复、物理前缀滚动控制核心，以及实际仿真反馈的接受判断。完整 PhyGile 仍未复现。**

## 本轮实际完成

| 工作 | 实际证据 | 能说明什么 |
|---|---|---|
| 课程路径修复 | 3 个文件的独立补丁、{len(regression['checks'])} 项回归检查 | 异常数值不会变成零误差；恶化/震荡不算收敛；无效评估不能因超时升级；缺失文件/不足尝试不被忽略 |
| 修复副本的实际训练 | 32 环境×24 步×1 轮 PPO，退出码 0 | 该副本仍能运行训练；课程关闭，仅是运行验证 |
| 前缀滚动控制核心 | 两窗口人工状态机检查、拒绝/回滚/重试上限/末端约束检查 | 接受的实际状态成为下一段输入，拒绝片段不污染前缀 |
| 实际仿真反馈 | 固定 2000 轮策略、1 个训练片段、32 个首回合 | 真实运行了仿真，保存自动重置前的状态，并用新接受逻辑核查 |

本轮一轮 PPO 为另外的工程诊断，768 个环境转移，不计入此前主策略的 2000 轮。它不是继续提高主策略成功率的训练，也没有进行闭环适配。

## 实际反馈检查结果

固定首秒窗口 50 帧、阈值 0.1 m。该阈值为事先写入 `configs/prefix_diagnostic_plan.json` 的工程诊断设置，未确认为作者超参数。

- 接受 **{gate['accepted']}/{gate['episodes']}** 个窗口；拒绝 **{gate['rejected']}/{gate['episodes']}** 个。
- 拒绝原因计数：{dict(reasons)}。同一回合可能同时有多个原因，计数不能相加当作回合数。
- 不完整窗口或失败终止即拒绝；误差直接采用米制 MPJPE，没有按模型自身输出归一化。
- 接受数只是“此策略、此参考、此阈值”的诊断结果，不是文本生成成功率，也不等于动作本身的绝对物理可行性。

参考来自已有视频重建、GMR 重定向和 FK 处理的运动数据，不是真机实测。此次仿真使用数据集参考，没有生成器提出候选。采集到的状态为 **71D 原生状态**（根状态 13 + 关节角 29 + 关节速度 29），另存 14 个跟踪刚体位置；不能把它直接当作论文的 262D 特征。

## 修复范围与保留的限制

原始官方快照及旧结果不改写；`prepare_guarded_upstream.py` 生成单独的 `cache/guarded_upstream`，启动时核对 3 个修复文件哈希。补丁见 `patches/curriculum_integrity.patch`。所有新增逻辑均为本工程实现，不冒充作者发布。

对非有限误差采用立即停止的处理；对“收敛”采用连续相邻评估均非恶化且改善小于阈值的保守判据。这是明确的实现选择，未获作者确认。训练 EMA 仍不能代替独立验证；既有阈值、成功率是否参与决策、Runner 与 CurriculumManager 的规则一致性还需后续对齐。

两条不可靠的公开评估路径已**禁用**，而不是宣称修好：进程内评估有双重帧推进问题；子进程评估在自动重置后读取误差。完整课程评估器仍待补齐。此次没有用虚构难度标签启动完整课程。

`physics_prefix.py` 提供生成—模拟—拒绝/接续的状态机接口。它要求仿真适配器完整保存和恢复物理状态、控制器、命令及随机数状态；当前多段回滚只在人工状态机中验证。实际仿真仅采集一段参考，再离线做首秒接受判断。**在线多段生成—执行闭环、仿真回滚适配器、末端站立及闭环 PPO 均未验证。**

## 与原文相比仍缺什么

1. 与文本配对的机器人动作数据、作者一致或明确验证的 262D 编解码、坐标与 palm 语义映射。
2. 生成器在真实运动数据上的完整训练，以及 ASFO 镜像、对应的生成质量评价器。
3. 完整 Stage I/II GMT：有依据的课程标签、独立课程评估、全部专家与阶段切换、足够预算和多种子。
4. 把已训练生成器、262D 编解码和真实仿真回滚接起来，完成滚动 physics-prefix 与 GMT PPO 适配。
5. 原文同划分/同指标的主实验与消融，以及用户需要的仿真对比；真机部署按用户要求不做。

下一项关键工程工作是完整的重置前课程评估器和 262D 编解码。真实生成器训练仍受配对数据缺失限制。此前 2000 轮测试完整片段成功率 26/128（20.31%）保持为历史结果，本轮没有产生新的正式跟踪分数。

## 复跑与证据

- 运行约定：`docs/prefix_integrity_invocation.md`。
- 各阶段退出码和代码哈希：`results/prefix_integrity/document_run/`。
- 接受判断：`{gate_path.relative_to(ROOT).as_posix()}`。
- 独立文档运行验证：`results/prefix_integrity/document_witness/`。
- 大文件包含原始仿真记录与诊断检查点，发布时单独附带 SHA-256；不写进源码历史。

{audit_text}

方法来源：[PhyGile 原文 v3](https://arxiv.org/html/2603.19305v3)、[作者跟踪代码固定版本](https://github.com/Baojch/phygile_tracking/tree/d48621620f522f5c901ef378d57db8df92d2888d)。原文闭环阶段冻结生成器，对 GMT 做 PPO 适配；本轮没有把此阶段误写成已训练扩散生成器。
'''
report = ROOT/'reports/PhyGile课程修复与物理前缀推进报告.md'
report.write_bytes(text.encode('utf-8'))
style='body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.7 "Microsoft YaHei",sans-serif;color:#182434}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #cbd5e1;text-align:left}th{background:#edf3f9}code{word-break:break-all}a{color:#075ac1}'
page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>PhyGile 推进报告</title><style>'+style+'</style><body>'+markdown.markdown(text,extensions=['tables'])+'</body></html>'
report.with_suffix('.html').write_bytes(page.encode('utf-8'))
(OUT/'report_build.json').write_text(json.dumps({'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
    'html_sha256':hashlib.sha256(report.with_suffix('.html').read_bytes()).hexdigest(),
    'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print(report)
