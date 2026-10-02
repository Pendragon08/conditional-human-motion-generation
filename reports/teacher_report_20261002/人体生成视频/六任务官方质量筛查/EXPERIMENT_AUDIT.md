# Experiment Audit Report

Date: 2026-10-02

Auditor: GPT-5.6-Sol ultra, fresh reviewer; same-family, provisional.

Overall verdict: PASS after correction (initial WARN preserved in review trace).

Scope: six-task self-supervised geometric proxies; audit PASS is not a physical-quality Gate or dynamics certificate.

## 复核结论：PASS

上次全部整改项均已解决。确定性检查已独立复算；语义审计仍按技能要求标为同模型家族、暂定结论。

### A. Ground-truth provenance — PASS

- 输入仍明确为官方发布的生成动作，不是数据集 GT；筛查仍标为 `self_supervised_proxy`：`audit_six_official_humans_20261002.py:26-31,121-124`。
- 新增来源固定脚本从官方 GitHub `main` ref 读取 commit，再读取该 commit 下的 LFS pointer：`pin_six_human_source_identity_20261002.py:6-20`。
- `source_identity.json:3-15` 等六项记录固定 commit `ec28b508313d6837de52c68247fbc8cf81fe4935`、仓库路径、LFS OID、字节数和不可变 pointer URL。
- 独立在线核验结果：

  - GitHub API 当前 `main` 确为该 commit；
  - commit API 返回同一提交；
  - 六个固定 URL 的 pointer 文本逐字匹配；
  - 六个 LFS OID、manifest SHA、本地文件 SHA、当前官方媒体文件 SHA 和字节数全部相同。

- 原 manifest SHA 仍为此前的 `a45a7d…db6e`：`summary.json:284`，未被修改。

### B. Score normalization — PASS

- 骨段指标仍只除以该样本、该骨段自身中位数，并明确不是解剖 GT：`audit...py:58-64`。
- 原始中位长度、绝对米制偏差和相对偏差均继续保存：`audit...py:81-83`；代表行见 `per_sample.json:58-83`。
- HTML 保留“自身一致性代理、无外部骨长真值”的限定：`实验结果汇报.html:26,30`。
- 没有把该归一化结果改写成准确率或物理通过率。

### C. Existence and numbers — PASS

- 独立从六个 `.npy` 重构全部 342×34 = 11,628 个 JSON 字段，零不一致。
- CSV 有 342 行、34 列，所有带类型值与 JSON 完全一致。
- 新增原始负高度计数独立复算为：

  - Kick 24/100
  - Punch 3/100
  - Jump 0/13
  - Sit 9/20
  - Pick 43/100
  - Bimanual-Pick 3/9
  - 合计 82/342

  与 `summary.json:12-135`、`实验结果汇报.html:32-33` 完全一致。

- 全局最小脚点高度为 −6.916 mm；没有任何样本触发同脚 `y<−0.03 m` 连续 3 帧。
- 原有滑动、骨段、快速位移、腾空及敏感性计数全部保持不变。
- 196 帧仍正确区分为 9.8 秒名义视频时长和 9.75 秒首末采样跨度：`audit...py:121-123`；`per_sample.json:7-10`。
- HTML 所链接的 summary、thresholds、JSON、CSV、README、source identity 和自检凭据副本均存在，并与结果目录 SHA 相同。

### D. Dead code and execution evidence — PASS

- 新增 `any_foot_y_below_zero` 实际计算、汇总并写入逐样本结果：`audit...py:67-87,126-131`。
- 它没有混入原 `geometry_warning_any`，因此不会改变旧警告定义：`audit...py:77`。
- 新增交替骨段夹具明确验证逐骨持续性：`audit...py:90-107`。
- `metric_self_check_execution.json:3-13` 记录命令、退出码、stdout、stderr 和脚本 SHA。
- 我原样重跑了只读 `--self-check` 分支：

  - 退出码、stdout、stderr 与凭据逐字一致；
  - 脚本 SHA 与凭据和 `summary.json:283` 一致；
  - 结果目录所有文件的大小、mtime 和 SHA 在执行前后均未变化。

- 实际数据仍保留关键反例：Jump sample 7 若合并双脚会产生错误的 3 间隔提示，但逐脚最长段为 `[2,1]`。实际数据未发现交替骨段导致的错误提示；独立合成夹具中逐骨最长为 1，错误合并则为 6。

### E. Scope and threshold semantics — PASS

上次警告已完整修正：

- 主表现在写明“同脚低于 −3 cm 连续≥3帧”：`实验结果汇报.html:20`；README 同样修正：`README.md:5-12`。
- 文案明确说明“未触发阈值”不等于从未低于 `y=0`：`实验结果汇报.html:19,24`。
- 新表公开 82/342 的零高度越界、各任务最小原始值和阈值计数：`实验结果汇报.html:32-33`；README 同样披露：`README.md:16`。
- 阈值仍明确为未做人评校准的探索性代理：`实验结果汇报.html:30`。
- 论文段落现在明确写为“训练参考轨迹”筛选，并排除评估 rollout 和人类自然度筛选解释：`实验结果汇报.html:40-49`。
- “其他任务人工选 1 条”的选择准则未披露也已明确写出：`实验结果汇报.html:49`；`README.md:24`。

### F. Evaluation classification — PASS

- 本地六任务筛查：`self_supervised_proxy`，作用于官方发布的模型生成动作。
- 新增 `y<0` 是原始坐标描述项，不是新的失败标签。
- DreamControl 表 XI 仍正确归类为论文报告的、RL 前训练参考启发式筛选。
- 人工选一条属于训练数据人工选择，不能据此称为正式 `human_eval`。
- 本轮仍没有 `real_gt`、动力学仿真验收或人类评分。

### 剩余限制

这些限制均已在材料中正确披露，不构成审计缺陷：

- 阈值未经人工质量标签校准，只支持描述性代理计数。
- 342 条没有接触力、力矩、摩擦、平衡、碰撞、语义或人类自然度认证。
- 表 XI 数值来自论文，未在本轮重跑。
- `source_identity.json` 固定的是审计时观察到的官方 commit 和 LFS 内容；未来 `main` 变化不会影响该固定身份。

