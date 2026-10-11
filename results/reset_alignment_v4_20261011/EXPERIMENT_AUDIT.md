# Reset-cache v4 实验完整性审计

**日期**：2026-10-11  
**审计者**：fresh read-only Codex subagent  
**审查独立性**：same-family  
**接受状态**：provisional  
**生产证据**：只读；未改生产源码、阈值或结果，未启动 GPU 作业，未下载许可原始数组，未接触 Overleaf  
**远端根目录**：`/data/code/ouyanghao/oyh/conditional-human-motion-generation`

## 总体判定

**总体完整性：WARN。测量/效能就绪度：FAIL。**

冻结工件、216 次尝试、执行请求、输出哈希、七个导出张量的 reset/first-action 一致性、显式时钟和逐行指标重算均能独立复核。没有发现伪造结果、用预测自身统计量归一化分数、虚构文件或未调用的主指标路径。

但该结论只支持“七个导出张量与上游 reset cache 的有限一致性诊断”。它不支持完整 PhysX 状态同一性。更严重的是，在运动学状态逐位相同的配对轨迹上，net-contact force 及其导出的足部 support 标签不重复，直接否定了 force-gated support/slip 代理的比较可靠性。任何物理门、模式优劣或效能声明必须停止，直至新的同步采样诊断与代理校准通过。

结果文件自己也作出了正确限制：状态仍是 `COLLECTION_COMPLETE_REQUIRES_INDEPENDENT_CHECK`（`results/reset_alignment_v4_20261011/state.json:2`）；摘要明确写明不是 efficacy 实验、不得与历史结果合并、物理代理和阈值仍需构念校准（`results/reset_alignment_v4_20261011/measurement_repair_summary.json:641-642`）。

## 审计范围与独立复核

审计覆盖指定的六个脚本、两个配置、说明文档、顶层状态/版本/预算/摘要、八个执行请求、八个子目录的 metadata/metrics/receipt/lineage，以及服务器本地的 `initial_states.npz` 和 `trajectories.npz`。原始 NPZ 仅在服务器内计算聚合比较。

独立验证脚本逐个加载八个子目录，校验文件存在、版本/请求/输出/receipt/NPZ 哈希、七组 bootstrap/first-action 数组、有限值、时钟和指标重算（`.aris/traces/reset-cache-v4-integrity-audit-20261011/remote_verify.py:65-85,92-205,317-351`）。执行时使用原实验 overlay 的 NumPy 环境；八个子目录的指标对象均逐字段精确重算一致。

## A. Ground-truth provenance：WARN

数据运动从缓存中读取，并用数据集的 `Mean.npy`/`Std.npy` 归一化（`scripts/physics_guided_collect.py:57-62`）。数据运动经坐标变换后生成固定 endpoint goal 和 heading，目标在生成前冻结（`scripts/physics_guided_collect.py:94-105`）。metadata 对此标为 `dataset motion, fixed before generation`，同时明确语义未经独立验证（`scripts/physics_guided_collect.py:129-139`；`results/reset_alignment_v4_20261011/full_val_planned_s11/metadata.json:2785-2788`）。

参考轨迹是混合来源：前 60 帧来自数据集，后 60 帧来自模型生成 continuation（`scripts/physics_guided_collect.py:147-204`）。指标在 generated segment 上将实际运动与该生成参考比较（`scripts/physics_guided_core.py:131-141`）。因此：

- 固定 endpoint goal/heading 可称为 **dataset-derived target**。
- 后半段 tracking 只能称为 **generated-reference controller-adherence proxy**，不能称为真实 GT。
- 语义质量没有独立 evaluator 或人评，不能作语义效能声明。

**支持的声明**：目标在生成前固定，来源可追踪。  
**不支持的声明**：整段 reference 是真实 GT；tracking 是真实动作误差；语义已验证。  
**下一检查**：为每个指标写入明确的 `eval_type`，并为语义/生成质量使用独立 evaluator。

## B. Score normalization：PASS

唯一的输入标准化是 `(raw - dataset_mean) / dataset_std`（`scripts/physics_guided_collect.py:56-62`）。endpoint、heading、time、tracking、jerk、penetration 和 slip 都以原始 SI 单位保存，并与冻结绝对阈值比较（`scripts/physics_guided_core.py:100-155`；`configs/physics_guided_20261011.json:17-32`）。没有指标除以当前模型输出的 max/min/mean。

endpoint-time 在 generated segment 中选择距离本次最小误差 1 mm 内、且最接近指定终点时刻的帧（`scripts/physics_guided_core.py:121-130`）。这不是自归一化，但它是尚未外部校准的事件定义，不能解释为官方 timing metric。

**支持的声明**：没有 prediction-statistic denominator 或自归一化抬分。  
**不支持的声明**：endpoint-time 是外部验证的标准事件时间指标。  
**下一检查**：对事件定义和容差做预注册/外部校准。

## C. 文件、计数与哈希：PASS

八个子目录均存在 `metadata.json`、`metrics.json`、`receipt.json`、`lineage.json`、`initial_states.npz` 和 `trajectories.npz`。六个 mini run 各 12 cases，两个 full custom-val run 各 72 cases，共 216 attempts；摘要也记录 216（`results/reset_alignment_v4_20261011/measurement_repair_summary.json:3`）。两个 full receipt 分别记录 72/70/0 和 72/71/0 的 cases/completed/joint-pass（`results/reset_alignment_v4_20261011/full_val_planned_s11/receipt.json:2-7`；`results/reset_alignment_v4_20261011/full_val_native_s11/receipt.json:2-7`）。

冻结版本记录了脚本、配置、manifest、外部 motion inputs、模型权重、upstream commit 和选择范围（`results/reset_alignment_v4_20261011/versions_before_execution.json:3-36`）。独立复核发现：

- 所有当前输入哈希与冻结记录一致；
- 每个 execution request 等于 lineage 内嵌 request；
- 所有 lineage output 哈希与当前文件一致；
- receipt trajectory hash 与 NPZ 一致；
- initial-state NPZ 与 metadata hash 一致；
- 八个目录的每一行指标在冻结实现下精确重算一致。

摘要自身列出的八个 request/output binding 均为 true（`results/reset_alignment_v4_20261011/measurement_repair_summary.json:585-633`）。预算账本有八个 exit 0 记录（`results/reset_alignment_v4_20261011/budget_ledger.json:2-49`），独立求和与摘要 GPU-hour 字段一致。

**支持的声明**：文件、次数、哈希、指标行和冻结输入真实存在且相互一致。  
**不支持的声明**：文件存在与哈希一致等同于指标构念有效。  
**下一检查**：接触代理修复后对新工件重复相同哈希/逐行重算流程。

## D. 主指标调用路径：PASS

collector 在保存 trajectory/metadata 后，对每个 case 直接调用 `physics_guided_core.metrics(...)` 并写 `metrics.json`（`scripts/physics_guided_collect.py:275-299`，实际调用在 `:284`）。主指标实现位于 `scripts/physics_guided_core.py:70-155`。审计从 NPZ 重新调用同一函数并比较序列化对象（`.aris/traces/reset-cache-v4-integrity-audit-20261011/remote_verify.py:126-131`），八个目录全部精确一致。

`physics_guided_lineage.certify_cached` 另有 metric recertification 路径（`scripts/physics_guided_lineage.py:44-153`），但 v4 runner/analyzer 没有调用它。本报告不把该未调用函数冒充执行证据；真正的执行路径是 collector `:284`，独立重算是本次审计完成的。

**支持的声明**：结果中的指标确由当前主指标函数生成。  
**不支持的声明**：v4 原执行已运行 `certify_cached`。  
**下一检查**：今后把精确 recertification 接入 analyzer，并记录实际 NumPy 版本。

## E. Scope 与 split：WARN

runner 从自定义 manifest 中每个 family 取前两个 `val` source，形成四个 source × 三个 offset 的 12-case mini batch（`scripts/run_reset_alignment_checks_v4.py:73-80`）。它对 planned/native 运行生成 seeds 11/12/13；full custom-val 只在 seed 11 运行 planned/native 两个 72-case batch（`scripts/run_reset_alignment_checks_v4.py:155-173`）。因此多 seed 证据只覆盖四个 source，完整 24-source 范围没有多 seed 复现。

自定义 train/val/test 在 clip_id、source_id 和 cache key 上互不重叠，但它并非 HumanML3D 官方 held-out 划分。独立交叉表显示：

- custom train：113 official-train + 15 official-test；
- custom val：19 official-train + 5 official-test；
- custom test：41 official-train + 7 official-test；
- mini 四个 source：2 official-train + 2 official-test。

聚合代码与检查位于 `.aris/traces/reset-cache-v4-integrity-audit-20261011/remote_verify.py:276-315`。代表性 metadata 同时显示 `official_split: test`/`split: val`（`results/reset_alignment_v4_20261011/planned_s11/metadata.json:13-20`）和 `official_split: train`/`split: val`（`:82-89`）。

**支持的声明**：这是一个内部无重叠的 custom-validation diagnostic。  
**不支持的声明**：这是官方 held-out 验证；full 结果已跨 seed 稳健；结果具有 comprehensive scope。  
**下一检查**：效能工作使用官方 held-out split，或始终保留 `custom diagnostic` 标签并排除 official-train source。

## F. Evaluation type：WARN（混合类型）

| 指标/证据 | 分类 | 判定 |
|---|---|---|
| dataset endpoint goal/heading | `real_gt` 的 dataset-derived target | 可作固定条件误差 |
| generated continuation tracking | `synthetic_proxy` / `self_supervised_proxy` | 只能作控制器跟随代理 |
| penetration/slip/contact | `simulation_only` | 几何/接触代理；当前 contact gate 不可靠 |
| semantics | 无独立评价 | 不支持语义声明 |
| human evaluation | 不存在 | 不支持人评声明 |

代码已将 contact metric 标为 `body-force and collision-primitive surface proxy`（`scripts/physics_guided_core.py:70-98`），并把语义状态写为 `not_verified_by_independent_evaluator`（`:152-154`）。这些标签是诚实的，但不使代理自动具备构念有效性。

## Reset/cache 与真实 simulator state：WARN

v4 在构造 dataset-to-world anchor 前调用上游 `task.reset()`，暂时抑制 observation，然后立即捕获七个导出张量；注释明确不再做额外 refresh（`scripts/reset_aligned_collect_v4.py:25-41`）。第一 action 前再次比较七个张量、保存 NPZ，并检查四个 rigid-body 字段等于 `_reset_rb_*` cache（`:55-81`）。八个目录都通过了这个有限 gate（full 代表：`results/reset_alignment_v4_20261011/full_val_planned_s11/metadata.json:4182-4193,4777-4804`）。

上游语义说明了这个 gate 的边界：

1. `HumanoidAMP._reset_envs` 在非空 reset 时把 `_state_reset_happened=True`，随后进入父类 reset（`closd/env/tasks/humanoid_amp.py:328-335`）。
2. reset 将 root/DOF 写入 simulator；progress/reset/terminate/contact view 被清零（`closd/env/tasks/humanoid.py:625-643`）。
3. AMP refresh 先刷新 simulator root/DOF/rigid tensors；若 reset flag 为真，再用 motion-library `_reset_rb_*` 覆盖四个 rigid fields，随后把 flag 设回 false，最后刷新 force/contact（`closd/env/tasks/humanoid_amp.py:594-613`）。
4. safe full reset 会做一次额外 `gym.simulate` 后再次 reset，然后返回（`closd/env/tasks/humanoid.py:562-577`）。第二次 refresh 同样消费 flag，因此 `task.reset()` 返回时 flag 已为 false。
5. 首次真实 step 的 post path 增加时钟、完整 refresh simulator tensors，再调用 reward/reset/observation（`closd/env/tasks/humanoid.py:1355-1369`）。flag 已 false，因此该步的 body trajectory 不再被 reset cache 覆盖。

所以，**初始 NPZ 证明的是导出张量与 reset cache 一致，不是完整 simulator state 一致**；而第一真实 step 以后保存的 body kinematics 是 PhysX refresh 输出，没有发现 reset cache 后续覆盖。摘要也正确限定为“不包含 PhysX hidden state、contacts、RNG 或 policy observations”（`results/reset_alignment_v4_20261011/measurement_repair_summary.json:4`；`versions_before_execution.json:26-28`）。

`full_reset_calls_before_first_action=3` 的字段名不精确：wrapper 对每次 `task.reset` 调用都计数（`scripts/reset_aligned_collect_v4.py:27-35,62`），player 在外层和进入首步前都调用 `env_reset`（`closd/learning/im_amp_players.py:240-294`），其中可包含空 done-index reset。它不是“三次完整 full reset”的可靠证明。

**支持的声明**：七个导出张量/四个 reset cache 在 reset-return 到 first-action 间一致；post-step body tensors 来自 PhysX refresh。  
**不支持的声明**：完整 PhysX 隐藏状态、contact、observation 或 RNG 同一；初始 cache 就是 simulator FK ground truth。  
**下一检查**：在禁用一次性 cache 覆盖的明确 fetch/refresh 点取得 simulator-derived rigid snapshot，并单独记录 hidden/contact/obs/RNG 范围。

## 时钟、目标、offset、seed 和 mode：PASS

adapter 用 `store['steps']` 定义显式参考帧，pre-step 断言 reference frame 与 physics step 相等（`scripts/physics_guided_collect.py:190-212`）。八个 `reference_frame` 数组均精确为 `0..119`（复核：`.aris/traces/reset-cache-v4-integrity-audit-20261011/remote_verify.py:201`）。

mini 六个 label 的 original goal、heading、first-action root 和前 60 reference frames 与 planned_s11 最大差为零；三组生成 seed 的 noise hash 各不相同；同 seed planned/native 使用配对 noise；±0.15 m/0 offsets 的独立误差检查在浮点精度内（计算：`remote_verify.py:242-274`；原 analyzer：`scripts/analyze_reset_alignment_v4.py:56-78`）。环境 seed 固定 11，生成 seeds 为 11/12/13（`results/reset_alignment_v4_20261011/versions_before_execution.json:28-35`）。

**支持的声明**：当前 adapter 的显式 reference/action clock、target、offset 和 mini seed/mode 配对控制成立。  
**不支持的声明**：这些控制可弥补 split 或 physical proxy 的有效性缺陷。  
**下一检查**：物理代理修复后沿用并重新验证这些控制。

## Physical proxy 构念与重复性：FAIL

`surface_metrics` 对四个足部碰撞几何求最低表面点与材料点速度；只有 `force_z > contact_force_n` 且表面近地时才纳入 support/slip（`scripts/physics_guided_core.py:70-98`）。冻结阈值是 `contact_force_n=10 N`、`surface_near_ground_m=0.02 m`、`sliding_fraction=0.10`（`configs/physics_guided_20261011.json:23-29`）。physical gate 直接依赖该 slip proxy（`scripts/physics_guided_core.py:139-149`）。

collector 的采样时序是：上游 post-step 完成 simulator tensors refresh；wrapper 先执行 reward，再立即 `.cpu().numpy().copy()` body/contact tensors；之后才执行 reset 与 observation（`scripts/physics_guided_collect.py:213-243`；`closd/env/tasks/humanoid.py:1355-1369`）。HumanoidIm reward 只读取 `_rigid_body_*` 并写 reward buffer，没有回写 body state（`closd/env/tasks/humanoid_im.py:827-898`）。contact tensor 按每 env 的全部 body reshape后只截取首个 humanoid actor 的 `num_bodies`（`closd/env/tasks/humanoid.py:214-230,903-908,959-983`）；没有发现其它 actor tensor 行错位。

服务器端配对诊断读取六个不同 inode/SHA 的 NPZ，在每个 planned/native 同-seed 对上比较前 60 帧。算法与原始 proxy 相同（`.aris/traces/reset-cache-v4-integrity-audit-20261011/contact_pair_diagnostic.py:36-58,66-100,111-185`）：

| pair | body pos/quat/linvel/angvel | contact 每帧不同 | 最大绝对差 | support Jaccard | 720 个 frame×case 中任一足标签不同 | 单 case slip fraction 最大差 |
|---|---:|---:|---:|---:|---:|---:|
| planned/native s11 | 逐位相同 | 60/60 | 656.943 N | 0.8495 | 329 | 0.0156 |
| planned/native s12 | 逐位相同 | 60/60 | 614.968 N | 0.8530 | 322 | 0.0338 |
| planned/native s13 | 逐位相同 | 60/60 | 693.784 N | 0.8487 | 323 | 0.0360 |

`±1..3` 帧的简单时间移位不能解释 contact 差异。reward/obs 或 reset cache 回写也不能解释，因为采样发生在完整 refresh 后、observation 前，且 post-step flag 已为 false。剩余原因属于尚未分辨的 PhysX 接触流形/solver history/采集语义或 GPU 非确定性。

正式指标从 `context_steps=60` 开始计算 physical proxy（`scripts/physics_guided_core.py:131-141`），所以这个前 60 帧配对诊断**不能证明某个已保存 full metric 数值具体错误**。它证明的是更基础的问题：同样显式运动学输入不能稳定得到相同 support selector。接近阈值的比较可因此翻转，physical proxy 尚不适合效能比较。

full custom-val 两个模式的 `physical_pass` 与 `joint_pass` 都为 0（`results/reset_alignment_v4_20261011/measurement_repair_summary.json:557-583`），analyzer 也明确停止于 `STOP_NONDISCRIMINATING_JOINT_GATE`（`:636-642`）。这是一项诊断 floor，不是任一模式的 efficacy 结论。

**支持的声明**：已保存 proxy 值由冻结代码和 NPZ 确定性重算得到；当前 gate 无区分度。  
**不支持的声明**：force-gated support/slip 可重复、构念有效、可比较 planned/native，或能证明物理效能。  
**必要下一检查**：运行同一 arm 的同步重复 witness，在标注清楚的 refresh/reward 前后同时捕获 rigid state、net contact 和足部 force sensor，并记录 `contact_collection`/solver 设置；之后校准或替换 force-gated slip proxy。

## Claim impact

| 声明 | 影响 |
|---|---|
| 八个 run 的七个导出张量在 reset snapshot 与 first action 间逐位相同 | **supported，须限定为 exported tensors/reset caches** |
| 初始快照证明完整 simulator-state identity | **unsupported** |
| 第一真实 step 后的 rigid-body trajectory 被 reset cache 继续覆盖 | **不成立；调用链支持 PhysX refresh 输出** |
| 216 attempts、文件、哈希、计数与冻结指标行真实存在 | **supported** |
| custom val 是官方 held-out | **unsupported** |
| tracking 是真实 GT 误差 | **unsupported；generated-reference proxy** |
| contact/slip physical gate 可重复并可比较模式 | **unsupported / FAIL** |
| planned/native 存在效能或物理质量优势 | **unsupported** |
| 语义质量经独立验证 | **unsupported** |

## 必要后续检查

1. 完成同一 arm、同步采样的 contact repeatability witness；在修复前停止 physical gate 和 efficacy 解读。
2. 用禁用 reset-cache 覆盖的 fetch/refresh 点比较 motion-library cache 与 simulator-derived rigid state；若要声称完整状态，还要覆盖 contact、observation、RNG 和隐藏 solver state 的明确边界。
3. 用可重复 simulation signal 或外部标签校准/替换 support/slip proxy，预注册阈值，并先证明 gate 有非零区分度。
4. 效能评估改用官方 held-out split；否则明确维持 custom diagnostic 标签。
5. 只有在 split 与 physical gate 构念有效后，才运行完整多 seed 评估。

## 审计结论

本次冻结证据可作为一个诚实、可追踪的 reset-cache 诊断存档，因此总体完整性为 **WARN**，provenance 为 **same-family/provisional**。接触力驱动的 physical proxy 在等价显式运动学下不重复，其有效性为 **FAIL**。当前没有证据支持任何数值效能、物理质量优势、官方 held-out 或完整 simulator-state 声明。
