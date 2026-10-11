# 条件误差反馈诊断 V2：仅修正浮点复核

历史 `condition_feedback_probe_20261011` 首批与第二批均实际完成。第二批重算朝向误差时出现约4.44e-16弧度差异，旧JSON散列完全相等断言失败；原始轨迹、分数、失败状态和已用预算全部保留。V2在新目录实际重新生成，不覆盖或冒充历史运行。

使用隔离的 `physics_guided_numeric_lineage.py`，其余输入/输出绑定检查与历史验证器相同。浮点值只允许 `32 * float64_epsilon * max(1, abs(saved), abs(recomputed))` 的重算舍入差异；类型、键、分类、计数、None及文本必须完全一致。逐批保存差异证书。不改变分数、阈值、评价公式或实验设置。旧验证器与216次重置测量不修改。

## 文档执行见证

独立见证者原样执行以下命令一次，不重试，不执行 --run，不修改源码。允许读取该首批证据与哈希。

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_condition_feedback_probe_v2_20261011.py --check
```

预期退出0，标记 `CONDITION_PROBE_BATCH_PASS val_native_s11_r0` 及 `CONDITION_PROBE_DOCUMENT_CHECK_PASS`。检查 `results/condition_feedback_probe_v2_20261011/raw/val_native_s11_r0` 的receipt、metadata、lineage、metric_numerics，与requests、versions、budget_ledger。保留12次首次尝试，七组重置状态与首次动作前状态差值均为0。目标、冻结模型和源数据绑定一致。不要求任务成功；同一目录出现执行者的后续批次不影响首批见证判据。

## 后续 --run 诊断范围

4个预选验证来源（跳跃/舞蹈各2个）、3个目标偏移、生成种子11，共12输入。冻结DiP和跟踪器，固定原始目标。四臂为原生实际前缀3随机候选筛选、有界执行误差反馈、零反馈、来源打乱反馈。每臂3候选，全部实际仿真；同一筛选规则，并独立回放选出的参考。反馈每轮≤5cm/5°，位置累计≤15cm；文本、长度和目标时刻不变。

9批构建共108首次尝试、4批回放共48首次尝试。失败和提前终止留在分母。共用首批总预算计一次，逐臂成本按独立运行的三候选等效成本报告。此诊断只有4来源和1种子，不代替原计划432输入/3种子正式测试，不证明创新成立。

沿用冻结质量判据，足部滑动明确为足部表面几何与支撑状态的代理。语义/多样性护栏未运行，标为未验证。若联合合格率仍为零，则继续暂停正式测试、G1和下游训练，不将条件误差下降解释成已验证创新。

启动要求数据盘≥20GiB、累计新增输出≤15GiB；历史失败（包括V1两批）、重置测量与V2均计入72GPUh总上限，本诊断计入12GPUh对比上限。仅使用空闲GPU，不安装依赖、不删除文件、不进行实机测试或VLM调用。实验结果同步GitHub及本地报告，暂不写入Overleaf。
