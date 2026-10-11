# 条件误差反馈：重置修复后的验证集诊断

本轮不写入 Overleaf。冻结 DiP 和 CLoSD 策略；不安装依赖，不改变原始目标，不修改历史成功阈值。

216 次重置/坐标校验完成后，完整验证集的联合质量指标出现零合格率。本轮只检查有限次条件纠偏的作用，不把小规模诊断当作原计划的正式测试，更不据此宣布论文创新成立。

## 独立文档执行见证

从本机执行下面命令一次：

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_condition_feedback_probe_20261011.py --check
```

预期退出码0，并出现 `CONDITION_PROBE_BATCH_PASS val_native_s11_r0` 和 `CONDITION_PROBE_DOCUMENT_CHECK_PASS`。检查 `results/condition_feedback_probe_20261011/raw/val_native_s11_r0/` 的 receipt、metadata、lineage，以及 requests、versions、budget_ledger。应保留12个首次尝试、所有七组重置状态差值为0、原始目标绑定有效。首次尝试可能失败，不以12个成功作为门槛。不得重复执行、改代码或自行运行 --run。

## 后续诊断

执行同一命令的 --run 版本，仅复用散列与请求绑定相同的已完成首批。

- 沿用修复校验中预选的验证来源：每类2个原始来源、3个偏移，共12个输入，生成种子11。原片段及筛选在运行前固定。
- 比较3候选的原生实际前缀随机筛选，与最多3轮的有界条件误差反馈；加入去除反馈、打乱来源反馈两项消融。去除反馈仍实际生成和仿真3个候选，不拿缓存冒充计算。
- 固定文本、长度、目标时刻和原始目标；纠偏每轮≤5cm、累计≤15cm，朝向每轮≤5°。纠偏臂和两项消融使用相同随机噪声；随机筛选臂使用三个独立候选噪声。各臂都获得三个候选，筛选规则相同。
- 合计9个构建批次、4个独立参考回放批次，108个构建尝试、48个回放尝试。中途终止保留分母。共同首批只在总体账本记一次，在每臂等效成本中仍计入。
- 使用修复后的收集器与已冻结阈值。报告条件、物理代理、跟踪、联合合格率和原始来源聚类差值。单种子/4来源仅是诊断，不等于原计划的三种子/48来源独立测试。
- 若联合指标继续为零，保持正式测试、G1与下游训练暂停，先解决物理代理的鉴别能力和质量瓶颈。条件误差下降不能替代主要指标，也不能作为有效创新结论。语义/多样性护栏未运行，明确标为未验证。

所有失败记录和许可轨迹保留服务器。前置≥20GiB数据盘空闲、本轮累计输出≤15GiB、历史失败及本轮累计≤72GPUh；本诊断计入12GPUh对比预算。保存真实子进程耗时与失败状态。空闲GPU容量检查保留已有查看器进程，不删除其他项目、不进行实机部署、不调用VLM API。
