# CLoSD官方训练状态敏感性诊断

目的：当前固定参考重放使用`closd_t2m`环境和多任务微调权重；官方README的文本动作示例使用文本动作微调权重。诊断固定控制策略的训练状态是否影响当前物理质量下限。三个检查点属于同一个CLoSD算法，不作为三个独立方法，不是C2/C3正式试验或新创新点。模型不训练，生成器不采样。

固定24个内部验证来源、三个原始水平偏移、种子11，每状态72输入。沿用`support_reference_probe_v63_20261012/references/baseline.npz`，不改原目标、文本、动作时长、前缀、评分或阈值。预先绑定所有权重，再观察成绩；不能按成绩筛选测试来源。内部验证含上游训练/测试来源混合，不是独立最终测试。

顺序：多任务微调作为新入口测量见证，然后文本动作微调、未微调。三个状态全部使用同一新入口，绑定实际加载的参考、项目源码和传给torch.load的权重字节。核对上游Git HEAD与已跟踪closd源码无改动。上游缺失权重时可能退回新训练，新入口要求确实加载指定发布权重，否则保留失败并停止。

先在服务器执行准备（CPU检查，没有仿真）：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_20261012.py --prepare
```

独立文档跟随者原样执行一次：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_20261012.py --check
```

预期实际进程exit0及`CONTROLLER_STATE_DOCUMENT_CHECK_PASS`。独立见证检查raw/multitask实际输入及权重绑定、72行完整分母、零生成调用、重置缓存门，写`DOCUMENT_WITNESS.json`含status、runtime_witness_sha256、versions_sha256、review_independence=same-family、acceptance_status=provisional。失败不重跑或覆盖。

通过后执行同运行时命令的`--run`；仅新增两次回放，再以冻结v23指标从原始数组复算。预期`CONTROLLER_STATE_DIAGNOSIS_COMPLETE`。跨状态所有导出初始数组必须逐元素一致，bootstrap到first_action的七种状态必须一致；隐藏PhysX状态不因此得到认证。所有失败、提前终止保留，不按是否完成排除分母。

预算：新增最多三个仿真批次，600秒GPU占用墙钟上限；每次最大240秒且受600秒剩余额限制。计入原measurement4GPUh、总72GPUh及15GiB输出上限，启动前数据盘至少20GiB。不删除旧记录、不安装依赖、不训练、无G1、无下游学习、无Overleaf结果同步。

判读：若文本动作微调改善代理质量，只能说明当前配置对控制器训练状态敏感，需要重新冻结正式实验基础，不能追溯改变C1结果或算新创新。若仍没有联合合格，记录否定诊断，C2/C3门继续不满足。质量指标仍为模拟接触/滑移及合成参考跟踪代理，不是机器人实机或语义真值。
