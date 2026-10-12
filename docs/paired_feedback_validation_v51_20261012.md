# V51：修复接口后的配对反馈验证

本轮仅验证集，不改写V28首轮否定结果，也不视为完成原十配置独立测试。使用修复后的V43参考速度接口、冻结的生成器／跟踪器与原始条件／阈值。

24来源×3水平偏移×3种子11/12/13，每方法216输入。反馈法与随机候选法都用实际执行前缀及坐标锚点，每输入最多3候选，第一候选共享。随后两次候选中，两方法使用同一候选序号对应的随机噪声；只有反馈法使用执行误差调整条件。位置步幅5cm、累计15cm；朝向每次5度。文本、目标时刻、长度与原始评估目标不变。

与V28历史协议的区别：同一候选序号在两方法间配对噪声，两方法均使用实际前缀；这不是把旧分数重新算成新分数。选择规则完全沿用已有联合质量优先规则。构建出的物理精化参考必须独立回放后评价，不能以候选构建成功代替数据可用性。15个构建批次和6个回放批次，累计最多1GPU小时；全局72小时、测量4小时、对比12小时、剩余磁盘20GiB和输出15GiB上限仍适用。失败批次不能重试或覆盖。

## 独立见证：仅执行下面命令一次

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_paired_feedback_validation_v51_20261012.py --check
```

预期退出0，出现`PAIRED_FEEDBACK_VALIDATION_DOCUMENT_CHECK_PASS`。只运行种子11首轮72输入，计入测量预算。逐数组与V46原生构建对照比较，参考速度、观测、实际轨迹及初态必须完全一致。核对执行请求、源码／权重／数据哈希、预算及日志。V52解析校准的独立见证是前提；V50路径失败被保留，该校准不证明真实接触点正确。

见证在`results/paired_feedback_validation_v51_20261012/DOCUMENT_WITNESS.json`记录`status`和`runtime_witness_sha256`，并记录命令与结果证据。通过后才能由主执行者运行`--run`；首轮不重跑，只接受执行绑定及独立见证。命令失败必须保留状态、日志和预算，不修复后暗中重试。

## 主执行者继续命令

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_paired_feedback_validation_v51_20261012.py --run
```

预期终态`COMPLETE_VALIDATION_QUALITY_FLOOR`或`COMPLETE_VALIDATION_REQUIRES_FULL_ANALYSIS`。后者不等于创新成立，还需效应区间、语义与多样性、消融和独立测试。若所有联合合格为零，停止该选择／条件补偿路线的继续扩张，记录缺乏可判别收益；不能通过放宽判据、改目标或删除失败样本启动G1与下游训练。所有实验结果只同步GitHub及本地报告，不写入Overleaf。
