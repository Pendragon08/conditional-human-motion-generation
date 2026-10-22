# ReactiveBFM 完整课程训练阶段报告

实际优化器更新：1,000 / 1,000,000。
配置：双 RTX A6000，512×2，全局 batch1024，400k warmup，EMA、BF16、激活重算；使用官方三子集组合与各自归一化。
该快照不等于已复现原文性能；目标到达数据/官方完整评测仍未公开。

## 检查点备份

下列文件大小及 GitHub 服务端 SHA256 与服务器保存记录一致。

- model000000999.pt: 857731942 bytes, SHA256 `ceba713b33bc9709473351424dbb56c852c2eae8aa897533d588face7ac7d0f5`
- opt000000999.pt: 852468483 bytes, SHA256 `069bd1ad4c61e19cb1473481e079b8cdef8de48ae9b1cc599f592e6966099839`

## MuJoCo 诊断

本阶段执行 9 个测试；完整执行闭环 9 个。
根部高度始终≥0.4m 的诊断数量：0/9。这是高度阈值代理，不能作为原文成功率。
未评估阶段不计算动作成功率。CPU 原生 PyTorch 跟踪器沿用官方网络/适配器，未使用 TensorRT，不声称实时性能。

- stand_still seed42: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.05469106465242099m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- stand_still seed43: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.054676977390490496m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- stand_still seed44: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.0987588862396173m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- walk_forward seed42: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.05463616189696922m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- walk_forward seed43: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.054671463618648435m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- walk_forward seed44: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.09164328210995315m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- punch seed42: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.057725176820916914m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- punch seed43: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.054445395675153746m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。
- punch seed44: PHYSICAL_CLOSED_LOOP_EXECUTED; 控制步 400; 最低根部高度 0.1092579224337478m。
  状态反馈一致性检查：FEEDBACK_CONSISTENCY_PASSED；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。

## 数据与保存

训练数据文件保留授权本地/服务器；GitHub 保存来源、划分、校验值。
本检查点备份前不会清理其副本；仅已验证备份的旧检查点可按保留策略清理，最新检查点始终保留。
仿真 NPZ/视频为模型与物理环境生成的数据，不包含原始训练动作包。
