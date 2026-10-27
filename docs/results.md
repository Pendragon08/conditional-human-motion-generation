# 既往 DreamControl / CLoSD 结果汇总

下文由整理前 README 的结果部分迁入，保留原数值、协议和限制。它不是本次新运行的评价。

PhyGile 与 ReactiveBFM 的结果分别见 [PhyGile 独立索引](../baselines/phygile/INDEX.md) 和 [ReactiveBFM 独立索引](../baselines/reactivebfm/INDEX.md)，不合并进下面的既往基线表。


### DreamControl: archived control baselines

使用官方参考动作，8192 个并行环境、2000 次训练迭代、seed 42；以官方评价器在 1000 环境、500 帧下评价。部分任务有明确记录的恢复或显示 I/O 适配，详见 [原始报告](../reports/03_dreamcontrol_multitask_baselines.md)。

| 任务 | 成功率 | 任务 | 成功率 |
|---|---:|---|---:|
| Kick | 85.3% | Punch | 87.2% |
| Jump | 100.0% | Sit | 100.0% |
| Pick | 93.9% | Bimanual Pick | 99.9% |
| Ground Pick | 38.1% | Ground Pick Top | 0.0% |
| Button Press | 99.9% | Open Drawer | 100.0% |
| Open Door | 99.0% | | |

11 项均已完成评价，不等于 11 项均达到论文性能，也不代表生成方法创新已得到验证。成功率遵循官方任务判据，不能扩展解释为自然抓握、持续接触或未见场景泛化。

### CLoSD: text-to-motion baseline report

[完整报告与视频](../results/closd_baseline_20261010/README.md) 已同步至 GitHub，提供可离线打开的 HTML 与下载包。报告包含作者公开动作重评、20,000 条新采集动作评测、9 段任务演示与 3 段生成动作预览。报告内容更新于 2026-10-17，补充发布于 2026-10-21；原指标不变，没有新增训练。该文本动作评测与下表多任务控制评价分别记录。

### CLoSD: released-checkpoint evaluation

4096 环境、500 帧、seed 0，每个状态至少 1000 次重置。来源：[评估记录](../reports/13_CLoSD_official_checkpoint_eval_20261001.md)。

| 官方已发布权重 | REACH | STRIKE | SIT | GET_UP |
|---|---:|---:|---:|---:|
| 未微调 | 100% | 85% | 76% | 56% |
| 已微调 | 100% | 94% | 89% | 97% |

STRIKE 合并拳击与踢击计数。这是官方权重的评价复现，不是本项目重新训练取得的成绩。

### Cross-simulator and candidate-method findings

MuJoCo 测试已保存成功与失败案例，部分任务存在跌倒、接触或物体抬升失败；见 [跨仿真器结果](../reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md) 与 [扰动和实体接触实验](../reports/11_Kick_Punch_robustness_contact_20261001.md)。本项目尚未建立候选执行反馈机制的可靠下游收益，也没有实机测试结论。


[返回首页](../README.md)
