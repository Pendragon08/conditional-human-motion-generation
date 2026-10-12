# C1任务匹配权重配对验证（独立新实验）

目的：检查原条件误差补偿机制在官方CLoSD文本动作跟踪权重下，是否获得联合质量收益。三个官方检查点是同一算法的训练状态，不能当作三种独立方法。旧C1否定结果、旧判据和审计均保留，不合并旧新成绩。选择t2m依据官方文本动作例程的配置，不依据某条记录是否成功。

范围：继续使用预先确定的24个内部验证来源、每来源三种偏移和种子11/12/13，每方法216输入。不能只选前一轮物理合格记录。生成器冻结，跟踪器固定为官方CLoSD_t2m_finetune。原始目标、文字、长度、时间、v23所有阈值和每方法3个候选保持不变；共享首候选，后续候选按同一噪声序号配对。比较误差反馈与等预算随机候选；两组使用相同候选排名和独立固定参考回放。选择参考仍是控制器执行得到的合成代理，不是MoCap或机器人真值。

先运行三批独立测量：首轮生成、相同种子/噪声且零条件更新的生成、首轮原生成参考固定回放。前两批导出的初始状态、观测和轨迹逐元素一致，实际消费的外部数据、反馈JSON、模型字节和参考内容全部绑定，才能通过。生成模式和固定回放模式都需实际运行，不能沿用前轮仅回放的见证。隐藏PhysX状态不认证。

测量与后续调用均计入原4/12阶段上限和72GPU小时总预算。此新实验新增上限1800秒墙钟，单批最多180秒；数据盘至少20GiB，整个授权轮输出低于15GiB。固定跟踪器和冻结原判据；脚滑指标是碰撞表面代理，时间是最接近原目标时刻的代理，跳跃腾空不判漂浮。全部失败和提前终止保留，不重试失败批次，不改变目标或阈值。

CPU准备（执行者执行一次）：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/task_matched_feedback_20261012.py --prepare
```

新文档跟随者独立原样运行一次：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/task_matched_feedback_20261012.py --check
```

只有实际exit0和TASK_MATCHED_FEEDBACK_CHECK_PASS才可写PASS。核对results/task_matched_feedback_20261012的versions、三批请求/lineage、实际载入数据与权重、初始状态、完整72输入及测量计费。写该目录DOCUMENT_WITNESS.json/.md，包含实际命令、终态退出码、runtime_witness_sha256、versions_sha256、same-family/provisional。任一失败均保留，不重跑。

独立见证通过后同命令改为--run。预期TASK_MATCHED_FEEDBACK_VALIDATION_COMPLETE及exit0；21个正式构建/回放批次中首批已在测量运行，不重复，测量额外两批另计，最多23批。只是验证集诊断，未见来源测试、语义/多样性护栏和下游学习未完成时不得宣称创新有效。C2/C3及G1启动门不放宽。暂不写Overleaf，更新现有GitHub与本地HTML。
