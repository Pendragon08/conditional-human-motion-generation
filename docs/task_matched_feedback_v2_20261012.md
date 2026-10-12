# C1任务匹配配对验证：独立第二准备入口

原CPU准备命令退出1：启动时__file__为相对路径，写源文件绑定时relative_to绝对项目根失败；未写出versions、未启动GPU。原目录及PREPARATION_FAILURE.json保留，不重运行。第二入口用Python模块导入取得绝对__file__，使用独立results/task_matched_feedback_v2_20261012目录，原实现不修改。两组、输入、预算、阈值及范围均沿用docs/task_matched_feedback_20261012.md；这不是新的机制。

CPU准备（执行者一次）：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/task_matched_feedback_v2_20261012.py --prepare
```

新独立跟随者原样执行一次：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/task_matched_feedback_v2_20261012.py --check
```

实际exit0和TASK_MATCHED_FEEDBACK_CHECK_PASS才通过；核对三批各72输入（生成、零更新重复生成、固定参考回放）、生成实际数据/反馈JSON字节、实际模型字节、初始状态/观测/轨迹重复一致和回放参考时钟。写新目录DOCUMENT_WITNESS.json/.md：status、runtime_witness_sha256、versions_sha256、实际命令/退出码、same-family/provisional。FAIL保留不重试。隐藏PhysX状态、语义正确性和效力不认证。

通过后同命令--run。测量及正式实验最多23个不同批次、新增1800秒墙钟及原72GPU小时/阶段上限/15GiB输出/20GiB空闲限制不变。不放宽原目标、阈值及C2/C3/G1门，不重写旧C1，暂不写Overleaf。
