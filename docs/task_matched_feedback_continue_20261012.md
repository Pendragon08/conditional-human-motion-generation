# C1任务匹配验证：独立后续调度预算

上一入口三批实际测量已完成：约93、92、91秒，共约276秒。原1800秒上限预测不足以完成23批；原入口只完成测量，不执行其--run、不延长其上限。新调度入口在正式反馈对比前冻结3600秒新增墙钟预算，实际新增20批；原测量另计且不重复计费。全轮仍受用户72GPU小时、测量4小时、对比12小时、输出15GiB、数据盘20GiB限制。没有增加每输入候选数或改变评分。

科研设定沿用docs/task_matched_feedback_20261012.md：固定24内部验证来源、三偏移、三生成种子，文本动作微调权重，冻结生成器、原目标和阈值，3候选/方法，逐序号配对噪声，反馈与等预算随机对比。复用已见证的种子11首候选，它保持原执行请求/versions和完整哈希，不重新标注为新执行。新批次使用独立目录results/task_matched_feedback_continue_20261012及新versions；实际初始数据、权重字节和参考仍由同一冻结收集器校验。

唯一调度实现差异是把新目录自己的剩余调用墙钟额度从1800设为3600；原源码和原测量结果不改写，完整组装函数哈希记录在新versions。不是新算法或新创新。原三批已验证生成、零更新重复执行和回放，模拟器/环境/收集器没有变化。本节独立检查只验证新调度绑定，不再次认证GPU或隐藏PhysX状态。

准备（执行者一次）：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/task_matched_feedback_continue_20261012.py --prepare
```

新独立文档跟随者原样运行一次CPU调度检查：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/task_matched_feedback_continue_20261012.py --check
```

实际exit0及TASK_MATCHED_CONTINUATION_CHECK_PASS才写DOCUMENT_WITNESS.json/.md（status、runtime_witness_sha256、versions_sha256、实际命令/退出码、same-family/provisional）。核对新调度/组装函数哈希、旧三批实际PASS、旧首候选全部绑定、科学输入/模型/阈值不变、新增20批及预算限制。明确本次new_gpu_batches=0，实际GPU接口依据前一独立三批见证。不重跑失败命令、不改源文件或版本。

通过后同命令改--run。预计TASK_MATCHED_CONTINUATION_COMPLETE及exit0，三种子两方法各216输入。保持所有失败和提前终止。只有独立分析和审计完成后解释效力；此验证集不支持未见来源泛化或机器人收益，C2/C3/G1/下游门不放宽。更新现有GitHub和本地HTML，不写Overleaf。
