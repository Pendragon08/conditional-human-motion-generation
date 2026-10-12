# CLoSD训练状态诊断：独立第二入口

第一入口`controller_state_diagnosis_20261012`的独立执行见证为FAIL：versions缺少启动器要求的upstream_commit，校验阶段退出1，未启动仿真。完整失败记录保留，不重新运行或改写。

第二入口只在CPU准备时将实际核实的上游Git HEAD写入新的versions。冻结的三个权重、72个内部验证输入、原始参考、目标、阈值、速度/接触测量、训练状态顺序、600秒新增总预算均沿用第一版说明`docs/controller_state_diagnosis_20261012.md`。三个权重仍是同一算法的不同训练状态；不认证候选创新、G1或下游学习。

CPU准备命令（已由执行者执行时不得重复）：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_v2_20261012.py --prepare
```

新的独立文档跟随者原样执行一次以下测量命令：

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_v2_20261012.py --check
```

实际exit0及`CONTROLLER_STATE_DOCUMENT_CHECK_PASS`均出现才通过。核对新目录`results/controller_state_diagnosis_v2_20261012`中请求与lineage、实际加载参考和权重字节、72行完整分母、零生成采样、重置状态/缓存门。写新目录DOCUMENT_WITNESS.json/.md：status、runtime_witness_sha256、versions_sha256、实际命令/退出码、same-family/provisional。失败保留、不重跑。

通过后执行同命令的`--run`，新增文本动作微调和未微调两次回放，预期exit0及`CONTROLLER_STATE_DIAGNOSIS_COMPLETE`。导出的全部初始状态逐元素一致才解释跨状态成绩；隐藏物理状态不认证。全部输出及失败计入原阶段与总资源限制，数据盘20GiB、输出15GiB、原测量4GPUh及总72GPUh限制保持。600秒是本诊断新增三批次墙钟上限；第一入口未运行GPU。
