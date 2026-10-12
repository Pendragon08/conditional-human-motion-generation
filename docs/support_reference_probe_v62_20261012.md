# 支撑阶段参考修正：独立验证集诊断

新诊断不替换C1、C2、C3的协议、计数或否定结果。C1联合收益为零，C2/C3原有效性启动条件仍未满足。这里先检验参考侧物理修正是否值得继续研究，不声称提出了新的足锁定算法，也不将它称为PhysDiff或Morph复现。

复用V51的单次原生构建 `val_shared_s11_r0`：24验证来源、3原始偏移、种子11，共72输入。所有提前终止、补齐与不合格案例保留。输入是模拟精化后的合成参考，跟踪误差是相对于此参考的代理，不是MoCap真值。每个方法共用相同来源的生成与首次物理构建开销，再各执行一次参考回放；缓存构建成本不能宣称为零。新生成次数为零，新仿真批次恰为3，自己的计费上限600秒，受原72 GPU小时和磁盘上限共同约束。

三方法：原参考、支撑阶段水平根平移修正、相同修正加末帧位姿保持。输入运动的相对身体姿态和高度不变，朝向、长度、文本、原始目标及其评估时刻不变。根据首次模拟的足部受力与碰撞几何代理建立支撑权重，优化水平平移速度以减小接触点速度，加入位移及二阶时间差分正则；每帧平移不超过10厘米，前61帧不改变。仅完整输入接受修正，未完成输入原样补齐并继续计入分母。第三方法额外固定末帧平移为零；第二方法取消该约束用于消融，但评估仍对同一原始目标，不能重定义目标。

该有限参数化未解决关节局部修正、完整动力学或执行器约束。物理质量、连续性、条件及联合合格指标继续使用原阈值。单种子结果仅用于判断机制是否有值得追踪的物理改善；不能作为创新验收、下游收益或G1启动证据。若仍没有物理及联合改善，停止该参数化；不能放宽阈值获得通过。

## CPU准备（执行者恰好一次）

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/prepare_support_reference_probe_v62_20261012.py
```

期望exit0和 `SUPPORT_REFERENCE_V62_PREPARED`；没有任何仿真、生成或模型更新。冻结三个参考、源绑定、原目标、版本及脚本哈希。

## 独立文档执行见证（恰好一次，不修复或重试）

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_support_reference_probe_v62_20261012.py --check
```

期望exit0和 `SUPPORT_REFERENCE_V62_DOCUMENT_CHECK_PASS`。见证者核对所有源与参考哈希、72输入完整分母、原始目标及顺序、三项参考修正边界、原生重置门、120帧时钟、同时间的参考／实际状态、参考速度规则、数据来源及实际预算。记录actual PID、收据和批次绑定，不能以文件存在替代真实运行。写 `DOCUMENT_WITNESS.json/.md` 于本诊断目录；JSON仅在通过时含 `status: PASS` 和实际 `runtime_witness_sha256`。同步本地与远程并核对字节。

独立PASS后，执行者才用同一命令的 `--run` 运行另外两批。它不重复baseline，结束 `SUPPORT_REFERENCE_V62_COMPLETE`。后续只读分析、来源聚类统计和独立审计仍必须完成；启动器结束不等于实验有效。本诊断不写入Overleaf。
