# 参考侧诊断续跑：保留失败，按冻结数值规则复核

V63独立文档见证的原FAIL和baseline回放保留，不覆盖、不重跑。独立NUMERICAL_CONTRACT_ADDENDUM仅检查原冻结数值契约及实际CUDA速度表达式；不改变实验判据、输入、参考、控制器或生成器。只有该追加复核为PASS，才执行另两组原定参考回放。原V63入口仍不直接使用。

新增入口逐组核对原见证绑定的versions、preparation、三个参考、baseline输出及收据哈希，复用一次baseline。三组每组72输入，24验证来源、3偏移、种子11。所有物理、条件、跟踪指标与失败分母保持原样。仍是单种子的参考参数化诊断，不认证C1/C2/C3有效性，不进入G1或下游训练，不记录Overleaf结果。

## 独立文档预检：恰好一次

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_support_reference_probe_v65_20261012.py --check
```

期望exit0及SUPPORT_REFERENCE_V65_PREFLIGHT_PASS。不执行仿真、不调用生成器、不修改原见证。独立执行者核对追加复核、原始失败及当前冻结输入哈希，确认两组尚未执行及baseline没有重跑。写CONTINUATION_DOCUMENT_WITNESS_V65.json/.md，只有通过才记录status:PASS、source_sha256（本续跑入口文件）、preflight_sha256（continuation_v65_preflight.json）与document_sha256（本文）。同步本地、服务器并核对字节。

## 两组仿真：执行者恰好一次

同一命令改为--run，仅在独立预检PASS后执行。原先600秒自己的总预算及原72 GPU小时预算共同约束；数值CUDA复核的实际耗时也计入账本。数据盘至少20GiB可用，新增输出总量小于15GiB。自动检查空闲GPU，不抢占其他用户任务。保留真实启动PID、时间、版本和追加证据。

预期SUPPORT_REFERENCE_V65_CONTINUATION_COMPLETE，生成零次、新增两次仿真。所有参考/实际状态按同一动作转移索引记录：参考和观测在动作前，实际状态在动作后；不能称为同时刻传感器真值。完整对照只在两组实际结束并通过来源校验后分析，分析同时报告每个指标的有限值数量、缺失数量及完整输入分母。

分析只决定是否继续研究这一个根平移参数化：联合收益为正仍需更多种子、独立测试及语义/多样性核验；只有物理改善时不声称联合有效；没有物理或联合改善则停止该参数化。保持末帧参考不保证实际终点准确。最终结果须独立审计后同步GitHub和本地HTML。
