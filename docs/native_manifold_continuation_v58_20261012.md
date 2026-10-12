# V58：明确保留首次主入口失败后的缺失批次续行

V56独立`--check`已实际完成，反馈臂72输入的33项数组及原指标完全一致。但root在见证文件尚未复制服务器时过早调用`--run`，入口在读取文件前失败，没有启动随机臂。另原入口只接受literal PASS，跟随者给出的是PASS_DOCUMENT_FOLLOWING_RUNTIME_WITNESS。不能改写见证判定以适配旧入口。

原代码、见证和`failed_main_invocation_state.json`均保留。V58独立入口校验真实见证、文档、源码、运行结果哈希及全部一致性证据，接受两个明确PASS枚举；确认不存在随机臂，且台账只有先前一条。它只续行缺失的随机臂一次，保存启动绑定与终态收据，不重跑反馈臂，不把原V56命令改称成功。

原环境/模型/目标/参考/判据不变，原V56文档实验边界和600GPU墙秒上限继续有效。此次依然是接触测量诊断，不是C2/C3效力。独立跟随者先完成V56见证双端复制，再只执行以下命令一次，预期exit0及NATIVE_MANIFOLD_V58_CONTINUATION_COMPLETE。若目录或manifest已存在，读真实进程/终态，不再次执行。失败保留，不调用原`--run`、不自动重试。

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v58_20261012.py
```

核对`random_equality.json`的全部数组与指标一致、非零接触与零解析错误、generation_calls=0，台账恰好两条measurement；原反馈轨迹哈希和首次失败文件不变。将独立文档核验写入V56目录`CONTINUATION_DOCUMENT_WITNESS_V58.json/.md`，保留原见证，不覆盖它。JSON记录status、实际命令、exit_code、doc/source/hash、continuation manifest/receipt/summary哈希。同步两端；不做后续离线诊断或效力判断。
