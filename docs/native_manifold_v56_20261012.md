# V56：修复接口后的原生接触观测诊断

只读采集模拟器原生足部接触流形，检查V51物理质量地板。不是C2/C3效力验证、不是新评分、不是实机接触真值。24个既定验证来源、三种既定偏移、种子11，每方法72输入，不按得分选样本。跟踪参考来自V51所选合成精化轨迹，明确为synthetic proxy。

执行器仅重新开启已校准的被动导出。生成器采样零次；固定模型、策略、环境、重置、参考速度、目标与判据。每批必须逐元素核对9项轨迹、任务观测、初始状态、传感器诊断及原指标与V51对应回放完全相同。不同则保留失败并停止，不能将变动后的结果解释为原实验。

独立文档见证只执行以下命令一次。不要由状态文件推定进程存活；输出目录存在时先读终态和既有见证，不重复执行。无自动重试，无其他项目清理，无外部通知。

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_native_manifold_v56_20261012.py --check
```

预期exit0及 `NATIVE_MANIFOLD_V56_DOCUMENT_CHECK_PASS`。核验 `results/native_manifold_v56_20261012/runtime_witness.json` 全部exact_arrays为true、frozen_metrics_exactly_equal为true、generation_calls为0，实际接触记录非零且解析错误为零；预算台账只有一条measurement计费。读服务器真实文件，核对源码与本文文件哈希。

见证将独立判定写入 `DOCUMENT_WITNESS.json`、`DOCUMENT_WITNESS.md`（两端同步），JSON至少有status、runtime_witness_sha256、doc_sha256、source_sha256、实际命令及exit_code。状态机仍为待审，见证不是后续完成宣告。root在通过后用同一命令将`--check`替换为`--run`，只执行随机筛选参考的另一批。总计两次回放，单独上限600GPU墙秒，计入原4小时measurement上限；全轮72GPU小时/15GiB输出/至少20GiB可用磁盘仍有效。

后续离线分析以对应物理时刻的接触材料点及原生COM计算速度，明确区分原最低碰撞点代理与模拟接触点诊断。检验两种测量的差异，不放宽阈值、不改历史成功率、不把原生lambda/接触观测当真实力。失败、提前终止及所有来源保留。仅服务器保存人体数组，本地/GitHub保存小型汇总、代码、日志和绑定记录。Overleaf不写实验结果。
