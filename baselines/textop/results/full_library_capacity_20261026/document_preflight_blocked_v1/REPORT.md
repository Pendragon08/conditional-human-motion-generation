# 独立文档见证：前置连接失败，未派发

结论：`DOCUMENT_WITNESS_NOT_DISPATCHED_PREFLIGHT_SSH_TIMEOUT`。本次没有运行 `full_library_document_v1`，没有独立数值结果，不能把全库容量前提标记为已完成。此次失败发生在只读 SSH 前置检查，属于连接层失败，不是 kernel、PPO 或 SDK 数值失败。

## 独立性与执行边界

见证者先读取 run-experiment 技能、`FULL_LIBRARY_CAPACITY_ENVIRONMENT.md`、`configs/full-library-capacity-spec.json` 和 `SERVER_NOTES.md`，之后读取 compute-env-contract、检查适用的 AGENTS.md 路径与本地协调器状态。检查的项目祖先路径未发现 AGENTS.md；服务器入口来自明确指定的环境文档。未修改文档、配置、算法、依赖、网络或其他工作者，未 Git 提交/推送，未启动正式训练。

文档要求 GPU1、CPU0–3、Torch allocator fraction 0.65；启动前必须检查真实 GPU/RAM/磁盘和并行个人 SDK 工作。本次只尝试了下面这一个只读前置连接，Python 检查内容保留在 `preflight_remote.py`：

```powershell
$preflight | ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 -'
```

- 开始：2026-10-25 20:03:29.6949622 UTC（北京时间 2026-10-26 04:03:29）。
- 结束：2026-10-25 20:03:44.9041233 UTC（北京时间 2026-10-26 04:03:44）。
- 实际 SSH 退出码：255。
- 原始错误：`ssh: connect to host 10.147.17.32 port 22: Connection timed out`。
- GPU/RAM/磁盘与个人 SDK 工作者检查：均未取得远端结果，无法确认。
- 文档见证 supervisor 调用次数：0；child exit：无；新 GPU 作业：0。
- 没有连接重试、kernel 重试或修复尝试。

PowerShell 证据保存程序本身正常结束；其外层 0 不能冒充 SSH 的 255。`preflight_execution.json` 单独保留真实 SSH 返回值。由于连接尚未建立，也没有可复制的新远端 receipt、外层 exit 或 SDK 日志。

原本应在前置检查通过后执行的文档命令如下，本次**没有执行**：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_full_library_capacity.py --run full_library_document_v1'
```

## 已有作者证据的独立本地核对

以下仅来自已有本地作者聚合证据 `results/full_library_capacity_20261026/full_library_author_v1`；本次没有重新读取远端数据或执行作者任务。三个原始聚合文件逐字节复制至 `author_aggregate/`。

| 项目 | 作者已保存的实际结果 | 本次文档见证 |
| --- | --- | --- |
| SSH exit / child exit / timeout | 0 / 0 / false | 前置 SSH 255；child 无 |
| 数值状态 | TEXTOP_FULL_LIBRARY_8192_CAPACITY_PROBE_PASSED | 未执行 |
| 原始来源 / 原始帧数 | 8,278 / 4,825,012 | 无新验证 |
| 环境数 / rollout steps | 8,192 / 24 | 无 |
| 实际 Adam 更新 / 完整迭代 | 20 / 1 | 0 / 0 |
| transitions | 196,608 | 0 |
| policy / critic observation dim | 557 / 557 | 无 |
| 最大参数变化 | 0.17417466640472412 | 无 |
| value-function loss | 3.3047911274572776e+27 | 无 |
| surrogate loss | 3.3946707844734195e-08 | 无 |
| entropy | 41.149227142333984 | 无 |
| Torch allocated / reserved peak，bytes | 17,246,092,288 / 18,358,468,608 | 无 |
| 创建环境前 / 后 GPU free，bytes | 42,333,437,952 / 23,928,176,640 | 无新资源值 |
| 本设备 total，bytes | 50,907,250,688 | 无 |
| CPU affinity / Torch threads / allocator fraction | [0,1,2,3] / 4 / 0.65 | 未启动 |
| 新 checkpoint 数 | 0 | 0 |
| writers closed | true | 不适用 |
| supervisor elapsed，seconds | 195.4976666662842 | 未启动 |
| kernel receipt elapsed，seconds | 194.21594010666013 | 未启动 |

作者 loss 是有限数，但价值损失约 3.3×10^27 极大。作者结果只验证了完整可用参考库的容量及一次有限更新，不能证明稳定收敛、策略质量、物理通过率、40k 正式训练或生成器就绪。设备总采样包含非 Torch 分配和其他进程；0.65 不是整个 SDK 的显存上限。

作者 SDK 清理记录必须保留原义：`native_close_returned=false`、`teardown_fallback_used=true`、`shutdown_grace_seconds=20`、`shutdown_completion_verified=true`。写入收据、关闭输出写入者后原生退出没有在期限内返回，最终通过既有 fallback 结束进程；不能称原生退出成功。本次未启动 SDK，不能给本次伪造退出成功或 fallback 字段。

本地 `continuation_v1.json` 状态为 `FULL_LIBRARY_AUTHOR_VERIFIED_DOCUMENT_PENDING`，作者已通过而独立文档仍待验证；检查时 Windows PID39924 不存在。持久备份聚合收据报告 8,278 来源、6 个 cohort、`preprocessing_complete=true`、`output_writers_closed=true`，状态为 `PREPROCESSING_ALL_AVAILABLE_SOURCES_BACKED_UP_VERIFIED_FILTER_PENDING`。本次仅核对已保存收据，未重跑备份内容审计。

## 指纹比较

`comparison.json` 中六项本地文件 SHA256 与作者 receipt 均相同。文档运行对应字段全部为 null，不能将本地相同误报为两次执行相同。

| 内容 | 本地文件与作者 receipt 相同的 SHA256 |
| --- | --- |
| probe | 7d25ad8238977fa6397080732c7ddade1c7414305ae2f254fbe42d5326357e90 |
| activated capacity spec | 0f9dc72c4c616e673d343d5e8a49a091f48e40dc2cd9156d32a6e83854b19a40 |
| input verifier | 4b5784cf49e9b010de84be4194071cccb8f23a990fa179bb70b1157320265068 |
| recording runtime | ad3bb92f5836e5213450420e8f33deffd43443da7d451796adef3d6193ef6962 |
| paper filter config | e6b59d8fde8574d2c953279301dfc2df0c44ff88ac90683be995b7b97f0ff0d9 |
| persistent backup audit receipt | 8c742cdbeef4773bfc66fea828fb68d9c91d70f656e9d9385a70c31241fe307e |

作者原始参考顺序指纹：`0b786ba08bf8f33aa92cb4e91ed72cd5b56159c4e251cedf88f429269141c1f7`。各 cohort prepared/reference manifest 指纹原样保留在作者聚合 receipt；本次没有逐源清单或原始数组副本，也没有新的远端数据指纹可比较。

本地 supervisor SHA256 为 `4dcda5dfc9b81efc3a6719229505e1ad3cb731d52bd7beda836d455d22c52846`，与文档记录相同。输入文档 SHA256 另见 `document_input_hashes.json`。

## 文档与现实差异、保留状态

文档首段标题与“Verified so far”仍描述作者全库执行待验证；本地激活 spec 已是 `launch_enabled=true`，协调器和作者 receipt 已证明作者任务完成。文档后文明确要求依据实际协调器/收据判断，因此本次使用实际证据识别此状态滞后，没有修改文档。文档底部 disabled draft spec 哈希与 activated spec 不同是文档已预期的激活变化，不是本次校验失败。

真正阻断本次执行的是 SSH 端口连接超时。连接层原因尚未确定；不能据此推断服务器 kernel、驱动、依赖或训练程序失败。全库文档数值见证继续 pending，`physical_filter_executed`、`dataset_accepted`、`formal_filter_training_started`、`full_generator_training_started` 保持 false。已完成的持久预处理和作者容量结果不撤销，也不升级为独立文档通过。

本目录只包含聚合证据、输入配置快照、本次只读前置检查文本和报告。未复制原始数组、逐源索引、USD、checkpoint 或权重。
