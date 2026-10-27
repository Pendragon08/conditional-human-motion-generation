# SMPL-X 模型安装文档独立验证

结论：PASSED。本 fresh agent 按 run-experiment 与 compute-env-contract 的 follows-doc 要求，直接执行安装文档中的服务器命令，未更改参数、代码、环境或文档，未覆盖 author 输出。执行前确认 model_witness_document.json 不存在。

## 实际执行结果

- 退出码：0。
- 状态：SMPLX_MODEL_CPU_FORWARD_PASSED。
- device：cpu；seed：42。
- female、male、neutral：vertices_shape 均为 [1, 10475, 3]，finite 均为 true；joints_shape 均为 [1, 127, 3]。
- female：本次见证结果 actual_ssm_body_frame_used=true；num_betas=16。
- stdout 包含完整成功 JSON，与取回的远程结果内容一致。stderr 为空，无额外警告或提示。
- 程序 source_sha256 与指定值 b15c278efbaa548eb3e95184b89e3ca5870928d5545c8566a9723e66a65466ec 一致。
- 归档 SHA-256 与文档值 88d35123fc97151bc258dc6f28f3967ae4c16117e4b452445f20598c43c43099 一致。

## 命令和完整记录

原样服务器命令见 command_verbatim.sh；它作为一个完整参数传给 SSH，未添加 wrapper、参数或环境前缀。SSH 端点与 key 只从私有 ledger 读取，未写入本证据目录。

- stdout.txt、stderr.txt：文档命令的完整标准输出与错误输出；本次无需隐藏或删改任何连接细节，因为输出中未出现这些内容。
- exit_code.txt、execution.json：文档命令退出码和 UTC 起止时间。
- preflight_stdout.txt、preflight_stderr.txt、preflight_exit_code.txt、preflight_execution.json：输出文件不存在的事前检查。
- model_witness_document.json：从原服务器原样取回的新结果 JSON。
- transfer_stdout.txt、transfer_stderr.txt、transfer_exit_code.txt、transfer_metadata.json：结果取回记录。
- invocation_metadata.json、validation.json：文档哈希、执行说明和逐项判断。

## 文档差异和附加观察

安装文档的执行与成功契约未发现偏差；文档在本次执行前后哈希相同。

provider ledger 的原始区块仍保留“模型权重缺失，不能做人体前向验证”的历史状态；其后追加的 model weight addition 已说明模型已添加，覆盖了该旧状态。本次读取时，追加区块仍写 independent witness pending；本次成功见证收据已解决该待验证项。为保持独立性，本 agent 未修改 ledger。

run-experiment 技能引用的 shared-references/compute-env-contract.md 实际位于技能目录的上一级 shared-references，已按技能内的 ../shared-references/compute-env-contract.md 路径读取。

## 结论边界

本次仅确认模型安装后的 CPU 前向运行。结果明确记录 gmr_retargeting_executed=false、g1_pairs_generated=0、new_262d_sequences=0、generator_training_started=false。它不证明完整手脸动作恢复、G1 重定向、配对数据、文本时间对齐、过滤、262D 编码或训练已完成。
