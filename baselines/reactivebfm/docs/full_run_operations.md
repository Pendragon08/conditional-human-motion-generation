# 完整训练运行与故障恢复

当前训练：服务器 `/home/ouyanghao/reactivebfm-full-20261021/training_v1`；会话 `reactivebfm_full_v1`。当前从头训练的是运动规划器，跟踪器使用官方固定预训练权重。

## 查看进度

本地目录 `D:/具身智能/ReactiveBFM复现_20261021/full_training_20261022` 的 `current_training_status.json` 记录两rank最近一次状态和真实更新数，`worker_events.jsonl` 为追加式备份日志。历史的 `worker_launch*.json` 保留进程启动信息；运行中的进程ID以 `worker_state.json` 的最新worker_pid为准。阶段报告在首次检查点处理后生成，GitHub Release的每个检查点报告为独立文件，不覆盖历史。

本地电脑须保持开机、网络和SSH/GitHub授权可用。备份程序只阻止空闲睡眠，不能阻止手动关机/休眠。训练在服务器独立运行。空间不足时保存等待；NCCL默认3小时超时可能终止任务。备份或训练进程退出后不会自动重启。

## 重启本地备份进程

先查看worker_state中的进程是否仍在运行，避免启动重复进程。程序用文件锁阻止重复实例。使用既有Python运行 scripts/backup_full_run.py，并传入 configs/full_backup_worker_v1.json；后台启动必须使用隐藏窗口，stdout/stderr使用新的时间戳文件名，保留历史日志。不要覆盖 worker_state.json、旧报告、旧检查点或历史启动日志。

已有本地文件与服务器哈希不同时，程序保留文件并拒绝继续处理。应保存差异证据、核对来源后再选用新目录，不能简单删除文件绕过检查。检查点模型和优化器均已验证上传、阶段报告/证据上传后，旧副本才进入清理；最新服务器副本保留。

## 训练退出后的恢复

首先检查实际训练进程、状态、stdout/stderr和退出原因，确认本任务已退出；不要停止其他任务，也不要在仍运行时重复启动。保存当前全部结果。底层检查点写入不是模型/优化器两件原子事务，不能只按最高文件名选择。

恢复助手 `prepare_verified_resume.py` 仅核验文件并输出计划，不启动GPU任务。它读取ready清单、流式核对两件SHA256/大小、核对训练配置哈希，并要求恢复输出目录不存在。只通过该助手选中的完整对才能用于后续恢复；未通过的文件保留原位调查。

服务器调用参数为：

- --run-dir：原有 training_v1 目录。
- --new-output：一个全新的恢复输出目录，不能复用原目录。
- --config：原 configs/full_training_v1.json（原始字节和哈希）。
- --plan-output：一个尚不存在的计划JSON文件。

计划中包含环境变量与torchrun参数列表；执行前重新核对GPU资源。恢复保留全局1024和原百万步目标，读入模型与优化器，但未保存随机数/数据迭代器的完整状态，因此不承诺逐位回放。恢复助手目前只完成虚拟字节协议测试，尚未对本次完整训练检查点进行真实恢复演练。

新尝试须使用新的备份配置/本地输出/evaluation_root/Release tag，保留旧尝试。这样重新生成的同名step检查点不会覆盖或冲突于旧Release的不可变资产。将新配置的server_output指向恢复目录，再启动相应备份进程。不要改旧worker配置后让它误处理另一尝试。

## 证据边界

PHYSICAL_CLOSED_LOOP_EXECUTED仅表示请求的控制步和重规划执行完毕；高度阈值为诊断代理，不能称论文成功率。FEEDBACK_CONSISTENCY_PASSED仅表示保存的实际状态与输入前缀接线一致。只有模型/优化器准备好、GitHub校验通过且阶段证据上传后，该检查点才算备份完成。
