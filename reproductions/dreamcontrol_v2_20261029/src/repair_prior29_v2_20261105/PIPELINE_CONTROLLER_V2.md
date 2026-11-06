# 修正版参考生成自动接续

`pipeline_controller_v2.py` 等待新先验的 10,000 步完整检查点，核对实际 Adam、哈希、训练状态和锁。随后每项任务最多尝试 8 个不同候选。每次完整运行 1,000 步采样和运动保真后优化，达到运动学门槛的候选进入原生 PhysX 几何回放。每类至少有一个通过后按同一规则追加候选，逐项取得 500 条通过的参考。每个批次先保存原始候选和新后优化记录，按实际结果保留拒收原因、原生证明及源哈希。原生阶段通过临时兼容清单和同字节硬链接调用已冻结的入口，不改原脚本或旧参考。Pick 使用实际存储的抓取帧，既有回放证明只用于几何和接触检查。

## 启动配置

用以下结构单独生成计划。路径和哈希需要按现有服务器文件核实。必须先确定采样批量和每批写入预算，再绑定计划 SHA。

```json
{
  "schema": "repair_prior29_pipeline_plan_v2",
  "training_config": {"path": "/absolute/repair/config.json", "sha256": "<sha256>"},
  "output_dir": "/absolute/repair/new_references_v2",
  "sources": {
    "controller": {"path": "/absolute/repair/src/pipeline_controller_v2.py", "sha256": "<sha256>"},
    "sample": {"path": "/absolute/repair/src/sample.py", "sha256": "<sha256>"},
    "postopt": {"path": "/absolute/repair/src/postopt_v2.py", "sha256": "<sha256>"},
    "native_pick": {"path": "/absolute/legacy/src/stage3_pick500_replay.py", "sha256": "<sha256>"},
    "native_supported": {"path": "/absolute/legacy/src/stage3_supported_native_replay.py", "sha256": "<sha256>"},
    "task_conditions": {"path": "/absolute/legacy/src/stage3_supported_task_conditions.py", "sha256": "<sha256>"}
  },
  "urdf": {"path": "/absolute/upstream/g1_29dof.urdf", "sha256": "<sha256>"},
  "fk": {"path": "/absolute/legacy/g1_fk.py", "sha256": "<sha256>"},
  "native_runtime": {"path": "/absolute/legacy/runtime_bridge_spec.json", "sha256": "<sha256>"},
  "native_device": "cuda:1",
  "sample_python": "/absolute/existing/robot_diffusion/env/bin/python",
  "postopt_python": "/absolute/existing/robot_diffusion/env/bin/python",
  "postopt_env": {"JAX_PLATFORMS": "cuda,cpu", "CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
  "cohort_size": 32,
  "sample_batch_size": 2,
  "cohort_write_budget_bytes": 33554432,
  "checkpoint_wait_seconds": 172800,
  "phase_timeout_seconds": {"sample": 86400, "postopt": 14400, "native": 900},
  "seed": 20261105
}
```

`postopt_env` 需要复制既有 PyRoki 环境的实际路径和 JAX 分配参数。示例的 32 MiB 是每批最低预留，正式值要参考容量见证。`sources` 可另外绑定原生环境和模型依赖等关键源文件。控制器每阶段重查源文件、配置、每作业 12 GiB 显存和 32 GiB RSS、项目合计 24/64 GiB。阶段按顺序运行，运行磁盘底线沿用 8 GiB 加 456 MiB 的参考批次门槛。每批按此前实测的最大分配字节/候选动态预留，另记剩余参考估算。旧参考、失败前缀及所有新拒收文件保留。

```sh
"$PY" -B -u "$SRC/pipeline_controller_v2.py" --plan "$PLAN" --plan-sha256 "$PLAN_SHA"
```

## 状态与下游接口

- `checkpoint_binding.json` 记录正式检查点、训练结束状态和 CPU 读回收据。
- `pilot_receipt.json` 逐项记录真实单候选见证。每类最多 8 次，质量拒收保留并用不同 ID/噪声种子继续，所有尝试均计入每任务 1,000 个候选上限。8 次均拒收或出现执行异常时停止并保留前缀。
- `progress.json` 与分阶段启动、退出、资源峰值收据记录实际进展。每个任务最多尝试 1,000 个候选。批次无准入或预算耗尽时保留证据，控制器退出。
- `generation_manifest.json` 保存所有候选，不只最终入选的 2,000 条。每行包括真实生成词、原始样本、初噪声/特征内容哈希、条件坐标、后优化报告、可用的后优化轨迹、可用的原生证明，以及 `admitted` 和 `selected_reference`。最后一批的尝试数按缺口裁切，因此两项各恰好 500/任务。
- `generation_receipt.json` 的终态是 `ready_for_eval`。嵌入、几何指标和五指标报告应由正式评估入口从该哈希清单另行生成。控制器没有 PPO、SuperMimic 或扩散训练的启动入口。

本版本仅检查本机语法和 CPU 合成契约。采样后优化和原生准入全链需在已绑定服务器环境做实际见证。

