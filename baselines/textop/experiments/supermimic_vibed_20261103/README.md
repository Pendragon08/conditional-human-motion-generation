# SuperMimic Vibed：TextOp 50 Hz 适配

这是 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/0525a23a2dcfa61f8543f98b84c075fe66d9d286) 的独立适配导出，固定上游提交为 `0525a23a2dcfa61f8543f98b84c075fe66d9d286`。上游 MIT 许可证保留在 [LICENSE](LICENSE)。

## 文件入口

| 内容 | 文件 |
| --- | --- |
| 策略、世界模型与训练循环 | [supermimic/train.py](supermimic/train.py) · [networks.py](supermimic/networks.py) |
| Newton / MuJoCo-Warp 环境与 G1 控制 | [env.py](supermimic/env.py) · [robot.py](supermimic/robot.py) |
| NPZ 原帧加载、名称映射、运动库 | [NPZ loader](vibed_npz_motion_source_20261103.py) · [50 Hz adapter](vibed_motion_adapter_50hz_20261103.py) · [motion.py](supermimic/motion.py) |
| 明确使用的训练参数 | [stage1.json](configs/stage1.json) |
| 实测环境版本与固定源码 | [environment.json](configs/environment.json) |
| 输入清单格式 | [manifest schema](configs/motion_manifest.schema.json) |
| 验证范围与来源摘要 | [validation_summary.json](results/validation_summary.json) · [export manifest](EXPORT_MANIFEST.json) |

本次导出只改变 `robot.py` 的三个资产路径，使其通过 `VIBED_G1_URDF`、`VIBED_G1_MJCF`、`VIBED_G1_USD` 或本目录 `data/` 读取。其余导出 Python 文件与已运行部署逐字节一致。原部署 SHA-256 与公开文件摘要分别记录；这个便携导出尚未另行执行，不替代已完成的服务器运行见证。

## 数据与资产

输入是记录列表组成的 JSON 清单，每项含 `path, sha256, bytes, frames, joint_names, body_names`。实际 NPZ 保持原始 **50 Hz 全部帧**，至少两帧；加载器核字节数、SHA-256、名称、数组和帧数。没有把 50 Hz 数据按上游默认 30 Hz 再重采样。训练清单 8077 段、5,510,162 帧；独立生成留出为 564 段、58,674 帧，含 6 条两帧片段。

原始轨迹、私有文件路径、机器人网格、USD 和模型权重未包含在源码导出中。数据获取状态见[数据与附件](../../../../docs/datasets_and_artifacts.md)，各资产按原许可取得。G1 训练资产使用 29 个实际 sphere/capsule 碰撞体；原 USD 的父 Xform CollisionAPI 需要转换到实际 Gprim。纯物理 USD 去除了外部材质引用，保留经核对的质量、惯性与关节属性。完整物理资产的摘要及验证范围见验证文件；普通原 USD 不能被视作已经完成此适配。

## 运行环境与状态

需要 Python 3.12、固定 IsaacLab / Newton 源码及列出的包版本。采用 `--visualizer none --device cuda:0`；`cuda:0` 是进程可见设备索引，设备绑定由运行环境提供。服务器实测采用 CUDA 12.6、Torch 2.11、Warp 1.17 和 NVRTC_DISABLE_PCH=1；上游容器与默认安装脚本不代表本次已经执行的环境。

2026-11-03，本轮 2048 环境 / 4000 次优化迭代的正式训练已完成，共 20,611,072 次物理交互。三个检查点、160 行训练指标及实际优化器计数已由独立 CPU 审核通过，见 [训练完整性记录](results/main_training_integrity.json)。前 500 次只更新世界模型，此后更新策略和价值网络。新策略的 564 条独立质量评估尚无结果；检查点仍在服务器，不属于本次源码上传。记录冻结于本日期，不是实时任务面板。

本目录的 [evaluate.py](supermimic/evaluate.py) 保留上游按训练终止条件计算的 14 身体回报评估。它只记录步前状态，不能直接作为历史 **30 身体、全参考、0.3 米**主指标。后续可比评估需要收集初始及末步状态、保留全部 564 分母，并明确引擎、资产、控制、初速度和推理精度差异。既有结果见 [TextOp 索引](../../INDEX.md)。

本次训练使用共同锁、受限缓存/输出、GPU 与磁盘监测及原任务身份保护；部署监督器包含机器专用身份与路径，未作为通用启动器导出。新环境必须自行建立相应资源配置。训练参数清单是本轮配置记录，不是完整安装命令。

[返回 TextOp](../../INDEX.md) · [全仓库代码入口](../../../../docs/code_index.md)
