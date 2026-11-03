# 代码与环境索引

## DreamControl / CLoSD

既往代码位于根目录 [scripts](../scripts/) 和 [configs](../configs/)，对应 [DreamControl 环境](../configs/env-dreamcontrol-r3.json)、[CLoSD 环境](../configs/env-closd-baseline-20261010.json) 及各报告。这里还保留跨仿真器、可视化和执行反馈诊断脚本；按脚本日期和报告协议选择入口。

## 按论文划分的代码

| 方法 | 固定来源 | 运行 / 适配代码 | 配置与环境 |
| --- | --- | --- | --- |
| PhyGile | [upstream](../baselines/phygile/upstream/) · [来源清单](../baselines/phygile/upstream_manifest.json) | [scripts](../baselines/phygile/scripts/) · [独立实现组件](../baselines/phygile/reimplementation/README.md) | [configs](../baselines/phygile/configs/) · [环境](../baselines/phygile/docs/environment.md) |
| PhyGile Unitree | [项目说明](../baselines/phygile_unitree/README.md) | [scripts](../baselines/phygile_unitree/scripts/) | [configs](../baselines/phygile_unitree/configs/) |
| ReactiveBFM | [upstream](../baselines/reactivebfm/upstream/) · [来源清单](../baselines/reactivebfm/upstream_manifest.json) | [scripts](../baselines/reactivebfm/scripts/) · [后续恢复](../baselines/paper_faithful_reproduction/recovery_v4/) | [configs](../baselines/reactivebfm/configs/) · [环境](../baselines/reactivebfm/docs/environment.md) |
| TextOp | [upstream](../baselines/textop/upstream/) | [scripts](../baselines/textop/scripts/) · [provider_tools](../baselines/textop/provider_tools/) | [configs](../baselines/textop/configs/) |
| TextOp 11 月 2 日冻结来源 | [快照清单](../baselines/textop/server_snapshot_20261102/SNAPSHOT_MANIFEST.json) | [快照脚本](../baselines/textop/server_snapshot_20261102/scripts/) · [工具](../baselines/textop/server_snapshot_20261102/provider_tools/) | [快照配置](../baselines/textop/server_snapshot_20261102/configs/) |
| SuperMimic Vibed 11 月 3 日适配 | [上游提交与导出边界](../baselines/textop/experiments/supermimic_vibed_20261103/README.md) | [训练模块](../baselines/textop/experiments/supermimic_vibed_20261103/supermimic/) · [50 Hz NPZ 加载](../baselines/textop/experiments/supermimic_vibed_20261103/vibed_npz_motion_source_20261103.py) | [本轮参数](../baselines/textop/experiments/supermimic_vibed_20261103/configs/stage1.json) · [已安装版本](../baselines/textop/experiments/supermimic_vibed_20261103/configs/environment.json) |

## 运行前阅读顺序

1. 先确定方法、固定源码版本和对应实验日期。
2. 查看环境、数据清单、资产与关节顺序，再查看运行参数。
3. 复跑使用独立输出目录，记录种子、实际迭代和来源摘要。
4. 区分历史草稿 / 禁用工具、测试夹具和真正执行的训练或评估入口。源文件存在不代表它已经运行或通过验证。

不同方法的 Torch、CUDA、IsaacLab、Newton、MuJoCo 与机器人资产版本不同。便携源码导出也不等于其部署环境、输入数据和受保护启动器已经完整包含。

[总索引](project_index.md) · [结果](result_index.md) · [数据与附件](datasets_and_artifacts.md)
