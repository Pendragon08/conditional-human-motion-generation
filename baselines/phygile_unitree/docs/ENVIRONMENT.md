# 环境与独立执行见证（2026-10-22）

本次是框架迁移和冻结策略验证。所有下述指令均不执行 PPO 更新。
新 PPO 续训仍须先验证用户本地能查看实际训练窗口；目前用户暂时无法查看，未启动。

## 固定环境

- Isaac 路线复用服务器已有 Python 3.10.15、Isaac Sim 4.5.0、Isaac Lab 2.1.0、PyTorch 2.5.1+cu118、RSL-RL 2.3.1；不升级共享环境。
- Unitree RL Lab commit: `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`。
- 本地独立环境：Python 3.12.14、PyTorch 2.14.1+cpu、torchvision 0.29.1、tensordict 0.15.2、NumPy 2.5.4、SciPy 1.16.2。
- Mjlab 1.2.0、MuJoCo 3.5.0、MuJoCo-Warp 3.5.0、Warp 1.12.0、RSL-RL 5.0.1；仅加载冻结策略，不使用新版 RSL 算法替代论文 MoEPPO。
- Unitree RL Mjlab 0.0.1 commit: `1425b15f73bd4095f0df53709d7c389c3eb9e790`。
- 声明规格为 `configs/environment_plan_v2.json`，完整解析锁定为 `configs/mjlab_resolved_requirements_v2.txt`。初始规格保留以记录依赖解析时的 Torch 升级。

Windows 下 MuJoCo 的原生文件读取在中文包路径上失败。英文路径 `D:/phygile_unitree_env_20261022` 是指向原 `D:/具身智能/PhyGile复现/Unitree框架适配_20261022/mjlab_env` 的目录联接，未移动、替换模型或修改已安装的官方源码。必须通过英文路径的 Python 启动。

## 本机已经部署的资产

`D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets` 含原 checkpoint、原 MoE 源文件、完整控制契约、合成数值夹具和四个现有测试片段。
父 checkpoint SHA256: `bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336`。
动作来源为原复现工程的四个保留测试片段；它们不是完整 HumanML3D 或论文全量基准。
原始动作和模型资产保持私有，只发布索引、哈希、来源、代码与汇总指标。

## 独立见证必须逐字执行的两条 PowerShell 指令

每条指令单独执行。输出目录必须尚不存在；不修复、不删除失败产物。第二条是实际 Unitree G1、MuJoCo-Warp CPU 三个控制周期（十二个物理步），不是 import 检查或模拟返回值。

```powershell
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts/mjlab_kernel_witness.py' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/doc_witness_kernel_v1.json'
```

预期退出码 0、输出 `MJLAB_CPU_KERNEL_WITNESS_PASSED`；执行带种子的 Torch 反向传播和真实 MuJoCo-Warp 两个物理步。

```powershell
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts/mjlab_evaluate.py' --unitree-checkout 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/vendor_unitree_rl_mjlab' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/doc_witness_g1_v1' --clip 0 --max-steps 3
```

预期退出码 0、`receipt.json.status=BOUNDED_PHYSICS_PROBE_COMPLETED`、`frames_evaluated=3`、`physics_steps=12`、`action_scale_offset_exact=true`、`optimizer_steps=0`。截断探针不能被称为完整动作成功。

## 服务器图形推理与后续训练

使用 `scripts/run_isaac.sh scripts/lab_run.py`；`--mode evaluate` 从第零帧记录第一次终止前的真实状态。`--mode preview` 只预览；`--mode train` 是单独分支，须有 `LOCAL_VIEWER_VERIFIED` 回执。GUI 渲染 GPU0，物理和策略 GPU1；Xorg :10 只能由 GPU0 正常呈现。每次运行使用全新输出目录，不覆盖已有检查点。原生 GUI 的图像、日志与回执已保存；用户本地实时查看尚未验证。
