# AMASS 重定向依赖扩展与 CPU 验证（2026-10-21）

环境 spec：`configs/gmr-amass-env-spec.json`，规范化 JSON SHA-256 前 8 位为 `141547bf`。

本环境是在原服务器已有重定向依赖之上创建的独立扩展，安装 `smplx==0.1.28`。大依赖通过明确的 `.pth` 路径复用；父环境版本变化时，验证程序会拒绝版本漂移。没有更新父环境的包。

环境目录：

`/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021`

SSM 数据目录：

`/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021`

SSM 的压缩文件 SHA-256：`42f91a1a0e74056b918e714192784b3aa9ae6a300031c8c94a5c2c65701741e4`。服务器已用原环境重新读取 30 条动作、3 份配套参数和许可文件，保留数据收据 `ssm_verification.json`。

## 文档独立验证命令

下面的命令在服务器上执行，输出文件必须不存在。它只使用 CPU，读取已部署的固定版本脚本、骨架及环境 spec，输出一份独立检查收据：

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021/environment_extension_v1/witness_gmr_amass_cpu.py --gmr-root /data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR --ingestion-root /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021/ingestion --spec /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021/environment_extension_v1/env-spec.json --output /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021/environment_extension_v1/cpu_witness_document.json
```

预期退出码为 0，收据 `status` 为 `CPU_WITNESS_PASSED`，`spec_hash` 为 `141547bf`。检查覆盖固定版本导入、seed=42 的 PyTorch CPU 运算，以及 16 个随机限位内 G1 姿态的 MuJoCo / 262D 骨架 FK 一致性；位置最大绝对差和旋转矩阵最大绝对差均须小于 `1e-5`。

首次执行者的记录是 `cpu_witness_author.json`；独立按文档执行者应生成上文的 `cpu_witness_document.json`。文件缺失或失败即表示对应检查尚未确认，不能只凭环境目录存在宣称可用。

## 验证边界

这只验证依赖与机器人运动学，未执行 SMPL-X 人体模型前向计算、GMR 真人动作重定向、动作过滤、新 262D 数据生成或训练。模型权重仍需从官网授权取得；没有完整人体模型时，`environment_fully_ready_for_humanml_processing` 必须保持 false。

当前磁盘只能确认 SSM 这一小批材料具备足够余量。完整 AMASS 各包的精确大小尚未全部核实，不能据此宣称全量已安装或可全部放入当前剩余空间。
