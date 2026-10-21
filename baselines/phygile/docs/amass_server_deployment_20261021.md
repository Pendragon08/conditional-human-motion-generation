# SSM 服务器部署与 AMASS 获取状态（2026-10-21）

**当前仅完成 SSM 子集的服务器部署及依赖扩展。完整 AMASS、SMPL-X 人体模型、G1 配对动作和生成器训练均未完成。**

## 已完成的工作

SSM 压缩包已从用户授权的本地副本上传到原实验服务器，在服务器重新计算 SHA-256、解压并读取必要数值数组。30 条 SMPL-X 动作、3 份 stage-I 参数和 1 份许可文件均已核验；stage-I 参数不包含完整人体模型权重。压缩包 SHA-256 为 `42f91a1a0e74056b918e714192784b3aa9ae6a300031c8c94a5c2c65701741e4`。

数据与部署材料目录：

`/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021`

压缩包位于该目录的 `SSM_smplx_g.tar.bz2`，解压文件位于 `extracted/SSM/`，服务器核验收据为 `ssm_verification.json`。相关 HumanML3D 文本、索引和处理代码位于 `ingestion/`。本次未删除磁盘中的任何旧数据。

原重定向环境缺少 `smplx`，首次失败记录予以保留。随后创建独立依赖扩展，安装 `smplx==0.1.28`，通过明确的路径复用原有大依赖，没有更新父环境的包：

`/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021`

环境 spec 为 `141547bf`。作者执行和独立执行者按文档原样运行均退出 0，状态为 `CPU_WITNESS_PASSED`。seed 42 的 16 个 G1 限位内姿态中，MuJoCo 与本实现骨架的最大位置差为 `4.56e-7 m`，最大旋转矩阵元素差为 `1.41e-6`，均小于 `1e-5`。可选 `xrobotoolkit_sdk` 未安装提示不影响这项检查。

该检查仅覆盖 CPU 运算、固定依赖和机器人运动学。SMPL-X 人体模型前向计算、GMR 真人动作重定向、过滤、262D 编码及训练均没有执行；没有使用 GPU 启动新训练。独立验证见 [公开收据](../results/amass_server_20261021/document_witness/status.json)及 [原始输出](../results/amass_server_20261021/document_witness/raw-output.txt)，复跑约定见 [环境说明](amass_server_environment_20261021.md)。首次执行者收据中的 `independent_document_witness_completed=false` 是当时状态，保留原样；后续独立验证另存收据。

## 文件大小与剩余容量

下表 MB、GB 使用十进制单位。服务器目录大小为采样时的表观字节数，后续日志可能带来小幅变化；不把共享的原有 PyTorch 计为本次新增。

| 项目 | 字节数 | 约合 |
|---|---:|---:|
| SSM 压缩包 | 38,268,098 | 38.3 MB |
| SSM 解压后文件 | 78,104,561 | 78.1 MB |
| 同时保留压缩包与解压文件 | 116,372,659 | 116.4 MB |
| 服务器 SSM、文本索引和部署目录 | 138,970,938 | 139.0 MB |
| 独立依赖扩展环境 | 20,599,874 | 20.6 MB |
| 本次两个新增目录合计 | 159,570,812 | 159.6 MB |
| 数据盘 `/data` 剩余 | 19,906,146,304 | 19.9 GB |
| 系统盘 `/` 剩余 | 4,436,242,432 | 4.4 GB |

完整 AMASS 的体积随子集、SMPL+H / SMPL-X、性别版本、是否保留压缩包而变化。本次尚未取得所有选定导出包的精确大小，**总大小未知**，没有用 SSM 的大小外推。当前约 19 GB 余量不能作为完整下载、解压、重定向和训练产物均可容纳的依据。后续批量获取需逐包确认大小与余量，并为解压和中间产物预留空间。

## GitHub 保存内容与许可

仓库新增 [AMASS 数据目录](../datasets/AMASS/README.md)及 [机器可读资产清单](../datasets/AMASS/assets.json)，保存来源、取得状态、大小、哈希、处理脚本和核验记录。

[AMASS 官方许可](https://amass.is.tue.mpg.de/license.html)允许符合许可的非商业科研使用，但限制再分发；[SMPL-X 模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html)也限制模型再分发。因此本次没有将原始 AMASS 动作、逐帧派生动作或人体模型权重上传到 GitHub。原始 SSM 已保存在用户自己的本地 D 盘和服务器目录。这里的目录及清单不能替代受许可约束的数据下载。

## 尚未完成及下一步

固定 HumanML3D 索引涉及 17 个 AMASS 来源，本次取得其中 SSM 一个，其余 16 个仍未取得；另有独立 HumanAct12 来源。这个复现清单不等于 AMASS 网站所有数据集。详见 [获取与对齐准备](amass_acquisition_and_alignment_20261021.md)。

SSM 所需的兼容 SMPL-X female 人体模型仍缺，安装 `smplx` Python 包不包含这些权重。模型需从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/)登录并按模型许可取得，当前没有已取得模型文件可供核验。

SSM 已按文件名对应 30 个 HumanML3D 录制、60 个含镜像的文本 ID，但原 HumanML3D 的 SMPL+H 时间轴与本次 SMPL-X 导出的对应关系仍未验证，不能直接按同名截段宣称配对成功。后续顺序是：取得兼容模型与所需原始导出 → 验证跨导出时间对应 → G1 重定向 → 文本时间段对齐及镜像 → 动作过滤 → 262D 编码。正式训练还须处理来源分组与官方划分间的泄漏风险。

**本批新增 G1 配对序列为 0，新增 262D 序列为 0，完整生成器训练未开始。** 本报告及 CPU 检查不构成完整 PhyGile 复现或论文性能验证。
