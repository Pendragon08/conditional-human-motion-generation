# AMASS 数据清单与服务器入口

2026-10-21后续更新：用户已停止全量下载，现按HumanML3D索引处理。SSM已产出首批66条真实G1配对序列，见[处理报告](../../reports/HumanML3D到G1配对数据处理报告.md)。下方数量为较早的获取快照。

本目录保存 PhyGile 复现涉及的 AMASS 来源、获取状态、文件哈希及服务器部署入口。原始动作和人体模型保存在用户授权的本地及服务器目录。

AMASS 官方科研许可含再分发限制，参见 [官方许可](https://amass.is.tue.mpg.de/license.html)。本仓库没有上传原始 AMASS 数据、逐帧派生动作或 SMPL-X 模型权重；本目录不能代替数据集下载。

## 当前状态（2026-10-21快照）

后续15个已获用户同意的动作包合计57,071,473,311字节，正在后台下载；本轮快照中9个已在本地核验、8个已在服务器核验，另有此前已完成的SSM。TotalCapture依用户要求跳过。各状态的确切时间见 [assets.json](assets.json)，最新过程见 [下载与模型安装报告](../../docs/amass_remaining_download_20261021.md)。任务结束后的新快照保存于 `results/amass_remaining_20261021/completed_*/`。

SMPL-X去头髻版模型包已取得并安装，含female、male、neutral；三种模型均通过CPU前向和独立文档验证。版本、模型路径及验证范围见 [模型安装说明](../../docs/smplx_model_installation_20261021.md)。原始模型权重不在Git仓库中。

## 此前取得的SSM

- SSM SMPL-X G：38,268,098 字节，SHA-256 为 `42f91a1a0e74056b918e714192784b3aa9ae6a300031c8c94a5c2c65701741e4`。
- 服务器重新读取通过：30 条动作、3 份 stage-I 参数、1 份许可文件。
- 已按来源文件名匹配到 30 个 HumanML3D 原始录制、60 个文本 ID（含镜像）；跨导出时间对应及机器人动作尚未生成。
- 完整生成器训练未开始，SSM的模型加载验证不等于配对动作已经生成。

完整状态见 [assets.json](assets.json)、[SSM 获取核验](../../docs/SSM_download_status_20261021.md)及 [服务器部署记录](../../docs/amass_server_deployment_20261021.md)。

## 服务器目录

原始动作及部署材料：

`/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021`

独立依赖扩展：

`/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021`

该环境的初次检查只覆盖依赖和CPU机器人运动学，见 [原环境说明](../../docs/amass_server_environment_20261021.md)。后续模型前向已另行验证，见上文模型安装说明；真人动作重定向和配对处理尚未执行。

## 获取与校验入口

1. 从 [AMASS 官网](https://amass.is.tue.mpg.de/)注册登录，依子集许可取得动作；从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/)取得兼容人体模型。
2. 使用 [来源与时间计划脚本](../../scripts/plan_humanml_amass.py)核对 HumanML3D 索引、文本时间段、镜像及来源划分。
3. 使用 [SSM 核验脚本](../../scripts/ingest_ssm_archive.py)核对本次压缩包；使用 [服务器传输脚本](../../scripts/stage_ssm_on_server.py)传输及校验。传输脚本不会自动接受未确认的许可或获取账号密码。

AMASS 不直接提供 HumanML3D 的文本标注；本仓库保留官方 HumanML3D 文本、索引与划分的固定版本来源。HumanML3D 所需的 HumanAct12 属于独立来源，未被冒充为 AMASS SMPL-X 数据。
