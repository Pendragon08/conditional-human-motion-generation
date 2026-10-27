# AMASS 获取与 HumanML3D 对齐准备（2026-10-21）

**更新：SSM 已实际下载并核验 30 条原始 SMPL-X 动作，详见 [SSM 核验记录](SSM_download_status_20261021.md)。人体模型及其余子集仍缺，新增 G1 配对序列仍为 0；本轮未执行重定向、动作过滤、262D 编码或生成器训练。下文的获取失败记录描述更早的尝试，按历史材料保留。**

## 已实际完成

- 重新核验 2026-10-18 固定官方版本的文本、索引和划分文件 SHA-256。
- 为官方划分内的 29,228 个动作 ID 生成逐 ID、逐文本行的时间对齐计划；对应 14,614 个原始索引片段、11,713 个来源录制文件。
- 来源包括 17 个 AMASS 子集及独立的 HumanAct12：HumanAct12 有 2,382 个动作 ID、1,191 个来源，不应冒充 AMASS SMPL-X 数据。
- 官方划分内有 84,138 行整段标注和 3,034 行有效片段标注；200 行异常时间标签隔离记录。与先前整个文本 ZIP 的统计口径不同，4 个划分外文本 ID 未并入本计划。
- 编写并实际通过 8 项标准库回归检查，覆盖预裁剪先于索引截取、子片段范围、`end=-1`、非整数采样率、无效时标、拒绝越界静默截断以及 HumanAct12 路线隔离。

`source_requirements.csv` 是按来源文件去重后的下载/映射清单。完整逐 ID 计划 `alignment_plan.jsonl` 保存在本地交付及忽略的 cache 内；它不是动作文件或训练 manifest。

## 官方获取状态

用户已完成 AMASS 主站登录，代理实际看到下载表格，并确认提供 SMPL-X G/N 等格式。BMLhandball 当前表格仅有 SMPL+H G，需单独适配或明确记录排除范围。

用户已明确同意本次 SSM SMPL-X G 的许可与下载；代理点击了许可弹窗的 Accept，随后弹窗关闭。但下载交互超时，未得到文件路径。使用页面实际暴露的精确官方 URL 尝试传输时，HTTP 200 返回的是 text/html，压缩文件头检查失败；该响应没有被冒充为动作压缩包。失败收据保存在 `SSM_acquisition_attempt.json`。

实际打开官方文件 URL 后，`download.is.tue.mpg.de` 显示 AMASS 登录表单；SMPL-X 模型下载页也显示独立登录表单。用户正在当前应用内页面完成登录。没有读取或导出密码、Cookie 或浏览器认证数据库。

## 数据到位后的处理约定

1. 核验来源、压缩包完整性、模型种类、性别、参数维度、动作字段与采样率，逐文件记录哈希。
2. 将 HumanML3D 原始录制路径映射到同一录制的 AMASS 导出。其原始重建使用 SMPL+H；GMR 官方入口要求 SMPL-X，不能通过重命名、截取 poses 列或补零就声称模型转换正确。跨导出需核实录制和时间对应。
3. 按官方 notebook 的顺序对齐：先按 `int(source_fps/20)` 抽帧，再裁掉数据源前缀，随后使用 index.csv 的 `[start_frame:end_frame]`，最后应用文本的 20-fps 子片段索引。EyesJapan/HDM05 前缀为 60 帧，TotalCapture/PosePrior 为 20 帧，Transitions 为 10 帧。参考导出帧号不能直接套到不同采样率的 SMPL-X 导出。
4. 镜像 ID 使用真实左右交换和几何镜像；不得只复制动作并更换左右文本。整段文本不能赋给任意截取片段。越界、非有限或不足两帧的标注应隔离。
5. GMR 到 G1 后，按名称映射 29 个关节，校核机器人描述、四元数顺序、FK、关节限位、时间连续性、穿地和接触滑移。过滤阈值须独立记录，不能伪称作者未公开阈值；运动学通过不等于物理可执行。
6. 使用现有版本化 G1 262D codec 编码并往返核验，记录 schema、骨架哈希、fps、初始根部位姿和每个文本片段来源。归一化只拟合训练集。

## 仍需解决的评估协议

官方动作 ID 划分存在原始录制来源交集：train/val 224、train/test 639、val/test 49。现有生成器训练入口拒绝来源组跨划分；必须明确是保留官方协议并记录录制重叠，还是另建录制隔离协议。没有改动现有训练入口或声称已解决该问题。

## 执行准备命令

从 `baselines/phygile` 运行（无需 GPU）：

```sh
python scripts/test_humanml_amass_plan.py
python scripts/plan_humanml_amass.py --public cache/humanml3d_public_20261018 --output cache/new-humanml-amass-plan
```

输出目录必须不存在；公共输入需与已提交的 provenance 匹配。命令只准备计划，不会自动生成机器人动作、训练 manifest 或启动训练。原始动作、人体模型和逐帧派生动作按各自许可保存在授权本地/服务器；GitHub 保存处理代码、来源哈希和统计报告。

## 官方来源

- [AMASS 数据与注册下载说明](https://amass.is.tue.mpg.de/)
- [AMASS 非商业科研许可](https://amass.is.tue.mpg.de/license.html)
- [SSM 本次许可](https://download.is.tue.mpg.de/amass/licenses2024/SSM/ssm_license_updated.html)
- [AMASS 官方 GitHub：教程和处理工具](https://github.com/nghorbani/amass)
- [SMPL-X 模型与获取要求](https://smpl-x.is.tue.mpg.de/)
- [GMR 官方数据准备](https://github.com/YanjieZe/GMR#data-preparation)
- [HumanML3D 固定版本的时间预处理代码](https://github.com/EricGuo5513/HumanML3D/blob/9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23/raw_pose_processing.ipynb)
