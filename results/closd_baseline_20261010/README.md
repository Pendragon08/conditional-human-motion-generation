# CLoSD：条件动作生成与物理执行基线复现报告

实验日期：2026-10-10；报告补充：2026-10-17；GitHub 同步：2026-10-21。

[下载完整报告与视频（ZIP）](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/closd-baseline-report-20261017/CLoSD_baseline_report_20261017.zip) · [HTML 报告](baseline_report.html) · [结构化指标](summary.json) · [附件清单](execution_demo_manifest.json)

## 如何查看

下载 ZIP 并解压，打开其中的 `baseline_report.html`。保持 `execution_demos_20261006/` 和 `previews/` 与 HTML 位于同一目录，即可离线播放全部视频。GitHub 的 HTML 文件页显示源代码；直接阅读本页可查看摘要。发布记录见 [Release](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/closd-baseline-report-20261017)。

## 复现范围

使用官方已发布检查点，在 Isaac Gym 中完成 HumanML3D 文本动作分支的生成与物理执行评测：重新评估作者公开动作，并重新采集 20,000 条完整、未提前终止的执行轨迹，分别运行官方评估器。

执行对象是 SMPL 仿真人体角色。本次没有新增 PPO 训练、G1 执行或 sim-to-real，不据此声称合成数据改善了机器人训练。

## 已归档指标

以下数值直接取自已有 `summary.json`，均为 10 次评估重复的均值 ± 报告中的 95% 置信区间。

| 指标 | 重评作者公开动作 | 本地新生成动作 |
|---|---:|---:|
| R@1 | 0.3847 ± 0.0031 | 0.3806 ± 0.0024 |
| R@3 | 0.6926 ± 0.0026 | 0.6908 ± 0.0035 |
| FID | 1.7533 ± 0.0050 | 1.2972 ± 0.0180 |
| 文本—动作匹配距离 | 3.6474 ± 0.0009 | 3.6120 ± 0.0087 |
| Diversity | 8.2266 ± 0.0927 | 8.3587 ± 0.0960 |

两列使用同一方法的不同动作样本集；FID 差异不构成新算法改进证据。官方录制会筛除提前终止回合，因此这些质量指标也不等于所有生成尝试的成功率。完整物理指标、筛选规则与评价边界保留在 HTML 中。

## 视频内容

- **任务演示：9 段。** 触达、踢击、拳击、坐下各包含两组官方已发布权重的固定录制，另有一段起身状态截取。未微调策略在原固定录制中没有触发起身，不补造对应视频。
- **生成动作预览：3 段。** 固定展示新采集数组的前 3 条轨迹，没有按视觉效果挑选。
- 演示来源于 2026-10-06 的冻结策略记录，和 2026-10-10 的 20,000 条动作评测分别记录。视频呈现记录轨迹的骨架回放，不是原生相机录像或真实机器人视频。

全部视频及图像保持原文件字节；[发布校验清单](github_publication_manifest_20261021.json) 提供大小与 SHA-256。本次只补齐报告发布，不修改原评估指标。

## 相关入口

- [原实验说明](../../docs/closd_baseline_20261010.md)
- [官方权重多任务评估](../../reports/13_CLoSD_official_checkpoint_eval_20261001.md)
- [CLoSD 官方项目](https://guytevet.github.io/CLoSD-page/) · [官方代码](https://github.com/GuyTevet/CLoSD)

原始人体动作数组、SMPL 资产、数据集及模型权重不包含在此报告包中。
