# AMASS / HumanML3D 数量口径核查（2026-11-04）

结论：11424是复用的ReactiveBFM发布机器人语料数量，不能称为DreamControl-v2论文AMASS组的完整训练规模。

| 来源 | 位置 | 数量及口径 |
| --- | --- | --- |
| [DreamControl-v2](https://arxiv.org/html/2604.00202v1) | 表I，# Samples (K)；IV-A Training Data Mixtures | AMASS 21.8K、AMASS+Nymeria26.4K、Full-Mix29.6K训练样本。正文称AMASS组与OmniControl所用语料一致；该表没有给出与ReactiveBFM各条动作ID的对应关系。 |
| [ReactiveBFM v3](https://arxiv.org/html/2606.30362v3) | 附录C，表8 | AMASS-HumanML3D11424个motions、1156021frames，按发布60fps口径约5.35小时。 |
| [HumanML3D官方仓库](https://github.com/EricGuo5513/HumanML3D#-statistics) | Statistics与Data augmentation | 原人体数据14616个motions、44970条文字；来自AMASS与HumanAct12。官方另说明通过镜像动作及对应替换左右等文字进行翻倍增强。 |

本地已完成的数据构建收据见reactive29_pairing_parallel_20261104/completion.json：11424条29DoF G1轨迹、34132句发布文字，20Hz目标版本。构建划分8146训练、738验证、2540测试，总计11424；该整库条数不同于论文表I的训练样本数。本地名义ID清单14614基础ID/29228含镜像ID只是该次固定清单的实测统计，不能替代官方14616总体数据说明；此前缺3190的差集也是相对本地14614清单的名义覆盖差，不是DreamControl-v2差集。

论文数量与本地条数可能受到划分、增强、片段定义、筛选和处理版本的影响；尚无逐条ID/变换的对应证据，不确定具体差额成因，不将21.8K推定为21800个独立原始录制，也不宣称当前11424覆盖论文全部AMASS样本。

此核查不重新重定向已覆盖11424条、不热替换当前GRAB+Omni扩散模型、不改变当前四类生成/Pick PPO或原生SuperMimic等待全部参考的计划。
