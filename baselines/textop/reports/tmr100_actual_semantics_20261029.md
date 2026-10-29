# 用户100轮语义评价器实际留出结果

真实100轮/101500次AdamW（从83703安全状态补17797，0skip）已完整保存、生产精确重载并正常0退出。CPU导出与fresh独立CPU复核均0/closed；真正mean embedding作者/fresh三数组逐位一致。两轮doc状态finding及一次schema标签激活拒绝原件保留，修正文档/禁用schema后实际接受；没有重跑旧模型或夹具。

全部1880条输入：32个原VAL family、188个独立标注段×5固定seed×2mode，包含全部7个物理失败。真实进程/监督/outer0，closed/无worker/共锁释放；终端collector独立重算全部标量。

| 指标（5seed等权） | 条件CFG5 | 无条件 |
|---|---:|---:|
| Rprecision_top1 | 33.6170% | 23.0851% |
| Rprecision_top2 | 49.7872% | 35.5319% |
| Rprecision_top3 | 60.5319% | 44.4681% |
| MMdist | 30.49171190 | 35.15665576 |
| FID | 146.26095444 | 132.73107566 |

文本检索与MMdist优于无条件，FID反而更高；不得概括为所有质量均改善。参考motion检索baseline Top1为28.7234%，模型存在有限样本/重复文本/精确标注positive定义等限制，不能用此baseline为生成质量背书。

预先固定协议：eval/sample_mean、256D raw latent、不逐样本L2，raw float64 Euclidean，32候选；最后28查询加4固定distractor，每个188查询均评分一次，精确距离ties取平均rank。每seed188参考/188生成、ddof1的raw-feature FID；188<256的协方差奇异/有限样本有偏，不加任意jitter，不池化重复参考为940独立参考。全部5seed原值保留，标准差只是描述性总体std，不是置信区间。

这是用户缩短为100轮的兼容69D/BABEL评价器与32-family独立短段诊断适配，不是全1774VAL/连续转折或作者完全相同协议，不称论文完整复现。原Huber条件更差、strict物理失败和所有负面不变。0生成器/旧teacher追加训练。

所有latent/source-join/数组和权重仅server+D私有；此报告只公开聚合数据。

receipt SHA256: c34b9ca92bac9b69115503c20399706e2783af3a266e779b97c75620f6948d5b

全部seed原值：

```json
{
  "conditional_cfg5": {
    "Rprecision_top1": {
      "equal_seed_mean": 0.3361702127659575,
      "descriptive_seed_population_std": 0.010848977688495286,
      "all_five_seed_values": [
        0.3404255319148936,
        0.3404255319148936,
        0.3191489361702128,
        0.32978723404255317,
        0.35106382978723405
      ]
    },
    "Rprecision_top2": {
      "equal_seed_mean": 0.49787234042553197,
      "descriptive_seed_population_std": 0.009865551590952874,
      "all_five_seed_values": [
        0.5053191489361702,
        0.48936170212765956,
        0.48404255319148937,
        0.5106382978723404,
        0.5
      ]
    },
    "Rprecision_top3": {
      "equal_seed_mean": 0.6053191489361701,
      "descriptive_seed_population_std": 0.020015838002368915,
      "all_five_seed_values": [
        0.6117021276595744,
        0.5851063829787234,
        0.6170212765957447,
        0.6329787234042553,
        0.5797872340425532
      ]
    },
    "MMdist": {
      "equal_seed_mean": 30.491711900113508,
      "descriptive_seed_population_std": 0.11916691409476446,
      "all_five_seed_values": [
        30.60302141462636,
        30.341086399384213,
        30.596309555625563,
        30.35217298346308,
        30.56596914746832
      ]
    },
    "FID": {
      "equal_seed_mean": 146.26095444021183,
      "descriptive_seed_population_std": 2.8589871702377776,
      "all_five_seed_values": [
        147.39010035359468,
        142.86470913749116,
        142.91431658319186,
        149.81031133479587,
        148.3253347919855
      ]
    }
  },
  "unconditional": {
    "Rprecision_top1": {
      "equal_seed_mean": 0.23085106382978723,
      "descriptive_seed_population_std": 0.007215244662899222,
      "all_five_seed_values": [
        0.22872340425531915,
        0.23404255319148937,
        0.21808510638297873,
        0.2393617021276596,
        0.23404255319148937
      ]
    },
    "Rprecision_top2": {
      "equal_seed_mean": 0.3553191489361702,
      "descriptive_seed_population_std": 0.01627346653274293,
      "all_five_seed_values": [
        0.324468085106383,
        0.3617021276595745,
        0.3617021276595745,
        0.35638297872340424,
        0.3723404255319149
      ]
    },
    "Rprecision_top3": {
      "equal_seed_mean": 0.44468085106382976,
      "descriptive_seed_population_std": 0.014893617021276588,
      "all_five_seed_values": [
        0.4308510638297872,
        0.4627659574468085,
        0.43617021276595747,
        0.4308510638297872,
        0.4627659574468085
      ]
    },
    "MMdist": {
      "equal_seed_mean": 35.15665576456638,
      "descriptive_seed_population_std": 0.4155794966362062,
      "all_five_seed_values": [
        35.869984722802386,
        34.67326915053533,
        35.28466095336802,
        35.12012886936367,
        34.8352351267625
      ]
    },
    "FID": {
      "equal_seed_mean": 132.7310756580549,
      "descriptive_seed_population_std": 4.923213036210005,
      "all_five_seed_values": [
        136.07076298774913,
        129.791685848962,
        136.15326128723586,
        137.23776076916846,
        124.40190739715898
      ]
    }
  }
}
```
