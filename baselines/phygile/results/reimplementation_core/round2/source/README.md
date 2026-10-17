# 按论文重建的生成模块核心（未训练）

此目录是根据 PhyGile v3 §III-B/C、附录 A 自行实现的研究代码。它与 `../upstream/` 中的作者代码分开保存。**目前没有训练后的动作生成权重、语言动作生成质量实验或生成—控制闭环结果；不能作为已复现完整 PhyGile 的证据。**

明确公开的规格：262 维输入；冻结 DistilBERT 的 768 维 token；8 层 AdaLN Transformer decoder，宽度 512、FFN 1024、4 头、dropout 0.1；TP-MoE 每层 12 个两层专家，门控 768→512→512→12，SiLU；空间 mask γ=24、β=0.25；负载均衡损失系数 0.01；x0 回归目标；训练文本置零概率 0.1；CFG 2.5。实现包含公式 7–11、13–14 的数学组件；公式 12 只启发了一个明确标注假设的固定边界采样器，尚不包含物理前缀闭环。

已实现范围：参数空间混合 TP-MoE、有效 token 的负载均衡、按帧/文本掩码的 decoder、块选择归一化、ASFO 权重、x0 DDPM 目标与采样、标准及负提示 CFG、固定前缀/后缀的采样约束。测试使用人工数值张量，只验证公式、形状、掩码与梯度，不属于生成质量或物理可执行性实验。

未明确公开、需作为本实现假设记录的选择：各 decoder 子层均采用时间条件 pre-norm；使用正弦帧位置/时间编码；TP-MoE 使用所有原始文本 token（不包含额外 summary token）；交叉注意力权重取各头均值，训练时包含 PyTorch 的注意力 dropout；线性投影和输出保留偏置，LayerNorm 的仿射设置及残差/dropout 顺序以源码为准。时间条件 MLP 与 AdaLN 调制使用 SiLU，这也是本实现选择，未获作者代码核验。

无条件分支将文本 embedding 置零，同时保留原始 token 有效掩码；因此与原始 token 内容无关，但仍可能受长度和已学习偏置影响。参考采样器采用每步重新加噪并重注入已知边界的 inpainting，已知区域跨时间步复用同一个噪声张量，并在末步精确覆盖边界。这是算法选择，不是论文完整 sampler 的复刻。调用者必须显式提供 diffusion beta schedule，代码不把未公开的 schedule 默认值冒充作者设置。

ASFO 函数只计算公式 10–11 的权重，不实现数据重采样、镜像和左右标签交换。`rho_max` 必须是正整数；按打印公式采用 round-half-up，允许高频标签的字面权重为 0，测试覆盖此情况。该返回值不能直接不加判断地作为保留原始样本的重复次数；将其用于 sampler 时，必须明确另选“允许删除高频样本”或“最少保留一次”的策略。论文的打印公式与原样本保留语义之间存在需要确认的差异，本代码不默默替作者解决。

尚缺的完整复现工作：作者一致的 262D 编解码与机器人关节映射、配对 HumanML3D 机器人动作/文本训练划分和统计量、生成器训练与评估器、Stage-II GMT、仿真反馈前缀滚动与 PPO 适配。现有 20/4 个武术参考片段不能代替作者的配对文本生成数据集。

尤其注意骨架顺序：论文附录列出的 12 个 body 索引**不能直接套在当前 Isaac Lab 文件顺序上**；当前参考的第 7、8 号 body 是 hip yaw，而非论文所列 elbow/wrist 等。当前 30-body 资产也没有独立 `palm` 名称。实现只接受已按显式语义映射准备的各块输入，不自动把不同顺序拼为“262D 真值”。

运行数值检查（现有服务器环境，无需下载模型）：

```sh
bash scripts/run_isaac.sh reimplementation/test_core.py
```

仅使用实际下载并核对过的 DistilBERT 权重才能训练/生成有语义的样本。数值检查中的随机 token 不是文本编码器或训练数据的替代。

## 实际运行边界

`results/reimplementation_core/verification.json` 记录 12 项数值检查，包含完整尺寸网络的 CPU 前向传播。`results/reimplementation_text/verification.json` 另行记录了真实预训练文本编码器的接口检查：冻结参数、训练模式下保持 eval、重复编码一致、批内 padding 一致，以及真实 token 接入未训练的完整网络。两者都不属于生成质量实验。

接口检查选择 [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be)，固定 revision `12040accade4e8a0f71eabdb258fecc2e7e948be`。论文未给出精确 checkpoint 标识，因此这个选择属于本实现假设，不能称为作者一致的编码器核验。权重来自官方 Hugging Face 页面，采用 Apache-2.0 许可，文件哈希和发布方 LFS 哈希见 `results/reimplementation_text/download.json`。服务器直连下载失败已保留；本机取得相同固定版本文件后传输并再次核对全部哈希，接口检查退出码为 0。外部权重没有混入 Git 历史。

已有离线文件时，可放到 `cache/text/distilbert-base-uncased/`，按 `download.json` 核对后运行 `bash scripts/run_isaac.sh reimplementation/test_text_encoder.py`。有正常 Hugging Face 网络的环境可先运行 `reimplementation/fetch_text_encoder.py`；该下载入口在原服务器因网络不可达而失败，不能把接口检查通过理解为已验证该服务器的在线下载能力。

用户已确认暂时没有 HumanML3D 配对动作/文本和对应 G1/SMPL 材料。当前交付停留在实现组件和接口验证，不使用 20 个无配对文本的武术片段或人工随机张量冒充生成器训练集。

## 训练入口与输入约定

`train.py` 提供本实现的训练入口：校验输入哈希和分组不交叉，只用训练划分拟合归一化，冻结真实文本编码器，执行 x0/负载均衡目标、梯度裁剪并保存优化器与模型检查点。它没有复用作者未公开的优化器或 schedule 设置。`train_config.example.json` 中的学习率、线性噪声 schedule、训练预算和 ASFO 上限是**未执行的提议设置**，不是作者超参数。实际启动前需用有效数据路径和决定采用的设置替换示例。

manifest 的最小结构如下；每个 `.npy` 必须已按明确的机器人映射和坐标约定制成未归一化的 `[frames,262]`。此入口检查数据结构，不能替代运动学正确性或作者一致性核查。

```json
{
  "format": "phygile262-explicit-v1",
  "purpose": "research_training",
  "mapping_status": "explicit_reconstruction",
  "feature_semantics": "填写骨架语义顺序、坐标系、旋转约定、接触提取和帧率",
  "source_and_license": "填写真实数据来源、处理链及许可",
  "splits": {
    "train": [{
      "features": "motions/clip_001.npy",
      "sha256": "填写实际SHA256",
      "source_group": "original_source_001",
      "texts": ["填写该动作真实对应的文本标注"],
      "annotation_source": "填写标注来源",
      "tags": ["填写实际动作标签"]
    }],
    "validation": [],
    "test": []
  }
}
```

在依赖和数据具备后运行（当前没有执行研究训练）：

```sh
bash scripts/run_isaac.sh reimplementation/train.py \
  --config /path/to/actual_train_config.json \
  --output /path/to/new_run_directory --device cuda:0
```

训练程序拒绝跨划分的来源组/文件重叠，不静默截断长动作并沿用整句标签，拒绝开启尚未实现的 ASFO 镜像。它只进行训练，不提供作者 FID、文本检索指标或闭环评价；也不支持逐比特恢复训练。ASFO 的 `keep_at_least_one` 是显式选择的保留策略，区别于打印公式可能产生的零计数。

`test_training_entrypoint.py` 已用**标记为 unit_test 的人工数组和缩小网络**执行两次 CPU 更新，检查输入拒绝规则、训练归一化、检查点保存和加载；记录在 `results/reimplementation_training_witness/`。这些数组不是动作数据，检查点不是可用于动作生成的模型，结果不能写入论文性能表。测试目的标签必须与 manifest 一致，人工测试材料不能以 `research_training` 标签直接运行。

来源：[论文 v3](https://arxiv.org/abs/2603.19305v3)。本目录新增代码采用 MIT 许可，条款同 `../upstream/LICENCE`。
