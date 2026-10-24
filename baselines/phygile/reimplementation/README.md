# 按论文重建的生成模块核心（未训练）

此目录是根据 PhyGile v3 §III-B/C、附录 A 自行实现的研究代码。它与 `../upstream/` 中的作者代码分开保存。**已有随机权重扩散候选的真实仿真工程验证，尚无真实配对数据训练后的生成权重或语言生成质量实验；不能作为已复现完整 PhyGile 的证据。** 最新证据见 [工程报告](../reports/PhyGile课程评估与262D闭环补全报告.md)。

明确公开的规格：262 维输入；冻结 DistilBERT 的 768 维 token；8 层 AdaLN Transformer decoder，宽度 512、FFN 1024、4 头、dropout 0.1；TP-MoE 每层 12 个两层专家，门控 768→512→512→12，SiLU；空间 mask γ=24、β=0.25；负载均衡损失系数 0.01；x0 回归目标；训练文本置零概率 0.1；CFG 2.5。实现包含公式 7–11、13–14 的数学组件；公式 12 启发明确标注假设的边界采样器，现已接入真实仿真接口，但尚无已训练语义生成或闭环 PPO。

已实现范围：参数空间混合 TP-MoE、有效 token 的负载均衡、按帧/文本掩码的 decoder、块选择归一化、ASFO 权重、x0 DDPM 目标与采样、标准及负提示 CFG、固定前缀/后缀的采样约束。测试使用人工数值张量，只验证公式、形状、掩码与梯度，不属于生成质量或物理可执行性实验。

未明确公开、需作为本实现假设记录的选择：各 decoder 子层均采用时间条件 pre-norm；使用正弦帧位置/时间编码；TP-MoE 使用所有原始文本 token（不包含额外 summary token）；交叉注意力权重取各头均值，训练时包含 PyTorch 的注意力 dropout；线性投影和输出保留偏置，LayerNorm 的仿射设置及残差/dropout 顺序以源码为准。时间条件 MLP 与 AdaLN 调制使用 SiLU，这也是本实现选择，未获作者代码核验。

无条件分支将文本 embedding 置零，同时保留原始 token 有效掩码；因此与原始 token 内容无关，但仍可能受长度和已学习偏置影响。参考采样器采用每步重新加噪并重注入已知边界的 inpainting，已知区域跨时间步复用同一个噪声张量，并在末步精确覆盖边界。这是算法选择，不是论文完整 sampler 的复刻。调用者必须显式提供 diffusion beta schedule，代码不把未公开的 schedule 默认值冒充作者设置。

`features.py` 的 ASFO 函数计算公式 10–11 的权重；`train.py` 已按显式配置执行样本重复。镜像和左右标签交换尚未实现。`rho_max` 必须是正整数；按打印公式采用 round-half-up，允许高频标签的字面权重为 0，测试覆盖此情况。训练入口要求明确选择“允许删除高频样本”或“最少保留一次”的策略。论文的打印公式与原样本保留语义之间存在需要确认的差异，本代码不默默替作者解决。

已有 `codec262.py`、`curriculum_evaluator.py`、`diffusion_proposer.py` 和真实 `replay_prefix_adapter.py`。已有40条数值时基一致的SSM动作/文本配对，但尚缺充分规模且通过物理过滤的配对数据、作者一致的编解码约定确认、生成器训练与评估器、完整两阶段 GMT 与闭环 PPO 适配。现有武术参考片段和SSM小样本不能代替完整配对生成数据集。

2026-10-23新增 `motion_clock.py`：真实训练检查点保存逐条codec时钟/坐标元数据哈希；已训练后端严格读取训练FPS，在固定物理时刻将生成姿态转换为控制帧，使用同一规范坐标系编码真实前缀与目标后缀，并采用论文CFG2.5。缺失时钟契约的旧检查点必须经审计迁移，不能默认50Hz。当前支持单帧终端，最终时刻必须落在特征网格；终端根位置/姿态仍需显式输出投影。冷启动保持姿态假设、生成但未执行的末尾时段、边界与安全投影均留有记录。6项数值反例测试和40条真实配对的时钟检查已通过，未新增真实生成器训练或物理性能结论。详细说明与冻结验证入口见 [时钟验证](../../paper_faithful_reproduction/GENERATOR_CLOCK_INVOCATION.md)。

尤其注意骨架顺序：论文附录列出的 12 个 body 索引**不能直接套在当前 Isaac Lab 顺序上**；当前参考第 7、8 号 body 是 hip yaw。新编解码按名称及 URDF 固定掌心变换映射，采用显式版本化的坐标、父子旋转和差分约定；不能把它称为作者数值约定的逐项验证。训练 manifest 应包含 `codec_schema` 和 `codec_skeleton_sha256`，检查点后端核对它们与归一化和文本编码器版本。

运行数值检查（现有服务器环境，无需下载模型）：

```sh
bash scripts/run_isaac.sh reimplementation/test_core.py
```

仅使用实际下载并核对过的 DistilBERT 权重才能训练/生成有语义的样本。数值检查中的随机 token 不是文本编码器或训练数据的替代。

## 实际运行边界

`results/reimplementation_core/verification.json` 记录 12 项数值检查，包含完整尺寸网络的 CPU 前向传播。`results/reimplementation_text/verification.json` 另行记录了真实预训练文本编码器的接口检查：冻结参数、训练模式下保持 eval、重复编码一致、批内 padding 一致，以及真实 token 接入未训练的完整网络。两者都不属于生成质量实验。

接口检查选择 [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be)，固定 revision `12040accade4e8a0f71eabdb258fecc2e7e948be`。论文未给出精确 checkpoint 标识，因此这个选择属于本实现假设，不能称为作者一致的编码器核验。权重来自官方 Hugging Face 页面，采用 Apache-2.0 许可，文件哈希和发布方 LFS 哈希见 `results/reimplementation_text/download.json`。服务器直连下载失败已保留；本机取得相同固定版本文件后传输并再次核对全部哈希，接口检查退出码为 0。外部权重没有混入 Git 历史。

已有离线文件时，可放到 `cache/text/distilbert-base-uncased/`，按 `download.json` 核对后运行 `bash scripts/run_isaac.sh reimplementation/test_text_encoder.py`。有正常 Hugging Face 网络的环境可先运行 `reimplementation/fetch_text_encoder.py`；该下载入口在原服务器因网络不可达而失败，不能把接口检查通过理解为已验证该服务器的在线下载能力。

2026-10-18 已取得 HumanML3D 官方公开文本、来源索引和划分，见 [配对数据准备状态](../docs/paired_data_status_20261018.md)。对应原始动作、人体模型和 G1 配对动作尚未准备好。当前没有在这些文本上训练生成器，不使用 20 个无配对文本的武术片段或人工随机张量冒充配对训练集。

## 训练入口与输入约定

`train.py` 提供本实现的训练入口：校验输入哈希和分组不交叉，只用训练划分拟合归一化，冻结真实文本编码器，执行 x0/负载均衡目标、梯度裁剪并保存优化器与模型检查点。它没有复用作者未公开的优化器或 schedule 设置。`train_config.example.json` 中的学习率、线性噪声 schedule、训练预算和 ASFO 上限是**未执行的提议设置**，不是作者超参数。实际启动前需用有效数据路径和决定采用的设置替换示例。

manifest 的最小结构如下；每个 `.npy` 必须已按明确的机器人映射和坐标约定制成未归一化的 `[frames,262]`。此入口检查数据结构，不能替代运动学正确性或作者一致性核查。

```json
{
  "format": "phygile262-explicit-v1",
  "purpose": "research_training",
  "mapping_status": "explicit_reconstruction",
  "codec_schema": "phygile-g1-262-semantic-urdf-discrete-v1",
  "codec_skeleton_sha256": "d2627ab9be14d9dd46fc483b754128e6c8f2515db5b1a274ba1b2b97d9aac6e7",
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

## 物理前缀滚动接口

`physics_prefix.py` 实现有界的提出候选→仿真→拒绝或接续状态机。它保存接受的实际状态，拒绝片段回滚，耗尽重试预算时返回未被污染的前缀；末端后缀另作误差检查。真实 `replay_prefix_adapter.py` 已实现 snapshot/restore/rollout 契约：独立进程重建同种子场景，重放全部已接受历史，并核对实际状态。直接写回仿真状态的回滚因重复执行不一致而禁用；当前工程使用显式确定性设置，保留原始失败。

本轮实际运行随机权重扩散采样→262D 解码→71D G1 跟踪→重置前反馈→拒绝/重试。两个 50 帧候选均在 1/50 帧失败，0 接受、0 新帧提交；另有两个各 1 帧的保持姿态代理窗口验证真实接受、反馈和历史重放。生成器与 GMT 参数冻结，完整语义生成训练及闭环 PPO 适配未运行。范围与证据见 [最新工程报告](../reports/PhyGile课程评估与262D闭环补全报告.md)；上一轮单参考离线反馈测试保留在 [历史报告](../reports/PhyGile课程修复与物理前缀推进报告.md)。
