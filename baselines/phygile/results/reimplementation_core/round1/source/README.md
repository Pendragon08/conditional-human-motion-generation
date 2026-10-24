# 按论文重建的生成模块核心（未训练）

此目录是根据 PhyGile v3 §III-B/C、附录 A 自行实现的研究代码。它与 `../upstream/` 中的作者代码分开保存。**目前没有训练权重、语言生成实验或生成—控制闭环结果；不能作为已复现完整 PhyGile 的证据。**

明确公开的规格：262 维输入；冻结 DistilBERT 的 768 维 token；8 层 AdaLN Transformer decoder，宽度 512、FFN 1024、4 头、dropout 0.1；TP-MoE 每层 12 个两层专家，门控 768→512→512→12，SiLU；空间 mask γ=24、β=0.25；负载均衡损失系数 0.01；x0 回归目标；训练文本置零概率 0.1；CFG 2.5。实现逐项对照公式 7–14。

已实现范围：参数空间混合 TP-MoE、有效 token 的负载均衡、按帧/文本掩码的 decoder、块选择归一化、ASFO 权重、x0 DDPM 目标与采样、标准及负提示 CFG、固定前缀/后缀的采样约束。测试使用人工数值张量，只验证公式、形状、掩码与梯度，不属于生成质量或物理可执行性实验。

未明确公开、需作为本实现假设记录的选择：各 decoder 子层均采用时间条件 pre-norm；使用正弦帧位置/时间编码；TP-MoE 使用所有原始文本 token（不包含额外 summary token）；参考采样器采用每个噪声步重注入已知边界的 inpainting。调用者必须显式提供 diffusion beta schedule，代码不把未公开的 schedule 默认值冒充作者设置。ASFO 按公式 10 四舍五入，允许高频标签乘数为 0；镜像概率对负值截为 0。

尚缺的完整复现工作：作者一致的 262D 编解码与机器人关节映射、配对 HumanML3D 机器人动作/文本训练划分和统计量、生成器训练与评估器、Stage-II GMT、仿真反馈前缀滚动与 PPO 适配。现有 20/4 个武术参考片段不能代替作者的配对文本生成数据集。

尤其注意骨架顺序：论文附录列出的 12 个 body 索引**不能直接套在当前 Isaac Lab 文件顺序上**；当前参考的第 7、8 号 body 是 hip yaw，而非论文所列 elbow/wrist 等。当前 30-body 资产也没有独立 `palm` 名称。实现只接受已按显式语义映射准备的各块输入，不自动把不同顺序拼为“262D 真值”。

运行数值检查（现有服务器环境，无需下载模型）：

```sh
bash scripts/run_isaac.sh reimplementation/test_core.py
```

仅使用实际下载并核对过的 DistilBERT 权重才能训练/生成有语义的样本。数值检查中的随机 token 不是文本编码器或训练数据的替代。

来源：[论文 v3](https://arxiv.org/abs/2603.19305v3)。本目录新增代码采用 MIT 许可，条款同 `../upstream/LICENCE`。
