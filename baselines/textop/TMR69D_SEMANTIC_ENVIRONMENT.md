# 真正训练冻结后的机器人语义诊断入口

CURRENT 2026-10-29：新的数值后端、900/7200秒监督和activate/launch/collect桥已部署禁用。`configs/tmr69d-semantic-spec.json`的`witnesses_authorized=false`、`metrics_authorized=false`、`frozen_model_binding=null`、`witness_bindings=null`。真实模型见证/语义诊断均0。新13个策略破坏负例和9个数值错误输入、5个解析FID案例在两份独立远端CPU进程真实退出0，仅用合成数字/元数据；不读真实latent、模型、数组，不构造CUDA。8种桥实际渲染通过Python3.8解析。不能把这些准备、合成Rprecision或解析FID当本任务效果。

当前唯一模型仍是`tmr69d_evaluator500_train_v1`。其source/spec/预算冻结不动；轻量观察只用原collector，不与任何CPU模型/GPU/SDK/媒体重叠。所有新模型作业经原`full_library_capacity.lock`串行。已闭合的旧数据/物理/影片/恢复/夹具不重跑，所有原件/失败/partial保留。

## 真正前提

只有真实500轮/507500 AdamW/0skip、epoch500完整状态和精确生产恢复、原process/supervisor/outer0/无worker/锁释放，按`TMR69D_FROZEN_CPU_ENVIRONMENT.md`完成作者CPU私有推理导出和独立新进程CPU复核，terminal collector明确`fresh_cpu_fullstate_and_inference_exact=true`后，才可启用本入口。未训练/随机/原作者人体/CLIP/Huber/38D特征不能替代。

实际frozen验证收据/spec/fullstate/inference SHA、完整闭合和trained_model_sha必须绑定；本入口不延长或恢复VAE/LDM/teacher。独立评价器与原557D特权跟踪器也不能称可部署WBC。

BASE为原`baselines/textop`，Windows显式UTF8、使用现有bundledPython `-B`、cwd为W。每条命令evidence必须唯一短ASCII。不得手改flag或提前pending启用，未知SSH/intent先核原client/worker/实际退出，禁止重复派发。

## 两个真正新进程见证

```text
provider_tools/tmr69d_semantic.py activate --stage witnesses --evidence 唯一短标识
provider_tools/tmr69d_semantic.py launch --role author --evidence 另一标识
provider_tools/tmr69d_semantic.py collect --role author --evidence 又一标识
```

作者固定run `tmr69d_semantic_author_v1`。必须真正`TMR69D_REAL_SEMANTIC_EMBEDDING_WITNESS_FINISHED`、1段/首seed/两mode共2trial、0优化、process/sup/outer0/closed/无worker/共同锁释放，独立collector全SHA/CRC/float32/finite/shape/source join接受后，才执行第二个新进程：

```text
provider_tools/tmr69d_semantic.py launch --role document --evidence 唯一短标识
provider_tools/tmr69d_semantic.py collect --role document --evidence 另一标识
```

固定run `tmr69d_semantic_document_v1`，与作者同首segment/首seed/两mode；两轮不是新增两个不同动作。重新实际加载冻结TMR/HF、原生成NPZ，两进程text/reference/generated三份mean latent数组形状/全SHA须逐位精确匹配。随后保存真实独立检查文档、退出/计数/发现/来源，不能凭空填写空findings。两个见证不是全部1880生成动作的正式语义分数。

## 全声明范围诊断

两见证真实闭合、全部byte与scalar audit通过后：

```text
provider_tools/tmr69d_semantic.py activate --stage metrics --evidence 唯一短标识
provider_tools/tmr69d_semantic.py launch --role diagnostic --evidence 另一标识
provider_tools/tmr69d_semantic.py collect --role diagnostic --evidence 又一标识
```

固定run `tmr69d_semantic32_v1`。启用会精确归档此前witness policy，只改授权/见证证据；runtime signature保持同值。所有已有intent/输出/partial都保留，不能重跑全job覆盖。真实失败先核原退出、完整输出/journal/源SHA，才设计明确续接；没有通过的续接工具时不可盲派。

GPU1共享/.65、CPU0–3总4核、16GiB可用RAM、GPU free>=20GiB、headless。witness900秒/diagnostic7200秒/15秒资源。DATA保留2GiB+128MiB日志+剩余8MiB输出上界，监督随已写字节递减剩余预算，不重复计算已有输出。数字每job重验，非预留。所有文本/NPZ/mean latent只server+D私有，8MiB元数据/数组上界不是许可公开原latent。推理不构造VAE/denoiser，不生成新动作，不更新优化器；只冻结TMR和DistilBERT两个神经模型，1 CUDA上下文。MPNet用于训练伪负例过滤，本诊断不无声添加MPNet语义过滤或CLIP替代。

## 精确数据与数值范围

只用已真实940pair/1880trial生成journal和原scope的32 VAL family/188独立标注段。每个seed/mode全量保留，包括7个物理失败及跟踪误差未通过者，不按物理成败筛掉语义输入。保留原文本/CFG/seed/标签，不做新策略回放或SDK。绑定实际scope与生成闭合，所有NPZ完整SHA/CRC/shape/float32/finite/逐数组hash重新核验，不重跑旧物理审计。

每段只编码`future_frames`对应原归一化参考/生成69D未来特征；生成pad到8的尾项、初始history/seed poses不计入语义。原contact和关节不clamp、不重新归一化。TMR训练norm SHA必须与生成scope接受的norm SHA精确相同，不能重新fit。相同参考在两mode/所有seed间必须逐数组hash一致。文本通过真正冻结DistilBERT token特征、禁用截断，超长显式失败而非零向量替换。所有原公共权重和推理源前后完整SHA/owner/fingerprint复核。

原TMR`encode(..., sample_mean=True)`、eval/dropout关闭，256维raw mean latent，不逐样本L2。编码前后CPU/CUDA RNG状态精确不变，原模型参数hash不变。输出私有3数组容器与source join：188个text、188个参考motion、2mode×5seed×188个生成motion。所有数组有限，终端collector独立核全bytes/SHA/CRC/shape及实际source join，并在0模型加载下重算全部正式标量。数值通过仍不能自动宣称语义质量好或论文完整效果。

## 预先固定的诊断口径

本任务没有已核验的作者完整分批资产，因此透明定义一个保留全部188查询的诊断适配，不宣称作者精确评价协议：

- 候选32条，按`SHA256('textop-tmr69d-candidates-v1:'+segment_id)`排序，顺序切5个32查询块与最后28查询块。最后块固定从前面排序取4个不在当前块的segment，仅作distractor，不重复计查询。全部188条每seed/mode各计一次，每查询都有32候选。相同候选计划用于所有seed/mode，与结果无关，不丢尾项。
- text→motion用float64 raw-latent Euclidean，唯一positive为精确配对标注段。距离完全相同按原上游平均tied rank，报告top1/2/3比例。重复文本/重叠标注保留并报告数量，不假设不同标注必有不同语义，不使用最有利同义过滤。MMdist为188个对角匹配距离的均值。
- FID每seed/mode用188参考与188生成raw motion latent，float64样本协方差ddof1及PSD对称平方根迹计算。记录有效秩、奇异性、浮点负特征值/极小负FID容忍与未clamp原值；不逐样本L2，不池化5次重复参考成940独立参考，不添加任意jitter。188小于256，协方差天然可奇异，有限样本有偏；不是全1774VAL指标。
- 报告全部5seed原值、等权平均和描述性总体std，不选最好seed、不把std当置信区间。参考motion检索baseline另列，用于检查评价器的留出检索能力，不用它调参/早停/延长生成器。

数学实现及解析对照在`tmr69d_semantic_math.py`和新测试；官方MIT mean/cosine/guo丢尾差异仍见`sources/tmr_evaluation_protocol_20261029`。正式数值结果到达后，报告必须同时呈现这些适配/有限样本/重复标注限制、真实负面与所有失败，不能只报正面。后续仍需WBC431D非特权actor和真实在线质量/许可影片接受。
