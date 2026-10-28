# 机器人TMR训练结束后的私有CPU导出与新进程复核

CURRENT 2026-10-29：以下独立后继已部署，但两份spec均`authorized=false`、`training_binding=null`、`export_binding=null`。实际导出、独立CPU复核与正式语义指标均为0。23个新隔离元数据破坏负例真实CPU退出0，仅证明拒绝错误计数、未闭合训练、错误来源、紧急状态及未完成内部验证；不读取真实checkpoint/模型/数组。旧夹具不重跑。

唯一活动模型仍为`tmr69d_evaluator500_train_v1`。活动训练source/spec/预算完全不改；禁止同时执行本文件的CPU模型作业、另一GPU/SDK/媒体作业。纯元数据收集不构造模型。所有新模型作业必须通过原共同`full_library_capacity.lock`；每次新验真实盘、16GiB可用RAM、worker与锁。

BASE为`work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop`。Windows使用现有bundledPython `-B`，先设置`PYTHONIOENCODING=utf-8`，cwd为原W。所有evidence必须每次唯一短ASCII标识。以下命令只在其真实上游已经闭合时执行，准备期不做pending activation。

## 必需的真实训练闭合

先用原轻量工具`provider_tools/collect_tmr69d_evaluator_training.py --evidence 新唯一标识`。只有真实`TMR69D_FORMAL_EVALUATOR_500_EPOCHS_FINISHED`、500轮/507500尝试/507500 AdamW/0skip、epoch100与epoch500两份稳定完整状态及生产模型/AdamW/全部RNG/sampler/nextbatch精确重载，且process/supervisor/outer0、after_exit无worker、共同锁释放，才继续。

新`tmr69d_trained_acceptance.py`独立核原训练spec/所有数值源SHA、输入接受收据、上游模型源、原receipt/process/resource全SHA及样本数、真实退出和checkpoint完整字节。固定只选`/home/ouyanghao/textop-tmr69d-evaluator-20261029/checkpoints/tmr_epoch_500.pth`。epoch100、emergency、尚有pending_validation或post-target状态均拒绝，不能把旧LDM100k接受器套到507500更新。

## 作者CPU导出

顺序运行，每个命令完成并检查真实结果后才下一条：

```text
provider_tools/tmr69d_frozen_cpu.py activate --stage export --evidence 唯一标识
provider_tools/tmr69d_frozen_cpu.py launch --stage export --evidence 另一唯一标识
provider_tools/tmr69d_frozen_cpu.py collect --stage export --evidence 又一唯一标识
```

固定run `tmr69d_frozen_export_v1`。activation归档原禁用spec精确字节，然后绑定真实训练闭合；不是模型作业。launch只派发一次，已有driver/intent/输出/未知SSH先核原client与worker，不能重派。轻量collector在活跃时只读receipt/logtail/worker；终端才核SHA与闭合，不反序列化模型。

CPU监督900秒/CPU0–3总4核/无可见CUDA/16GiB RAM/15秒资源。DATA保留2GiB+128MiB日志，HOME需要2GiB+128MiB日志+80MiB推理上界；每次新验，不当容量预留。输出仅私有HOME `textop-tmr69d-evaluator-20261029/inference_v1/tmr69d_epoch500_inference.private.pth`；不覆盖/删除旧状态或partial，不公开权重。完整state不变。

实际CPU代码将核全model/AdamW有限张量与全部507500步骤、sampler499/32462、真实请求/输入/runtime签名、所有RNG结构，恢复CPU AdamW但执行0更新。保存的CUDA RNG字节只检查，不在CPU恢复。采用原训练同架构/原class方法，精确加载完整model，核原final_model_sha；推理payload不含AdamW。真实fsync/rename后重新载入并严格比对，非零motion/text可变长度合成mean forward逐位一致，padding明确为0，0 CUDA。合成forward不是heldout语义分数。

必须真正`TMR69D_TRAINED_FROZEN_CPU_EXPORTED`、所有CPU/监督/outer0、after_exit无worker/锁释放，terminal collector给出真实`artifact_audit.export_binding`，才可启用fresh。

## 独立新进程CPU复核

```text
provider_tools/tmr69d_frozen_cpu.py activate --stage verify --evidence 唯一标识
provider_tools/tmr69d_frozen_cpu.py launch --stage verify --evidence 另一唯一标识
provider_tools/tmr69d_frozen_cpu.py collect --stage verify --evidence 又一唯一标识
```

固定run `tmr69d_frozen_verify_v1`。该进程从真实原完整状态和推理payload独立构造模型、恢复CPU AdamW执行0更新，核所有参数、原final_model_sha、保存clock/sampler/RNG与来源、非零mean motion/text forward及两文件全SHA/指纹；不复用作者内存或诊断权重。监督资源与原共同锁相同。真正`TMR69D_TRAINED_FROZEN_FRESH_CPU_VERIFIED`、process/sup/outer0及collector闭合审计后，才接受冻结69D评价器作为后续指标输入。届时必须保存实际复核文档/计数/发现，不能凭空写空findings。

后继真实heldout特征生成/语义指标入口仍需实现并实际接受，不能拿本次禁用源码、23负例、CPU完整性或合成forward冒充语义ready。只有真实冻结评价器才计算全部1880生成trial/全部seed/mode及失败；原尾帧不计分、原contact不clamp、原标签/CFG不变。WBC/在线质量仍另需真实非特权控制接受，不追加VAE/LDM或teacher训练。

## 评价协议的已核来源和差异

原样MIT参考保存于`sources/tmr_evaluation_protocol_20261029`，commit固定`6d74688730d15d43b0a755ce2b0e1f2d76138fc1`，Gitblob/完整SHA另存provenance。未执行这些上游入口、未替换生产源。

[官方retrieval.py](https://github.com/Mathux/TMR/blob/6d74688730d15d43b0a755ce2b0e1f2d76138fc1/retrieval.py)明确text/motion编码时`sample_mean=True`，随后`get_sim_matrix`对每条latent做L2归一化再算cosine。其guo协议以固定numpy seed0打乱排序后的keyids、只取完整32批，尾项被省略；normal/threshold/nsim另有范围。它没有在此文件提供FID或MMdist。本任务TextOp论文声明的batch32 Euclidean/MMdist/raw-feature FID不能无声替换成该cosine结果，也不能把“未在此入口找到”写成外部不存在。

[官方encode_motion.py](https://github.com/Mathux/TMR/blob/6d74688730d15d43b0a755ce2b0e1f2d76138fc1/encode_motion.py)与[encode_text.py](https://github.com/Mathux/TMR/blob/6d74688730d15d43b0a755ce2b0e1f2d76138fc1/encode_text.py)同样使用mean编码，输出原latent。因此本次推理payload固定mean/256维raw latent，不先逐条L2归一化。后续Euclidean候选分组、重复标注/tie、最后不满32批、样本/segment/seed权重与FID有限样本估计必须在真实计算前明确固定并透明报告；不能选更好分数的口径或丢弃失败/尾项。当前尚无本任务Rprecision/MMdist/FID结果，32family诊断不称全1774VAL或论文连续转折。
