# 接触采样 V5 文档见证

记录时间：2026-10-11 13:25:48 UTC。文档原样 `--check` 执行一次，exit 0；出现 `CONTACT_SAMPLING_BATCH_PASS repeat_a` 和 `CONTACT_SAMPLING_DOCUMENT_CHECK_PASS`。未发现本次文档调用与首批预期证据的偏差。

首批 receipt 为 complete，12/12 完成，4 个来源、12 个不同 case、120 帧、seed 11、一次生成批次。七张量重置至首次动作差值全为 0，七个状态 SHA 完全一致；四项初始 rigid cache 差值全为 0。

请求与执行时 lineage 逐项一致；配置、manifest、冻结阈值、versions 及三个输出的实测 SHA 匹配绑定。初始状态归档 SHA 也匹配元数据。服务器只读核对 26 个版本文件、源数据/语义文件与冻结权重哈希，全部一致；本地与服务器文档 SHA 一致。

metric_numerics 为 PASS_ROUNDOFF_ONLY，浮点变动数及最大绝对差均为 0，重算与保存对象 SHA 相同，分类、阈值和已保存分数保持一致。

joint_pass 为 0/12。此见证仅覆盖首批调用和采样接口；repeat_a/repeat_b 接触力可重复性、完整仿真重置修复、真实接触与主要效能尚不能由此认证。未执行 --run、重试、安装、代码修改、其他 Python 运行时、许可数组下载、Overleaf 操作或子代理。

详细证据与完整哈希见同目录 trace.json。

