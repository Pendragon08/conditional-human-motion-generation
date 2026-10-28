# 真实机器人69D文本运动评价器接续

CURRENT 2026-10-29：生成器VAE/LDM用户100000步均已完成，禁止恢复/延长。真实生成、FK、严格物理、9片D影片和保存姿态跟踪精度均已闭合，不重跑。本文件描述后继评价器，不把冻结公共文本模型或随机ACTOR架构称作已训练TMR。

最新覆盖：服务器tmr_text_models_v1已真实270.79秒后Network is unreachable失败1，原process/supervisor/outer1，无worker/锁释放，0完整模型文件，仅第一config的0B partial保留；不能重标成功或重复启动它。替代传输已唯一开始：本机provider_tools/cache_tmr_public_text_models.py，public_text_cache_v1/PID67324（线索），只通过现有127.0.0.1:7897代理下载相同pinnedHF revision/全SHA至D/models/evaluation_text_tmr_20261029。实际receipt/resources/exit在BASE/results/semantic_evaluator_20261029/public_text_cache_v1，tool exec session61659仅线索。11个文件707323471B，1小时有界/15秒D资源/原生owner/无reparse/fsync/rename/2GiB+128MiB保留；每文件最多3次、非0partial只HTTP206精确range续，拒绝从0覆盖。0服务器重派/模型载入/优化更新。结果未知先核原PID/创建身份/exit，不能再启动同目录或删partial。真正全文件D验证/当前原进程退出后，才实现同SHA的有界上传与服务器冻结CPU核验，后继训练/data入口尚未ready。此下载不占GPU模型锁，可独立于私有LDMcopy，不能误称服务器仍有模型作业。

## 当前真实入口

W=C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成。所有Windows命令先PYTHONIOENCODING=utf-8，用现有bundledPython -B；BASE为baselines/textop，ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024。

`tmr_isolated_runtime_v1`已真实process/supervisor/outer0/closed，15.15秒，6个公共wheel共16649167B/SHA/CRC通过，依赖仅装ROOT/evaluator_tmr_20261029/tmr_isolated_runtime_v1/dependencies。原Torch模块源SHA/6个缺失依赖的原环境查询前后精确一致，保护.venv-pbhc-training；没有pip写入原环境。原样MIT ACTOR源固定commit6d74688730d15d43b0a755ce2b0e1f2d76138fc1。真实CPU变量长度finite/padding-zero通过，3个随机架构合计16031301参数/64125204参数字节，Adam上界201326592B；0CUDA/数据/优化更新/预训练权重读取。收据SHA005420576b016387017afd4a010687b243e6c50225d76d3e5186f7fd9bb9821a。此作业已结束，禁止重派。

唯一后继`tmr_text_models_v1`已真正派发；以results/semantic_evaluator_20261029/tmr_text_models_v1/progress.json为准。supervisor401580/start1441142138、child401585/start1441142148仅线索，核真实argv/startticks。requestSHA4d80b12eef59cf3f48996a1cd0923c6a23ea672c17214697486918fd50155d84；executorc5a668940aa4e6380df2021942b2e1e062c837d310375169f536ad3ba4356e5a。CPU0-3/无CUDA/16GiB可用RAM/3600秒/15秒资源/全模型共同锁串行。固定revision的DistilBERT和MPNet公共文件707323471B，DATA实核2GiB+128MiB日志+全部输出+16MiB余量，数字不是预留。逐文件完整SHA或GitblobSHA、fsync/rename；网络失败最多3次，同run只按pinned URL的HTTP206及Content-Range续接，拒绝忽略Range从0覆盖。所有partial/intent/失败保留；不手改活动代码或重复派发。

轻量命令：`provider_tools/collect_tmr_text_models.py --evidence 新唯一短ASCII`。活跃时只查元数据/worker/日志；实际PINNED_TMR_TEXT_MODELS_CPU_VERIFIED+process/supervisor/outer0、after_exit/无worker/锁释放后，collector才核完整模型bytes/SHA和两个冻结本地CPU encoder的真实768D有限输出。仅合成英文句子，无BABEL文本/机器人数组/训练更新，不是已学会机器人语义。网络提前失败时先核原退出/原worker/锁/现有partial与逐attempt日志，再实现明确的签名前缀续接；不能盲派新run或删partial，不能把UNKNOWN当失败。

## 模型、数据与真实验收边界

原样源及LICENSE/provenance在sources/tmr_upstream_20261029；公共文本文件manifest在results/semantic_evaluator_20261029/public_dependency_manifest_v2.json。DistilBERT token特征与MPNet mean-pooled L2 sentence特征分别用于TMR文本编码器输入和InfoNCE伪负例过滤，不能用原CLIP替换。公开G1/38D/30Hz的Motius TMR并非本任务69D/50Hz/BABEL的已接受绑定；未下载其权重，不拿它生成本任务正式指标。

后继实现真正TRAIN-only数据scope：只用现有5361个TRAIN family，保留原2个隔离精确pose重复family，不触碰1774个原val family作为训练/归一化/调参。可以从TRAIN按固定hash另分内部验证family；原val只最终报告。每个真实BABEL annotation/chunk保留，≤200真实69D特征、50Hz/floor-ceil/T-1/end边界显式绑定；clip省略与失败分开，所有重叠标注原样保留，不捏造连续转折。真实norm/源数组/文本/manifest和缓存bytes必须签名。大文本特征可在有界RAM中生成和重建，不能在紧盘上假定已有缓存容量。

正式训练代码/独立见证/数据接受目前尚未完成，training_authorized=false。拟保留上游6层256latent架构、Adam1e-4、InfoNCE温度0.1与重构1/latent1e-5/KL1e-5/contrastive0.1；上游500epoch只是来源，不是已执行。实际批量、数据数、训练时间与完整Adam/allRNG/clock/nextbatch恢复、有限梯度和容量须新作业核验。随机probe不种正式权重。HOME三份201326592B状态+2GiB+128MiB日志的算术预算2885681152B，当前约2.975GB只是测量，未来要新验owner1013/ext4/no-symlink/写入/剩余槽；不能删原状态或降低2GiB。全部模型CPU/GPU、SDK和媒体经full_library_capacity.lock串行，GPU1共享/.65/CPU0-3、允许共享无需空卡。

只有真实TRAIN学得、独立恢复核验并固定的机器人69D文本/运动encoder，才能计算heldout Rprecision、MMdist、FID。原TMR代码和论文batch32/Euclidean/rawfeature分布口径需明示，不能用合成forward、Huber、CLIP或随机特征冒称正式语义结果。保持全部seed/mode和原负面结果；不因评价器质量差给生成器超过100k加训。431D可部署actor的真实WBC/在线链路仍要单独验收，当前557D特权teacher不是已部署WBC。

每个新job实核盘/RAM/GPU/worker/commonlock。原数据、完整Adam、失败/partial、唯一记录/SDK冷备份保留；0清理、0共享环境修改、0生产训练信号。公开只代码/聚合证明与MIT参考，私有BABEL文本/数组/轨迹/完整状态只server+D。SSH未知先核原intent/outcome/真实worker，不能把连接断开当作业退出。
