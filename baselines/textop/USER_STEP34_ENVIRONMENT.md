<!--tmr100-user-current-start-->
CURRENT 2026-10-29T08:22:19.012365+08:00：最新人类命令覆盖旧500轮：“500轮训练要太久了，改为100轮吧，不要500轮了。”现只选TMR第100轮/101500 AdamW完整状态，后续直接冻结评估与Step3/4，不再等500或507500。原run名tmr69d_evaluator500_train_v1和运行spec/source仍为原来源，冻结不编辑、不重启、不压缩数值/常数LR/RNG。VAE/LDM仍各100000，旧教师停止。
新唯一CPU端点user_tmr100_control_v1已实际派发并ARMED：采样Unix1793233316.210879为72轮/73300AdamW，WAITING_FOR_REAL_TMR_EPOCH100_SAVED_AND_RELOADED/0生产信号。原supervisor416285/start1441791429与worker416293/start1441791443保持原作业；新observer431950/start1443022390、supervisor431945/start1443022380仅身份线索，核真实argv/starttime。active control specSHA2f0498031e7fa9d5630848320dfed944c66c2c36ef45fa6cacbfe1989a02bba8，watcher4856467bd730b1f826a44e96b84faecbde57185f5e1ece20038b1d66d7379a24。控制每秒等真正100轮atomic fsync/rename、model/AdamW/allRNG/sampler/nextbatch生产精确重载及pending=false，再独立全SHA/owner/no-symlink/fingerprint后一次pidfdSIGINT，0模型/真实checkpoint载荷反序列化/GPU；独立metadata锁，不占GPU锁。严禁rearm/第二observer/额外信号。CPU/1GiB/2GiBdata/15秒资源/7天边界不降。
作者v2和fresh user_tmr100_control_document_v2 CPUfixture均实际0/reaped/final，14+4负例/重复intent拒绝，只有各自sleep进程收到1次SIGINT，生产信号/真实checkpoint/model/GPU均0。fresh真实SUMMARY SHA1a380fbc6971654a416f8c0937a498a6f947e31964b996a2e78fad79ec91d38a。prepare本地换行签名拒绝与author_v1测试argv子串错误失败1均保留；无控制/训练信号后已核空，归档精确旧字节才修复新禁用夹具，不改原训练，旧失败不改成功，不重跑已完成夹具。
只有USER_TMR100_EPOCH_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED+控制自身process/sup/outer0+原无worker/模型锁释放才接受端点。原RuntimeError('Owned TMR evaluator interrupted with signal 2')及原训练非零退出保留，post100内存/emergency不选，不重标500轮成功；原100轮immutable state唯一选用。
新100轮CPU导出/独立复核/语义桥已另名部署禁用，详见TMR100_FROZEN_AND_SEMANTIC_ENVIRONMENT.md。12个新prefix计数/RNG/sampler负例远端CPU0、14实际桥渲染Python3.8通过，仅准备；实际100轮出口/导出/语义均尚未接受，不提前pending启用。旧500轮源码/spec全保留但不再用于本目标。请用provider_tools/tmr100_endpoint.py collect --evidence新短ASCII轻量查唯一控制；正式模型live仍原collect_tmr69d_evaluator_training.py。阶段不是停工点，完成后直接新100CPU/fresh/语义→Step3增强与新one-stagePPO。Step3/4新增持久空间仍待用户输入，不能用本次减少epoch假装解决空间。
<!--tmr100-user-current-end-->

# USER STEP3/4：实际范围与新运行环境

先读reports/user_step34_augmentation_and_ppo_20261029.md、CONTINUATION顶部CURRENT和results/user_step34_20261029实际收据。人类授权5368条/31.48小时增强与新431D Actor one-stagePPO，原VAE/LDM各100000不加训，旧557D特权教师不恢复。健康TMR原作业不改源/spec/预算/信号。

实际scope user_augmentation5368_scope_v1已经闭合0、私有server+D通过，不重派。20新负例和backend标准库import已经结束，不重跑。provider_tools/user_step34_scope.py collect --evidence新短ASCII只读可查实际scope/容量/当前TMR身份；不加载数组或模型。不可使用launch再次派发同run。private scope ledgerSHA ac49c12afeedc8703cf2c3b8c8e19df4c25fc691c2220df9d927d32d3552cbb4。

当前user-augmentation5368-generation-spec.json两授权false，user-step4-one-stage-ppo-spec.json witnesses/training均false，无storage/input/witness实际binding。不得手改flag、提前pending执行、把新源码称ready。缺少完整生产生成entry/supervision/resume/activate/collector与混合数值库验收和PPO入口/恢复/独立文档验收；继续实现，不能停在此文档。新的存储输入仍必要，不得凭已接受旧HOME/data/TMR预算挪用或清理。

暖环境只读复用ROOT/scripts/run_tracker.sh，已有IsaacLab2.1/IsaacSim4.5.0/Torch2.5.1+cu118/RSL-RL2.3.1，configs/tracker-env-spec.json绑定。实际路径/URDFhash已核，本轮0新SDK kernel，不能称新431D PPO环境已READY。后续真实SDK数值见证必须按run-experiment环境契约先作者再fresh按完整文档执行。无安装/共享环境修改。

GPU1shared/.65/CPU0-3总4核/headless；全部CPU模型/GPU/SDK/媒体经ROOT/receipts/full_library_capacity.lock串行。metadata独立user_step34_metadata.lock已经释放不代表模型锁空。纯SHA字节复制可并行。不触碰他人/共享依赖/暂停旧项目/.venv-pbhc-training。不删数据、所有checkpoint、失败、partial、唯一记录或SDK冷备份。每SSH/SCP/Git唯一intent/outcome/directstreams，未知先核原client/worker/退出，连接断开不是服务器退出。短ASCII证据名、UTF8、现有bundledPython -B。

正式数值后才以实际journal/数组/时长计数，planned5368/31.48不得计actual；所有失败和未尝试分列。原VAL只作效果验收，不进入此增强/新PPO训练。全部BABEL原文本/轨迹/fullAdam只server+D，不公开权重或逐源清单。
