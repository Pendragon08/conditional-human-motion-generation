import pathlib,json,hashlib,shutil
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/filtered15000_handoff_tools_20261027'
proof=json.loads((O/'ldm_preparation_cpu_v1.json').read_text(encoding='utf-8'));dep=json.loads((O/'ldm_runtime_deploy_v1.json').read_text(encoding='utf-8'))
assert not proof['actual_gradients_or_sampling_executed'] and dep['new_gpu_contexts']==0
p=B/'configs/teacher15000-full-to-generator-autostart-20261027.json';plan=json.loads(p.read_text(encoding='utf-8'))
plan.update(corpus_activation_bridge='provider_tools/activate_filtered15000_corpus.py',corpus_activation_executed=False,
 full_ldm_numerical_runtime_prepared=True,full_ldm_actual_entry_completed=False,full_ldm_actual_gpu_witnesses_completed=False,
 ldm_plan='configs/filtered15000-ldm-plan.json',ldm_numerical_staged_sha256=dep['written'],
 next_implementation='Post-corpus acceptance/private recovery backup/real persistent capacity/VAE-witness activation bridge; then complete full LDM entry and real VAE200k inference export gates')
p.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
text='''# 筛后语料激活桥与完整LDM数值入口准备

全库物理回放继续运行，尚未结束。本轮没有新物理试验、模型加载、GPU上下文、语料构建或生成器训练，也没有修改现行回放四个源文件或配置。

新增activate_filtered15000_corpus.py实现正面CPU审计后的接受/激活桥：先验实际CPU进程、监督、外层退出与私有通过来源，再验构建器/依赖SHA、共同锁、GPU1、RAM/SHM/磁盘及新视图；保存原禁用配置后启用构建，不自行派发。由于完整实际结果尚未到达，这个桥没有执行。

完整LDM数值Runtime已准备并部署为不可启动状态。采用发布源码的四段完整时钟[50000,50000,50000,150000]，共300000 primitive步；stage>0的真实代码路径使用完整五步DDPM生成后续history候选。实际生成的候选数量与选用rollout history分别记载，恢复时从完整状态续接辅助计数。训练候选采样沿用native条件DDPM，不套用推理CFG；生成诊断仍CFG5。

差异显式记录：论文LDM最大rollout概率0.8，发布原manager上限1；use_full_sample开启而发布默认关闭；static_pose关闭时修正原stage>=2仍可能进入的边界行为；论文选择FIXED_LARGE，发布默认FIXED_SMALL保留在说明中。checkpoint计划100k/200k/300k加紧急槽，不压缩完整训练时钟，所有实际保存版本保留。

本轮仅新增10个纯CPU时钟/概率边界检查和源码语法/分支检查，没有真实梯度、扩散采样或阶段>0GPU见证。未来真实模型探针会临时测试later-stage gate和batch12的五次denoiser调用，恢复RNG/模型/时钟，并明确不宣称已经训练到50k。完整entry/supervisor、真实新筛选VAE200k和独立推理权重CPU导出、持久容量及作者/fresh GPU验收仍待完成。旧quick3000和prefilter8100均不复用为正式训练。

自动开训仍已授权，无需再问用户。下一步继续完成语料接受/私有小型恢复材料备份/真实目录容量及VAE见证启用桥。上述代码准备不计为实际筛选通过、生成器开训或语义效果提升。
'''
report=B/'reports/filtered15000_ldm_preparation_20261027.md';report.write_text(text,encoding='utf-8')
notice='''New staged follow-up: activate_filtered15000_corpus.py binds only positive
closed CPU audit and actual private accepted sources; it has not run. Full LDM
numerical source is prepared at filtered15000_ldm_runtime.py/full_ldm_history.py,
disabled plan filtered15000-ldm-plan.json. Native4-stage300k/fullDDPM history code,
paper rollout cap.8 vs native1 and disabled-static edge correction are explicit.
Only10 new CPU probability/schedule checks; no real GPU model/sample/gradient.
Full LDM entry/actual VAE200k export/author-fresh witness still pending. Do not
claim a later-stage optimizer clock from the future simulated-gate probe.
Continue implementing post-corpus data acceptance/private recovery backup and
persistent capacity/VAE activation bridge while real full replay is pending.
Read reports/filtered15000_ldm_preparation_20261027.md. No old diagnostics repeated.
'''
for name in ['CONTINUATION.md','README.md']:
 q=B/name;old=q.read_text(encoding='utf-8');mark='<!--filtered15000-tools-current-end-->';assert mark in old and notice not in old
 q.write_text(old.replace(mark,notice+mark,1),encoding='utf-8')
for src in [p,report,B/'configs/filtered15000-ldm-plan.json',O/'ldm_preparation_cpu_v1.json',O/'ldm_runtime_deploy_v1.json']:
 dst=D/src.relative_to(B);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
print(json.dumps(dict(status='DISABLED_LDM_CORE_AND_CORPUS_BRIDGE_DOCUMENTED',actual_new_gpu_runs=0,actual_full_result_complete=False)))
