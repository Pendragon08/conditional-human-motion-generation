import pathlib,json,hashlib,shutil
W=pathlib.Path(__file__).resolve().parents[2];T=W/'work/textop_reproduction_20261024';B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
O=B/'results/filtered15000_handoff_tools_20261027';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
deployment=json.loads((O/'deploy_v1.json').read_text(encoding='utf-8'));assert deployment['status']=='NEW_HANDOFF_SOURCE_STAGED_ALL_LAUNCH_FLAGS_FALSE'
planpath=B/'configs/teacher15000-full-to-generator-autostart-20261027.json';plan=json.loads(planpath.read_text(encoding='utf-8'))
plan.update(current_stage='FULL_STRICT_RUNNING_NEW_HANDOFF_SOURCES_STAGED_DISABLED',strict_audit_run='full_result_audit_v1',
 handoff_environment_document='FILTERED15000_HANDOFF_ENVIRONMENT.md',
 preparation_findings=['New closed-result CPU audit, actual-pass corpus and fresh VAE entries are implemented and staged with all launch flags false',
 'Syntax/CPU byte-selection primitives passed; actual full-result audit, filtered corpus, GPU gradient and fresh-agent witnesses have not run',
 'New light progress collector avoids unbounded full trace reads; actual closed trace validation belongs to the new durable CPU supervisor',
 'ROOT/data currently cannot satisfy five VAE checkpoint slots plus reserve; owned persistent /home candidate measured but not reserved or accepted',
 'Full LDM300k entry and actual acceptance remain pending'],
 staged_files_sha256=deployment['written'],actual_handoff_model_loads=0,actual_handoff_gpu_runs=0,actual_handoff_cpu_audits=0)
planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
notice='''<!--filtered15000-tools-current-start-->
Post-filter handoff sources are now staged DISABLED. Read
FILTERED15000_HANDOFF_ENVIRONMENT.md and results/filtered15000_handoff_tools_20261027/deploy_v1.json.
Use collect_teacher15000_strict_light.py for routine active/full-closed metadata;
do not repeat the older collector's heavy all-trace loop as a production audit.
New light collector actually exited0 with0 private trace reads;22:36 observation
5009finished/4385complete/624failed/3269pending, no new exits, sameSDK/lock.
After real full closure and lock release, enable_filtered15000_audit.py then
launch_filtered15000_handoff.py --stage audit --run full_result_audit_v1.
The source checks actual closed8278 inputs before enabling; do NOT run while
pending or repeat activation/dispatch on unknown outcomes. New CPU audit has
16GiB RAM/3GiB disk guard,900s supervised child and private pass/decision files.
Syntax and new byte/selection CPU checks passed;0 actual result audits,
0 filtered corpus jobs/gradient witnesses/new generator jobs/SDK/model loads.
All three new specs stay false/null until real input and positive witnesses.
New filtered corpus selects actual complete passes, source/mirror family split,
TRAIN69D refit/fullrealCLIP/loader checks; old prefilter corpus stays distinct.
New VAE entry reuses only unchanged numerical runtime and binds actual audit,
corpus, data, source, seed/model and checkpoint signature for both fresh witnesses.
Formal200k starts new state, not the four diagnostic updates or prefilter8100.
Metadata-only storage sample: data4772315136B/home5581582336B, ownedext4.
VAE five bounded slots+reserve+logs require5452595200B; home is merely a
candidate, not capacity acceptance. Choose fresh owned task destination and
bind allowed root after actual resource checks. Preserve all checkpoints/data.
No actual filtered acceptance or training yet. Full LDM entry remains pending.
<!--filtered15000-tools-current-end-->

'''
for name in ['CONTINUATION.md','README.md']:
 p=B/name;old=p.read_text(encoding='utf-8');first,rest=old.split('\n\n',1);assert '<!--filtered15000-tools-current-start-->' not in old
 p.write_text(first+'\n\n'+notice+rest,encoding='utf-8')
M=B/'sources/filtered15000_handoff_maintenance_20261027';M.mkdir(exist_ok=False)
for name in ['prepare_filtered15000_corpus_entry.py','prepare_filtered15000_vae_entry.py','check_filtered15000_handoff_tools.py','prepare_light_strict_collector.py','deploy_filtered15000_handoff_tools.py','finalize_filtered15000_tools.py']:
 shutil.copy2(T/name,M/name)
(M/'.gitattributes').write_text('*.py -text\n',encoding='utf-8');(M/'README.md').write_text('Source archive from work/textop_reproduction_20261024 for audit/recovery. Do not rerun these maintenance snapshots: existing stages and evidence must be inspected. Runtime instructions are in FILTERED15000_HANDOFF_ENVIRONMENT.md. No actual filtered acceptance or generator training is represented by preparation.\n',encoding='utf-8')
report='''# 15000轮完整筛查后的自动接续入口

新CPU闭合结果审计、真实通过来源语料构建和新VAE入口已实现并部署为禁用状态。未读取或审计尚未完成的全库结果，未创建新语料，未进行新梯度试验或训练，未添加SDK/GPU上下文，也未修改正在运行的完整物理回放。

新轻量读取器已真实退出0，只读取收据、退出记录、进程身份及少量日志，不打开私有轨迹数组。完整结果的逐源来源/哈希/有限姿态/试验覆盖检查交给新CPU监督器：真实输入闭合后才启用，16GiB可用RAM、3GiB磁盘保留、900秒上限、每15秒资源日志及真实子进程退出。它保留全部8278个决定，并为构建器生成私有真实通过来源清单。

构建器以真实通过来源选择原REFERENCE+BABEL，沿用源/镜像同family split，重新检查精确姿态泄漏并隔离TRAIN家族，重新拟合TRAIN-only69D统计，使用真实完整CLIP和实际加载检查。新VAE使用新的来源/数据/检查点绑定和两次实际梯度/模型/Adam/RNG恢复见证，正式200k从新状态开始。旧预筛选8100步和语料保留。

本轮只有新源码语法、三个合成姿态字节以及非法长度/NaN/四元数/未知来源的CPU检查与独立来源顺序检查，不是实际完整结果或生成器验收。原先已完成的诊断没有重跑。所有新launch标志仍false，实际输入SHA字段仍null；真实全库结果到达后自动执行接受与启用，无需用户重复批准。独立实际文档见证、完整LDM300k实现/验收仍待完成。

只读容量采样data可用4772315136字节，home可用5581582336字节，均本人所有ext4。VAE四个50k完整检查点加一份紧急保存槽、2GiB保留和512MiB日志预算需5452595200字节。home只是候选，尚未创建/接受新训练目录；真实开训前必须重新验资源并绑定自有持久目录。未删除任何文件。

实际命令、来源约束和明确的未完成项见FILTERED15000_HANDOFF_ENVIRONMENT.md。健康完整回放继续，不把这些准备工作通知成新物理或语义结果。
'''
(B/'reports/filtered15000_handoff_tools_20261027.md').write_text(report,encoding='utf-8')
for src in [planpath,B/'FILTERED15000_HANDOFF_ENVIRONMENT.md',B/'reports/filtered15000_handoff_tools_20261027.md',O/'cpu_static_v1.json',O/'static_final_v2.json',O/'deploy_v1.json']:
 dst=D/src.relative_to(B);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
print(json.dumps({'status':'NEW_HANDOFF_STAGED_DISABLED_DOCUMENTED','actual_new_acceptance':False,'actual_new_training':False}))
