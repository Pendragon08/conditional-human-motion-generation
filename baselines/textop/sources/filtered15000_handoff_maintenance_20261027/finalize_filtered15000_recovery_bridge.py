import pathlib,json,hashlib,shutil,ast
W=pathlib.Path(__file__).resolve().parents[2];T=W/'work/textop_reproduction_20261024';B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/filtered15000_handoff_tools_20261027'
source_names=['scripts/filtered15000_corpus_acceptance.py','scripts/prepare_filtered15000_recovery.py','scripts/supervise_filtered15000_recovery.py',
 'scripts/finalize_filtered15000_data.py','scripts/supervise_filtered15000_acceptance.py',
 'provider_tools/activate_filtered15000_recovery.py','provider_tools/backup_filtered15000_recovery.py',
 'provider_tools/sync_filtered15000_witness_activation.py','provider_tools/enable_filtered15000_vae_train.py',
 'provider_tools/launch_filtered15000_handoff.py','provider_tools/collect_filtered15000_handoff.py']
for n in source_names:ast.parse((B/n).read_text(encoding='utf-8'))
deployed=dict(json.loads((O/'recovery_bridge_deploy_v1.json').read_text(encoding='utf-8'))['written'])
layout=json.loads((O/'recovery_layout_fix_v2.json').read_text(encoding='utf-8'));final=json.loads((O/'recovery_finalizer_fix_v3.json').read_text(encoding='utf-8'))
deployed['scripts/filtered15000_corpus_acceptance.py']=layout['source_sha256'];deployed['scripts/finalize_filtered15000_data.py']=final['source_sha256'];deployed['configs/filtered15000-recovery-spec.json']=final['spec_sha256']
for n,h in deployed.items():assert hashlib.sha256((B/n).read_bytes()).hexdigest()==h,n
proof=dict(status='RECOVERY_AND_VAE_ENABLE_CURRENT_SOURCE_MANIFEST',deployed_source_sha256=deployed,
 local_provider_source_sha256={n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in source_names if n.startswith('provider_tools/')},
 prior_renderer_check_sha256=hashlib.sha256((O/'recovery_bridge_source_v1.json').read_bytes()).hexdigest(),
 build_verify_receipt_layout_reviewed=True,preparation_spec_archive_binding_reviewed=True,exact_receipt_bytes_carried=True,
 actual_corpus_recovery_jobs=0,actual_archives=0,actual_private_D_transfers=0,actual_acceptance_jobs=0,actual_new_model_loads=0,new_gpu_contexts=0,
 full_actual_acceptance_pending=True)
(O/'recovery_bridge_current_v3.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
planpath=B/'configs/teacher15000-full-to-generator-autostart-20261027.json';plan=json.loads(planpath.read_text(encoding='utf-8'))
plan.update(post_corpus_recovery_and_vae_enable_bridge_implemented=True,actual_private_recovery_prepared=False,actual_VAE_witnesses_enabled=False,
 recovery_spec='configs/filtered15000-recovery-spec.json',recovery_run='filtered15000_recovery_v1',data_acceptance_run='filtered15000_acceptance_v1',
 full_fresh_data_witnesses_and_actual_source_acceptance_pending=True,
 next_implementation='Complete full LDM entry/supervisor/real VAE200k inference export gates; execute post-filter acceptance only when full real inputs close')
planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
notice='''Post-corpus recovery/data/VAE enable bridges now implemented and staged
DISABLED; see reports/filtered15000_recovery_bridge_20261027.md and the updated
FILTERED15000_HANDOFF_ENVIRONMENT.md. No actual recovery/archive/D-transfer/
acceptance/gradient/model/GPU job has run. New CPU recovery requires two real
corpus witnesses and fresh actual SUMMARY; bounded16GiB/3GiB/900s supervision.
It preserves small norm/CLIP/private decisions plus derived-RAM rebuild recipe,
not duplicate large arrays. D2GiB/owner/path/bytes/SHA required before a separate
CPU acceptance job creates verified HOME capacity and enables only witnesses.
Preparation CPU receipt binds the archived pre-acceptance spec; only the
acceptance flag changes. Exact receipt bytes are returned separately by SHA.
Build-only split fields are not invented for verify-mode receipts. Earlier
inactive bytes/layout fixes remain preserved. Renderer/driver syntax checks are
source preparation only. New formal VAE gate requires two distinct real
gradient/Adam/RNG witnesses on the identical numeric/data/checkpoint signature;
all flags stayfalse until actual evidence. Full LDM entry/export acceptance is
still pending despite numerical runtime preparation. No user confirmation.
'''
for name in ['CONTINUATION.md','README.md']:
 p=B/name;old=p.read_text(encoding='utf-8');marker='<!--filtered15000-tools-current-end-->';assert marker in old and notice not in old
 p.write_text(old.replace(marker,notice+marker,1),encoding='utf-8')
report='''# 实际筛后语料恢复、数据接受及VAE启用桥

完整物理回放仍在正常执行。本轮只实现并部署禁用的新CPU恢复/接受代码和本地备份/启用桥，实际恢复作业、归档、D盘传输、数据接受、梯度见证、新模型或GPU上下文均为0。

接口闭合后可自动执行：两份真实corpus作者/fresh加载收据与退出 → 启用有界CPU恢复作业 → 私有小型归档server+D字节/SHA/所有权及2GiB保留核验 → 独立有界CPU数据接受与HOME持久容量核验 → 新VAE见证参数按原始字节/摘要同步 → 两份实际梯度/模型/Adam/RNG/后续批次恢复见证 → 启用并派发新VAE200k。所有步骤无需用户再次批准。

新CPU验证器只反序列化JSON并哈希已由真实GPU加载见证检查的缓存，不加载Torch/pickle/模型。构建阶段提供源与精确姿态隔离证明；verify阶段重载相同摘要的产物，不捏造不存在的build-only字段。私有小型恢复包包含归一化、CLIP文本缓存、真实逐源通过/失败决定和派生RAM缓存恢复说明；原数组和大型joblib不重复归档，全部原件保留。

CPU恢复与接受均有16GiB RAM、3GiB磁盘、900秒子进程及15秒资源监督。D传输有独占intent/输出/outcome和部分文件，退出未知先检查现场，不重派。数据接受比较真实准备收据对应的已归档准备spec，唯一允许的变化是接受标志；原始接受收据字节单独返回并验SHA。实际容量不足或已存在目标会保留现场失败，不删除或篡改来源。新VAE启用签名绑定严格审计、真实语料、数据、源码、seed/模型和持久检查点配置。四步见证权重不参与正式初始化。

新增11份源码语法及5段远端渲染源码、持久driver生成结果的解析检查已完成，没有执行模型或假造正面数据。静态复查发现并修正verify收据缺少build-only字段、启用接受后spec字节变化和本地JSON重序列化风险；此前禁用源字节与所有记录都保留，没有真实作业失败被删除。占位符采用唯一标记并在派发前解析远端源码。

这些桥尚未消费实际完整筛查或新语料输入，不是数据/训练就绪证明。实际作者/fresh文档见证、私有恢复备份及正面接受还需真实输入到达后自动执行。完整LDM数值核心已准备，完整entry/supervisor和真实VAE200k推理导出/容量/实际GPU验收仍待完成。
'''
p=B/'reports/filtered15000_recovery_bridge_20261027.md';p.write_text(report,encoding='utf-8')
M=B/'sources/filtered15000_handoff_maintenance_20261027'
for name in ['fix_filtered15000_recovery_receipt_layout.py','fix_filtered15000_recovery_finalizer.py','finalize_filtered15000_recovery_bridge.py']:
 dst=M/name;assert not dst.exists();shutil.copy2(T/name,dst)
for src in [p,planpath,O/'recovery_bridge_current_v3.json']:
 dst=D/src.relative_to(B);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
print(json.dumps({'status':proof['status'],'actual_new_acceptance':False,'actual_new_training':False}))
