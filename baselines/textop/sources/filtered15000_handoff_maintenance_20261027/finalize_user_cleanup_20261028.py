import datetime,hashlib,json,pathlib,re,shutil,tomllib
T=pathlib.Path(__file__).resolve().parent;B=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
E=B/'results/storage_user_cleanup_20261028';D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((E/n).read_text(encoding='utf-8'))
d=read('D_cleanup.json');s=read('server_cleanup.json');copy=read('installer_private_backup.json');cap=read('capacity_after.json')
assert d['status']=='VERIFIED_REDUNDANT_TRANSPORT_PARTS_REMOVED' and d['full_archive_retained']
assert s['status']=='OLD_SOFTWARE_INSTALLER_MOVED_TO_PRIVATE_D_VERIFIED' and s['extracted_sdk_unchanged']
assert copy['gzip_full_CRC_verified'] and copy['sha256']=='c51d77ac1144b91b356f439c7313d8d59468147e37187ebffe7cdbc3154fdb8f'
assert cap['both_export_and_ldm_budget_currently_sufficient'] and cap['D_remaining_VAE_backup_budget_currently_sufficient']
P=B/'results/filtered15000_handoff_20261027/filtered15000_vae200k_train_v1'
j=json.loads((P/'user_cleanup_post_20261028.json').read_text(encoding='utf-8'));r=j['files']['receipt']
assert r['status']=='RUNNING_FILTERED15000_VAE_TRAINING' and r['optimizer_updates']==r['completed_primitive_attempts']==56600 and r['skipped_updates']==0
assert {(x['pid'],x['start_ticks']) for x in j['workers']}=={(165374,'1432055921'),(165381,'1432055933')}
assert not j['common_lock_free'] and 'exit' not in j['files']
net=cap['D_free_bytes']-d['D_free_before_bytes'];tz=datetime.timezone(datetime.timedelta(hours=8))
sample=datetime.datetime.fromtimestamp(j['observed_unix'],tz).isoformat()
status=dict(schema='textop.user_authorized_cleanup_completed.v1',status='TARGETED_CLEANUP_COMPLETE_CURRENT_STORAGE_BUDGETS_SUFFICIENT',
 latest_user_authorization='你帮我清理。',predeletion_verified_commit=s['code_commit'],D_duplicate_transport_bytes_removed=d['duplicate_logical_bytes'],
 SDK_installer_bytes_relocated=s['logical_bytes_relocated'],D_observed_net_free_gain_bytes=net,data_free_bytes=cap['data']['free_bytes'],D_free_bytes=cap['D_free_bytes'],
 data_required_for_export_and_LDM_bytes=cap['data_required_for_both_export_and_ldm_bytes'],D_required_for_remaining_VAE_states_bytes=cap['D_required_for_remaining_three_VAE_upper_bound_states_bytes'],
 current_budgets_passed=True,formal_LDM_activation_executed=False,capacity_reserved=False,additional_user_storage_input_required_now=False,
 sample_beijing=sample,VAE_attempts=56600,VAE_actual_optimizer_updates=56600,VAE_skips=0,original_worker_identity_verified=True,
 common_GPU_lock_still_held_by_VAE=True,datasets_removed=0,checkpoints_removed=0,unique_records_removed=0,new_gpu_contexts=0,signals_sent=False,
 D_cleanup_receipt_sha256=sha(E/'D_cleanup.json'),server_cleanup_receipt_sha256=sha(E/'server_cleanup.json'),
 private_installer_backup_receipt_sha256=sha(E/'installer_private_backup.json'),capacity_observation_sha256=sha(E/'capacity_after.json'))
(E/'final_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
note=f'''用户已授权“你帮我清理”，新指定清理已完成并真实退出0：只删除D盘四个与保留完整包逐字节相同的运输分片，回收1925545992字节；闲置IsaacGym安装包201141673字节先私有备份D并核SHA/完整gzip CRC后从server移除，其已解压SDK不变。D净增约{net/1024**3:.2f}GiB，data回收约{201141673/1024**2:.1f}MiB；复查data空闲{cap['data']['free_bytes']}B（导出+LDM预算{cap['data_required_for_both_export_and_ldm_bytes']}B），D空闲{cap['D_free_bytes']}B（余三份VAE预算{cap['D_required_for_remaining_three_VAE_upper_bound_states_bytes']}B），当前均通过。此为实际样本而非预留，200k后仍要最终容量/退出/质量/导出/作者fresh验收。原先额外空间问题已由本次清理解决，无需等待用户提供目录或重复催问。数据、所有checkpoint、完整归档及失败保留；0新GPU/SDK/停机信号。{sample}原VAE实核56600尝试/56600Adam/0skip，原worker身份和锁占用一致。50k完整状态及其D私有备份继续保留，不重复复制。'''
for n in ['CONTINUATION.md','README.md','FILTERED15000_HANDOFF_ENVIRONMENT.md']:
 p=B/n;text=p.read_text(encoding='utf-8')
 text,count=re.subn(r'<!--remote-observation-current-start-->.*?<!--remote-observation-current-end-->',
  '<!--remote-observation-current-start-->\n'+note+'\n<!--remote-observation-current-end-->',text,count=1,flags=re.S)
 assert count==1;p.write_text(text,encoding='utf-8')
report=f'''# 用户授权空间清理：实际完成与恢复位置

用户回复“你帮我清理”，本次新指定操作已实际退出0，数据、检查点、完整归档及唯一实验记录保留。

| 项目 | 实际结果 |
|---|---:|
| D盘四个重复运输分片回收 | {d['duplicate_logical_bytes']}字节（约{d['duplicate_logical_bytes']/1024**3:.2f}GiB） |
| 服务器旧安装包私有转移 | {s['logical_bytes_relocated']}字节（约{s['logical_bytes_relocated']/1024**2:.1f}MiB） |
| D盘扣除私有安装包后的实测净增 | {net}字节（约{net/1024**3:.2f}GiB） |
| data当前空闲 / 导出及LDM保守预算 | {cap['data']['free_bytes']} / {cap['data_required_for_both_export_and_ldm_bytes']}字节 |
| D当前空闲 / 其余三份VAE状态加2GiB保留 | {cap['D_free_bytes']} / {cap['D_required_for_remaining_three_VAE_upper_bound_states_bytes']}字节 |

上述当前预算均通过，无需继续等待用户提供额外空间；并非磁盘预留或最终LDM接受。开训前和每次保存仍重新检查，不放宽2GiB保留线。VAE持久HOME剩余四槽预算{cap['vae_remaining_slot_budget_bytes']}字节也通过；WBC、未来媒体和完整LDM私有备份容量仍按实际需要单独验收。

D盘仅删除hab-contact-aware-world-model/release/contact-v1.tar.gz.part000至part003。保留完整contact-v1.tar.gz，SHA8ef8118835c1b58b8a4475b2ee9322d8584cfb140ebbf3de5e75cc85d191f14c。完整包2404个成员逐个SHA通过，四片SHA与完整包偏移范围逐字节相同，范围重新生成到哈希计数器验证通过，现有私有GitHub Release四片摘要也一致。完整包、清单、模型和原始结果保留，恢复工具provider_tools/restore_contact_transport_part_20261028.py可按清单恢复任意一片。旧断点包和失败partial未清理。

服务器仅移除本人cache/downloads中的旧IsaacGym_Preview_4.tar.gz安装包副本，201141673字节，SHAc51d77ac1144b91b356f439c7313d8d59468147e37187ebffe7cdbc3154fdb8f。完整私有副本保留在D:/具身智能/TextOp复现_20261024/software_recovery_private_20261028/IsaacGym_Preview_4.tar.gz，真实SCPexit0、所有权/字节/SHA和完整gzip CRC核验通过。安装包移动本身不算整体新增空间；运行中的已解压SDK和依赖、活动env、保护的.venv-pbhc-training均未修改。NVIDIA安装包没有公开上传。

恢复工具、精确目标和来源证明在删除前已实核GitHub提交{d['backed_up_commit']}。Windows原生PowerShell逐个固定绝对路径执行非递归删除，真实exit0、逐文件日志及前后容量保留。服务器独占意图/结果与真实SSHexit0保留。最初全局句柄检查拒绝于修改前，后续识别不可读者为SSH/会话/凭据服务，权限限制保留；没有提升权限或声称全局完全可见。所有可读自身进程及原VAE训练进程无安装包引用。两次Git原始字节预检也在删除前拒绝：先改为核直接原始输出，再为本次证据路径固定-text并实核Git原始字节，原始证明未重序列化冒充同SHA，全部失败记录保留。

{sample}实际VAE56600尝试/56600Adam/0skip，原supervisor165374及worker165381的startticks一致，无exit、GPU锁仍被VAE持有。活动源码/spec/全部绑定哈希不变，旧57D仍T暂停。此次0新GPU/SDK/训练、0信号，未重启teacher/prefilter/旧协调器。继续自动VAE200k→实际导出与验收→LDM300k，不再询问开训。
'''
(B/'reports/user_authorized_storage_cleanup_20261028.md').write_text(report,encoding='utf-8')
C=tomllib.loads(pathlib.Path('C:/Users/Pende/.codex/automations/textop/automation.toml').read_text(encoding='utf-8'))
prefix=f'''2026-10-28用户最新明确回复“你帮我清理”。此新授权的精确空间回收已实际完成，之前“等待用户扩容/新目录”的状态不再适用，不重复催问。先读reports/user_authorized_storage_cleanup_20261028.md和results/storage_user_cleanup_20261028/final_summary.json及capacity_after.json。

当前真实采样{sample}：唯一原VAE run filtered15000_vae200k_train_v1已56600尝试/56600Adam/0skip，RUNNING，同原supervisor165374/startticks1432055921与worker165381/startticks1432055933，无exit，共同GPU锁仍占用。新实际收据覆盖本样本。原50k真实完整状态513526701B/SHAa2f9e9611c83d69ff5cd27ff842fc4db5cbe538c42f0ef08eab742a6e7287321，server+D字节备份和原生产model/Adam/allRNG/clock/pending-eval重载已核验，勿再复制/重载50k。实际进度56.6k不是新保存点。

仅删除D盘contact-v1.tar.gz.part000至003四个逐字节重复运输分片，共1925545992B；原D完整包SHA8ef8118835c1b58b8a4475b2ee9322d8584cfb140ebbf3de5e75cc85d191f14c、全部2404成员及清单/原结果保留，原私有GitHub资产摘要实核一致。可用restore_contact_transport_part_20261028.py按偏移恢复。服务器本人旧cache/downloads/IsaacGym_Preview_4.tar.gz共201141673B已先私有移到D/software_recovery_private_20261028，再原source稳定SHA/可读训练进程无引用核验后unlink；工作SDK、共享依赖和环境不变。私有SDK完整SHA c51d77ac1144b91b356f439c7313d8d59468147e37187ebffe7cdbc3154fdb8f及gzip CRC通过，不能把这个新保留的唯一冷恢复副本当成冗余再删，不公开NVIDIA安装包。

Windows/server实际exit0和独占intent/outcome/逐文件journal保留。原全局fd预检因系统凭据服务不可读拒绝，分类限制如实保留；两次Git CRLF原字节检查拒绝也保留在修改前，后来以直接原始输出和.gitattributes -text固定本次证据字节并重新GitHub实核后才删除，不放宽来源SHA。恢复代码与证明在删除前GitHub实核2ea1c63b096253a6f29899f3f8aa592dc9894d7d；最新postcommit查provider_tools/github_sync.json。软件/目录大清单.private.json不公开，原始实验数据/数组/逐源文本/轨迹/完整Adam始终私有。

当前data空闲{cap['data']['free_bytes']}B，HOME空闲{cap['home']['free_bytes']}B，D空闲{cap['D_free_bytes']}B。data须同时容VAE180MiB导出+LDM四槽/180MiB缓存/2GiB保留/512MiB日志，保守总{cap['data_required_for_both_export_and_ldm_bytes']}B，目前通过；D余三份VAE上界状态+2GiB保留需{cap['D_required_for_remaining_three_VAE_upper_bound_states_bytes']}B，目前也通过。VAE HOME剩余四槽预算{cap['vae_remaining_slot_budget_bytes']}B通过。数字只是实测，不是预留或最终LDM容量接受；未来动态消耗需每新job/保存复查，不能用此替代实际200k闭合/质量/导出/作者fresh验收。

已解决先前空间输入需求，无需等待目录或重新问开训。后续真实100k/150k/200k稳定保存且容量足够后，用provider_tools/backup_filtered15000_vae_checkpoint.py --step 实际节点，来源/字节/所有权/2GiB保留门槛不变；已有intent/partial/未知结果先核原client/worker，拒绝盲重派。模型加载与新GPU/SDK不得与VAE重叠。全LDM/WBC/媒体后续容量按实际预算再验，不宣称所有未来阶段已落实。

此轮清理0数据/完整checkpoint/唯一记录删除，0训练源码/spec修改，0新GPU/SDK/训练/信号。旧57D仍T，PhyGile/ReactiveBFM保持暂停，.venv-pbhc-training保护。当前清理已完成，不重跑本轮或旧EngineAI/DreamControl清理，不重复大库存盘/全库审计/旧夹具。健康每小时轻量收集安静，只有新实质结果、需处理失败、必要输入或完整效果通知。之前在线询问与空间询问均已不再阻塞。最新代码/聚合证据增量同步，私有安装包下载201141673B；不重复旧大资产或声称Git包估计就是实际VPN计费。

'''
prompt=prefix+C['prompt'][C['prompt'].index('继续已授权TextOp完整复现。'):]
args=dict(mode='update',id=C['id'],kind='heartbeat',name=C['name'],prompt=prompt,rrule=C['rrule'],status=C['status'],targetThreadId=C['target_thread_id'])
(T/'user_cleanup_automation_args.json').write_text(json.dumps(args,ensure_ascii=False),encoding='utf-8')
M=B/'sources/filtered15000_handoff_maintenance_20261027'
for n in ['finalize_user_cleanup_20261028.py','check_cleanup_capacity_20261028.py']:shutil.copy2(T/n,M/n)
print(json.dumps(status))
