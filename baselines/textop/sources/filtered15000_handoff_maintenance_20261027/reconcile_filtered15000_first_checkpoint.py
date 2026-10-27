"""Accept real first-save byte backup metadata and update human reports."""
import datetime,hashlib,json,pathlib,re,shutil
T=pathlib.Path(__file__).resolve().parent
B=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/filtered15000_handoff_20261027';P=O/'filtered15000_vae200k_train_v1'
E=B/'results/filtered15000_vae_checkpoint_backups_20261028/step_50000'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
j=json.loads((P/'heartbeat_20261027_2252.json').read_text(encoding='utf-8'))
c=json.loads((P/'heartbeat_20261027_2252_client.json').read_text(encoding='utf-8'))
r=j['files']['receipt'];result=json.loads((E/'backup.json').read_text(encoding='utf-8'))
assert c['actual_client_exit_code']==0 and c['client_reaped'] and c['streams_final']
assert r['completed_primitive_attempts']==r['optimizer_updates']==50600 and r['skipped_updates']==0
assert r['status']=='RUNNING_FILTERED15000_VAE_TRAINING' and not j['common_lock_free']
assert {(x['pid'],x['start_ticks']) for x in j['workers']}=={(165374,'1432055921'),(165381,'1432055933')}
ck=r['checkpoints'][0];assert len(r['checkpoints'])==1 and ck==result['checkpoint']
assert ck['manager_step']==ck['actual_optimizer_updates']==50000 and ck['bytes']==513526701
assert ck['sha256']=='a2f9e9611c83d69ff5cd27ff842fc4db5cbe538c42f0ef08eab742a6e7287321'
assert result['status']=='PRIVATE_FILTERED15000_VAE_BYTE_BACKUP_VERIFIED' and result['D_free_after_bytes']>=2*1024**3
assert result['source_fingerprint_stable'] and not result['checkpoint_payload_deserialized']
assert sha(B/'configs/filtered15000-vae-spec.json')==result['spec_sha256']=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
tz=datetime.timezone(datetime.timedelta(hours=8));sample=datetime.datetime.fromtimestamp(j['observed_unix'],tz).isoformat()
old=json.loads((P/'access_status.json').read_text(encoding='utf-8'))
archive=P/'access_status_recovered41400_0557.json';assert not archive.exists();shutil.copy2(P/'access_status.json',archive)
old.update(status='REMOTE_SSH_AVAILABLE_ACTUAL_TRAINING_VERIFIED',attempt_unix=j['observed_unix'],
 last_verified_observed_unix=j['observed_unix'],last_verified_step=50600,last_verified_optimizer_updates=50600,
 last_verified_progress_sha256=sha(P/'heartbeat_20261027_2252.json'),ssh_client_evidence_sha256=sha(P/'heartbeat_20261027_2252_client.json'),
 latest_verified_checkpoint_step=50000,latest_private_byte_backup_sha256=sha(E/'backup.json'))
(P/'access_status.json').write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
note=f'''最新实际采样（{sample}）：唯一原run filtered15000_vae200k_train_v1已50600尝试/50600Adam/0skip，原supervisor/worker身份一致，RUNNING、无exit，共同锁仍被VAE持有。首个真实50000完整状态已稳定保存，513526701字节/SHA {ck['sha256']}，生产进程报告精确model/Adam/all RNG/clock/pending-eval重载通过后继续训练；新工具只读绑定来源并核验前后稳定inode/字节/SHA，D私有完整状态复制已真实exit0及字节/SHA/所有权验证，D仍保留2GiB。此次复制没有独立加载张量或新梯度验收，也没有新GPU/SDK、重启或停机信号。旧远程失败与恢复证据完整保留。'''
for name in ['CONTINUATION.md','README.md','FILTERED15000_HANDOFF_ENVIRONMENT.md']:
 p=B/name;text=p.read_text(encoding='utf-8')
 text,n=re.subn(r'<!--remote-observation-current-start-->.*?<!--remote-observation-current-end-->',
  '<!--remote-observation-current-start-->\n'+note+'\n<!--remote-observation-current-end-->',text,count=1,flags=re.S)
 assert n==1;p.write_text(text,encoding='utf-8')
needed_after=2*1024**3+3*553648128
deficit=max(0,needed_after-result['D_free_after_bytes'])
report='''# TextOp新筛后VAE首次真实完整状态与私有备份

'''+note+f'''

私有目标：{result['D_checkpoint']}。源码工具：provider_tools/backup_filtered15000_vae_checkpoint.py；它只接受同一实际筛后run、冻结spec/数值/语料/接受绑定及真实checkpoint收据。所有SSH/SCP都保留唯一intent/outcome/直接streams，失败partial不删除或覆盖。每个真实步数只运行一次，已有intent/partial或未知客户端须先查现场，不能盲重派。此50k工具实际验证前后原文件同一指纹、大小/SHA，SCP真实exit0，D完整文件SHA/字节和原生Windows所有权，复制后2GiB保留线仍满足。新元数据12个负例拒绝检查只算来源协议检查，实际复制才形成字节备份。

原50k文件仍在HOME保留，D副本私有，不发布完整Adam或数组。此次无Torch/模型/原数组反序列化、新GPU/SDK/训练或信号，也不改变任何活动生产source/spec。生产进程自身精确重载证明和本工具独立字节证明区分保留。

后续容量尚未接受：D复制后当时空闲{result['D_free_after_bytes']}字节，再保守容纳其余三份553648128字节完整状态并维持2GiB保留需{needed_after}字节，缺{deficit}字节。先前服务器data在VAE导出+完整LDM预算下差141750272字节。新只读软件候选细查cache约1503232字节、sources159744字节、upstream18685952字节；即使加14MB归档总量也不足以填补data缺口，且未证明可清理，当前依赖与数据不动，0删除。

已一次异步请求用户为data增加至少1GiB、D增加至少2GiB，或提供可用私有存储目录。这是存储输入需求，不是重新确认开训。当前VAE的现有五槽预算正常，继续运行；之后真实200k闭合/质量、CPU导出、容量和作者fresh见证通过即自动LDM300k，不恢复teacher/prefilter8100。存储未满足时不降低保留线或删状态。
'''
(B/'reports/filtered15000_vae_first_checkpoint_backup_20261028.md').write_text(report,encoding='utf-8')
summary=dict(schema='textop.filtered15000_first_checkpoint_handoff.v1',sample_beijing=sample,
 verified_progress=50600,optimizer_updates=50600,skips=0,checkpoint_step=50000,checkpoint_bytes=ck['bytes'],checkpoint_sha256=ck['sha256'],
 private_backup_receipt_sha256=sha(E/'backup.json'),private_D_backup_verified=True,
 D_free_after_bytes=result['D_free_after_bytes'],D_deficit_for_remaining_three_upper_bound_states=deficit,
 server_data_prior_ldm_export_deficit_bytes=141750272,additional_storage_requested_once=True,
 training_healthy_at_sample=True,ldm_started=False,ldm_capacity_accepted=False,new_gpu_contexts=0,signals_sent=False,files_deleted=0)
(O/'first_checkpoint_backup_20261028.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
