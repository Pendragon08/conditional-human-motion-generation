"""Reconcile positive remote evidence without modifying any training source."""
import datetime,hashlib,json,pathlib,re,shutil
T=pathlib.Path(__file__).resolve().parent
B=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/filtered15000_handoff_20261027'
P=O/'filtered15000_vae200k_train_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
j=json.loads((P/'heartbeat_20261027_2156.json').read_text(encoding='utf-8'))
c=json.loads((P/'heartbeat_20261027_2156_client.json').read_text(encoding='utf-8'))
r=j['files']['receipt']
assert c['actual_client_exit_code']==0 and c['client_reaped'] and c['streams_final']
assert j['run']=='filtered15000_vae200k_train_v1' and j['arrays_read']==j['model_loads']==0
assert r['status']=='RUNNING_FILTERED15000_VAE_TRAINING' and r['completed_primitive_attempts']==r['optimizer_updates']==41400 and r['skipped_updates']==0
assert {(x['pid'],x['start_ticks']) for x in j['workers']}=={(165374,'1432055921'),(165381,'1432055933')}
assert not j['common_lock_free'] and 'exit' not in j['files']
assert r['spec_sha256']==sha(B/'configs/filtered15000-vae-spec.json')=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
archive=P/'access_status_before_recovery_20261028_0557.json'
assert not archive.exists()
shutil.copy2(P/'access_status.json',archive)
old=json.loads(archive.read_text(encoding='utf-8'))
assert old['attempt_unix']<j['observed_unix']
new=dict(schema='textop.filtered15000_remote_observability.v1',status='REMOTE_SSH_RECOVERED_ACTUAL_TRAINING_VERIFIED',
 attempt_unix=j['observed_unix'],last_verified_observed_unix=j['observed_unix'],last_verified_step=41400,
 last_verified_optimizer_updates=41400,last_verified_skips=0,last_verified_progress_sha256=sha(P/'heartbeat_20261027_2156.json'),
 ssh_client_evidence_sha256=sha(P/'heartbeat_20261027_2156_client.json'),ssh_client_exit_code=0,ssh_client_reaped=True,
 previous_access_status_archive=archive.relative_to(B).as_posix(),previous_access_status_sha256=sha(archive),
 original_worker_identity_verified=True,training_exit_observed=False,server_training_changed=False,
 signals_sent=False,relaunch_attempted=False,network_settings_changed=False,
 previous_server_online_question_no_longer_blocks_observation=True,
 scope='Successful actual SSH observation; overlay CLI ONLINE state not separately asserted; no continuous monitoring during the outage')
(P/'access_status.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
tz=datetime.timezone(datetime.timedelta(hours=8))
sample=datetime.datetime.fromtimestamp(j['observed_unix'],tz).isoformat()
note=f'''最新远程观测已恢复（{sample}）：原run filtered15000_vae200k_train_v1实际41400尝试/41400Adam/0skip，原supervisor/worker的argv及startticks一致，尚无exit，共同GPU锁仍由此VAE占用。41.4k是实际进度，不是保存检查点；首个完整状态节点50k尚未到达。之前SSH超时和最后20.5k记录保留在access_status_before_recovery_20261028_0557.json及原客户端证据；期间运行连续性不据外推作保证。未更改网络、训练源码/spec，未重启、恢复、发信号或新增GPU/SDK。服务器是否在线的旧询问已不再阻塞观测与自动接续。'''
for name in ['CONTINUATION.md','README.md','FILTERED15000_HANDOFF_ENVIRONMENT.md']:
 p=B/name;text=p.read_text(encoding='utf-8')
 text,n=re.subn(r'<!--remote-observation-current-start-->.*?<!--remote-observation-current-end-->',
  '<!--remote-observation-current-start-->\n'+note+'\n<!--remote-observation-current-end-->',text,count=1,flags=re.S)
 assert n==1
 p.write_text(text,encoding='utf-8')
prior=json.loads((O/'vae_eta_user_request_20261028.json').read_text(encoding='utf-8'))
# Independent positive observations, never elapsed-time extrapolation of steps.
old_step=20500;old_unix=old['last_verified_observed_unix']
rate=(41400-old_step)/(j['observed_unix']-old_unix)
eta=j['observed_unix']+(200000-41400)/rate
estimate=dict(schema='textop.filtered15000_recovered_observation_eta.v1',sample_beijing=sample,
 actual_primitive_attempts=41400,actual_optimizer_updates=41400,skipped_updates=0,
 last_saved_checkpoint=None,first_checkpoint_step=50000,recent_actual_steps_per_hour=rate*3600,
 estimate_remaining_hours=(200000-41400)/rate/3600,
 projected_vae_completion_beijing=datetime.datetime.fromtimestamp(eta,tz).isoformat(),
 estimate_not_guarantee=True,later_curriculum_and_shared_gpu_speed_may_change=True,
 ldm_started=False,ldm_capacity_accepted=False,previous_manual_eta_evidence_preserved=True,
 ldm_start_requires_actual_vae200k_closure_quality_export_capacity_and_author_fresh_witnesses=True)
(O/'vae_eta_recovery_20261028_0557.json').write_text(json.dumps(estimate,ensure_ascii=False,indent=2),encoding='utf-8')
storage=json.loads((O/'storage_recovered_20261028_v1/observation.json').read_text(encoding='utf-8'))
summary=dict(storage)
# Publish aggregate capacity only, not the separate personal directory inventory.
summary['data_deficit_for_export_and_ldm_upper_bounds_bytes']=max(0,storage['ldm_required_bytes_including_cache']-storage['data_free_after_export_upper_bound'])
(B/'reports/filtered15000_remote_recovery_20261028.md').write_text(
 '# TextOp远程观测恢复与后续容量实测\n\n'+note+'\n\n'
 +f'按两次正面实核样本计算近期约{rate*3600:.0f}步/小时，剩余约{(200000-41400)/rate/3600:.1f}小时；外推约北京时间{datetime.datetime.fromtimestamp(eta,tz):%m月%d日%H:%M}完成VAE，仅速度估计，后段课程和共享GPU可能改变。LDM需真实200k状态、闭合退出、CPU导出、持久容量和两份实际梯度恢复见证后自动启动，不承诺固定开训时刻。\n\n'
 +f"只读容量样本：data空闲{storage['data']['free_bytes']}字节，HOME空闲{storage['home']['free_bytes']}字节；checkpoint目录无文件。LDM四槽+180MiB缓存+2GiB保留+512MiB日志要求{storage['ldm_required_bytes_including_cache']}字节；若同一data盘先保守分配完整VAE导出，尚差{summary['data_deficit_for_export_and_ldm_upper_bounds_bytes']}字节。该样本不是容量接受或预留。D空闲{storage['D']['free_bytes']}字节，2GiB保留后可容一份553648128字节上界状态，不能容全部四份。\n\n"
 +'本次仅元数据/容量与自有目录软件候选清单读取，没有模型/原数组加载、新GPU/SDK/训练作业、停机信号、文件删除或容量门槛降低。当前VAE五槽预算仍通过其实际运行检查。训练期间不修改活动入口和spec。全VAE私有备份及LDM实际容量仍待落实，所有原状态保留。\n',encoding='utf-8')
print(json.dumps(dict(observed=sample,step=41400,recent_steps_per_hour=rate*3600,eta=estimate['projected_vae_completion_beijing'],capacity_deficit_bytes=summary['data_deficit_for_export_and_ldm_upper_bounds_bytes'])))
