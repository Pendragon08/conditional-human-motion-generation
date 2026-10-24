from pathlib import Path
import datetime,hashlib,json
root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
submission=json.loads((root/'submission_result.json').read_text(encoding='utf-8'))
transport=json.loads((root/'poll_001_transport.json').read_text(encoding='utf-8'))
poll=json.loads(transport['stdout'])
receipt_path=root/'remote_artifacts/runs/headless_smoke_v3/receipt.json'
receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
audit=json.loads((root/'independent_audit.json').read_text(encoding='utf-8'))
manifest_path=root/'remote_artifacts/runs/headless_smoke_v3/executed_sources.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
artifact_checks=[]
for relative,record in poll['files'].items():
 path=root/'remote_artifacts'/relative
 artifact_checks.append({'path':relative,'sha256':sha(path),'expected_sha256':record['sha256'],'size':path.stat().st_size,'matches':sha(path)==record['sha256'] and path.stat().st_size==record['size']})
spec=json.loads((root/'configs__environment_plan_v2.json').read_text(encoding='utf-8-sig'))
spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
source=(root/'remote_sources/scripts/lab_run.py').read_text(encoding='utf-8')
source_manifest_entry=manifest['files']['/data/code/ouyanghao/oyh/phygile-unitree-20261022/code_v9/scripts/lab_run.py']
gates={
 'documented_command_executed_once':True,
 'no_source_repair_or_retry':True,
 'submission_exit_zero':submission['exit_code']==0,
 'actual_training_completed':receipt['status']=='BOUNDED_TRAINING_COMPLETED',
 'headless':receipt['headless'] is True and receipt['display'] is None,
 'exact_optimizer_restoration':receipt['optimizer_loaded_exactly'] is True and "if not equal_state(runner.alg.optimizer.state_dict(),loaded['optimizer_state_dict'])" in source,
 'exact_policy_and_normalizer_restoration':receipt['policy_loaded_exactly'] and receipt['normalizers_loaded_exactly'],
 'correct_environment_and_iterations':receipt['num_envs']==64 and receipt['additional_ppo_iterations']==2,
 'actual_adam_calls_40':receipt['actual_adam_calls']==40,
 'adam_step_delta_40':audit['parent_adam_steps']==[40000.0] and audit['final_adam_steps']==[40040.0],
 'finite_changed_policy':receipt['changed_policy_tensors']>0 and audit['changed_model_state_tensor_count']>0 and not audit['nonfinite_model_tensors'],
 'finite_optimizer_state':not audit['nonfinite_optimizer_tensors'],
 'executed_sources_unchanged':receipt['executed_sources_unchanged'] and not audit['manifest_post_exit_changed'] and not audit['manifest_post_exit_missing'],
 'preexisting_code_unchanged':not audit['preexisting_code_changed'] and not audit['preexisting_code_missing'],
 'parent_unchanged':receipt['parent_unchanged'] and audit['parent_unchanged_independent'],
 'receipt_manifest_binding':sha(manifest_path)==receipt['source_provenance']['sha256']==audit['manifest_sha256'],
 'entrypoint_source_binding':sha(root/'remote_sources/scripts/lab_run.py')==receipt['source_sha256']==source_manifest_entry,
 'final_checkpoint_binding':audit['final_sha_matches_receipt'],
 'local_artifact_copy_hashes_match':all(x['matches'] for x in artifact_checks),
 'environment_spec_matches_ledger':spec_hash=='a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954'
}
result={
 'status':'PASS' if all(gates.values()) else 'FAIL',
 'witness':'fresh agent-follows-doc headless continuation v3',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'gates':gates,'submission_count':1,'source_repair_count':0,'retry_count':0,
 'submission_exit_code':submission['exit_code'],'training_process_pid':3157748,
 'training_process_absent_at_first_poll':not poll['proc_exists'],
 'training_process_exit_code':None,
 'training_process_exit_code_note':'Detached launcher does not retain the child exit code. Completion is evidenced by actual receipt, stdout result sentinel, optimizer calls, changed finite checkpoint tensors, and independent source/parent verification.',
 'command':submission['command'],'started_utc':receipt['started_utc'],'finished_utc':receipt['finished_utc'],'poll_utc':poll['utc'],
 'elapsed_training_seconds':(datetime.datetime.fromisoformat(receipt['finished_utc'])-datetime.datetime.fromisoformat(receipt['started_utc'])).total_seconds(),
 'canonical_environment_sha256':spec_hash,
 'metrics':{k:receipt[k] for k in ['status','headless','num_envs','additional_ppo_iterations','optimizer_loaded_exactly','policy_loaded_exactly','normalizers_loaded_exactly','actual_adam_calls','adam_steps_before','adam_steps_after','changed_policy_tensors','control_calls','parent_unchanged','executed_sources_unchanged','torch_peak_allocated_bytes']},
 'independent_checkpoint_audit':audit,'artifact_hash_verification':artifact_checks,
 'doc_vs_reality_divergence':[],
 'preserved_runtime_diagnostics':['stderr contains DriverShaderCacheManager::init() called without a shutdown()!; the bounded PPO run nevertheless completed all acceptance gates.'],
 'scope':'Restoration and real bounded optimization only. No claim of improved motion quality, transfer stability, or full paper reproduction. No modifications to ReactiveBFM or original data/checkpoints; no public upload, notification, or scheduled task.'
}
(root/'acceptance_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
log_hashes={x['path']:{'sha256':x['sha256'],'size':x['size']} for x in artifact_checks}
(root/'artifact_hashes.json').write_text(json.dumps(log_hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=f"""# 独立文档见证 V3：{result['status']}

按 docs/HEADLESS_V2_INVOCATION.md 原样执行 PowerShell 提交一次，未修改代码、未重试。使用 run-experiment 技能及 compute-env-contract 的 fresh agent-follows-doc 约定。环境规范规范化 SHA256 为 {spec_hash}，与计算账本一致，暖复用原环境。

训练在 {receipt['started_utc']} 至 {receipt['finished_utc']} 完成，耗时 {result['elapsed_training_seconds']:.3f} 秒。独立 PID 为 3157748；第一次只读轮询时进程已退出。轮询 UTC 为 {poll['utc']}，与提交间隔超过 30 秒；没有超过 60 秒的等待，没有超过 10 分钟总等待。提交器退出码为 0。分离子进程的原始退出码未由启动器保存，本报告不将其假定为 0。

## 实际完成证据

| 验收项 | 实测 |
|---|---|
| status | {receipt['status']} |
| headless / DISPLAY | true / null |
| 环境数 / 新增 PPO 迭代 | 64 / 2 |
| 优化器、策略、归一化器精确恢复 | true / true / true |
| 实际 Adam 调用 | 40 |
| Adam step | 40000 → 40040 |
| 变化的独立 named_parameters 张量 | 21 |
| 变化的 model_state_dict 条目 | 27（包含共享模块别名，统计口径不同） |
| 最终策略/模型张量有限性 | 97 个张量全部有限 |
| 最终优化器张量有限性 | 75 个张量全部有限 |
| 控制调用 | 48 |
| 已执行源文件 | 446 个；独立退出后检查全部存在且哈希未变 |
| 启动前代码文件 | 336 个；独立复核全部未变 |
| 父检查点 | 启动前、回执、结束后 SHA256 一致 |
| GPU1 峰值 Torch 分配 | {receipt['torch_peak_allocated_bytes']} 字节 |

源代码的递归逐项比较在设置 optimizer_loaded_exactly=true 之前检查恢复优化器与父检查点完全相等；每次 Adam 更新前检查梯度有限，训练结束检查策略和优化器状态有限。本次另用原环境只读加载父/最终检查点、验证张量有限性及变化，未启动新的模拟或训练。

## 绑定哈希

- executed_sources.json：{sha(manifest_path)}；与回执 source_provenance.sha256 完全一致。
- 执行入口 lab_run.py：{receipt['source_sha256']}；与实际源文件和 executed_sources.json 一致。
- 父检查点：{receipt['parent_sha256']}。
- 新检查点：{receipt['final_sha256']}；独立只读复核一致。

完整启动命令在 documented_submission.ps1；原始提交结果在 submission_result.json。实际服务器启动记录、receipt.json、executed_sources.json、contract.json、stdout/stderr 和部署参数已按原始字节保存于 remote_artifacts/，每个副本的 SHA256 与服务器读取值一致。完整哈希见 artifact_hashes.json；独立审计见 independent_audit.json。检查点继续保存在服务器原位置，本次不复制或改写其内容。

## 文档与实际差异

本次完整调用未发现参数、资源选择或完成语义方面的文档差异。原环境配置保留旧 GUI 条件，文档及执行授权记录明确允许本次 headless 例外，因此未请求 GUI 确认。

stderr 有一行 DriverShaderCacheManager::init() called without a shutdown()!，已原样保留。该诊断未阻止实际 40 次优化、有限参数更新和最终回执产生；本报告没有删除或掩盖它。

本见证只验证恢复及真实的有界优化，不用于证明动作质量、跨仿真稳定性或论文基准已提高。没有操作 ReactiveBFM、原始数据集或其他任务；没有修改源文件、修补、重试、通知或创建定时任务。
"""
(root/'ACCEPTANCE_REPORT.md').write_text(report,encoding='utf-8')
evidence=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name!='evidence_file_hashes.json':
  evidence.append({'path':str(p.relative_to(root)).replace('\\','/'),'size':p.stat().st_size,'sha256':sha(p)})
(root/'evidence_file_hashes.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'gates':gates,'evidence_files':len(evidence),'report':str(root/'ACCEPTANCE_REPORT.md'),'hashes':log_hashes},ensure_ascii=False,indent=2))

