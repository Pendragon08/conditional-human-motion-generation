from pathlib import Path
import datetime, hashlib, json
W=Path(r'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/headless_smoke_doc_witness_v2')
p=W/'acceptance_report.json'
r=json.loads(p.read_text(encoding='utf-8'))
stderr=(W/'remote_files/headless_smoke_v2_stderr.txt').read_text(encoding='utf-8')
stdout=(W/'remote_files/headless_smoke_v2_stdout.txt').read_text(encoding='utf-8')
r['accepted']=False
r['outcome']='FAILED_BEFORE_BOUNDED_TRAINING_RECEIPT'
r['submit_invocation_count']=1
r['training_exit_code']=None
r['training_exit_code_reason']='Detached process exited before the first poll; no training receipt or exit marker was written. Only submit exit code 0 is observable.'
r['remote_stdout_exact']=stdout
r['remote_stderr_exact']=stderr
r['divergences']=[
 'The documented invocation submitted PID 3156839 successfully (submit exit code 0), but that process had already exited at the first read-only poll.',
 'The actual training entrypoint rejected its arguments: lab_run.py: error: argument --headless-authorization: expected one argument',
 'No receipt.json or executed_sources.json was written; all required continuation acceptance fields and receipt-to-source SHA256 binding remain unverified.'
]
expected={'status':'BOUNDED_TRAINING_COMPLETED','headless':True,'optimizer_loaded_exactly':True,'additional_ppo_iterations':2,'actual_adam_calls':40,'changed_policy_tensors':'> 0','executed_sources_unchanged':True,'parent_unchanged':True,'executed_sources_sha256_binding':'matching receipt binding and executed_sources.json bytes'}
r['checks']={key:{'expected':value,'actual':None,'passed':False,'state':'unverified_due_to_absent_receipt'} for key,value in expected.items()}
r['scope_observed']={'num_envs_requested':64,'ppo_iterations_requested':2,'device_requested':'cuda:1','cpu_affinity_requested':'16-19','threads_requested':4,'gpu1_free_mib_before':16330,'no_repairs':True,'no_retries':True,'no_process_stops':True,'no_source_or_dependency_changes':True,'no_external_messages':True}
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(W/'receipt_absence.json').write_text(json.dumps({'receipt_remote_path':'/data/code/ouyanghao/oyh/phygile-unitree-20261022/runs/headless_smoke_v2/receipt.json','receipt_exists':False,'executed_sources_exists':False,'observed_at':r['utc'],'poll_index':0},indent=2)+'\n',encoding='utf-8')
report='''# Headless continuation fresh document witness v2

Result: **FAIL — documented invocation did not reach bounded PPO completion.**

The documented PowerShell command was executed exactly once with the legacy Windows PowerShell shell and matched the invocation document byte-for-byte after trimming its code-block boundary. The submit command returned exit code 0 and remote PID 3156839 at 2026-10-22T11:50:53.255351+00:00. The launcher reported GPU1 had 16330 MiB free before submission.

The first read-only status poll found that PID absent. Training stdout was:

```text
There was an error running python
```

Training stderr ended with this exact failure:

```text
lab_run.py: error: argument --headless-authorization: expected one argument
```

The complete stderr, including argument usage, is preserved in `remote_files/headless_smoke_v2_stderr.txt`. The detached training exit code is unavailable; submit exit code 0 does not establish training completion.

Neither `receipt.json` nor `executed_sources.json` existed. Consequently the required completion status, headless flag, exact optimizer restore, two additional PPO iterations, 40 Adam calls, changed policy tensors, unchanged sources, unchanged parent, and receipt-to-source SHA256 binding are all unverified. No quality-improvement claim is supported.

The command, separate submit streams, full remote streams, launch record, read-only poll transport record, document snapshots, and SHA256 manifest are preserved in this directory. No repair, retry, source/dependency change, process interruption, deletion, or external notification was performed.
'''
(W/'ACCEPTANCE_REPORT.md').write_text(report,encoding='utf-8')
files=[]
for f in sorted(W.rglob('*')):
 if f.is_file() and f.name!='sha256_manifest.json':
  data=f.read_bytes();files.append({'path':str(f.relative_to(W)),'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
(W/'sha256_manifest.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'algorithm':'SHA256','files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'accepted':False,'directory':str(W),'files_preserved':len(files),'exact_error':'lab_run.py: error: argument --headless-authorization: expected one argument'},ensure_ascii=True))
