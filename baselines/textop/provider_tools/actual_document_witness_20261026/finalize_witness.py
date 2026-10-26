import pathlib,sys,json,hashlib
W=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(B/'provider_tools'))
from session import remote
j=json.loads((W/'document_audit.json').read_text(encoding='utf-8'))
summary=json.loads((W/'document_verification.json').read_text(encoding='utf-8'))
raw=remote(r'''
import pathlib,hashlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run='quick_pilot_smoke_document_v1';prefix='quick_pilot_'+run
paths={'receipt':r/'tracker_validation'/run/'receipt.json','exit':r/'receipts'/(prefix+'_exit.json'),'process_exit':r/'receipts'/(prefix+'_process_exit.json'),'resources':r/'logs'/(prefix+'_resources.jsonl')}
print(json.dumps({k:{'raw':p.read_text(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}))
''',timeout=60)
payload=json.loads(raw)
for key,v in payload.items():
    data=v['raw'].encode('utf-8')
    assert hashlib.sha256(data).hexdigest()==v['sha256']==j['document_hashes'][key]
    p=W/(key+'.server.'+('jsonl' if key=='resources' else 'json'))
    with p.open('xb') as target:target.write(data)
report='''# Fresh actual document smoke witness

Result: **DOCUMENT_ACTUAL_AUDIT_PASSED**. From server project `/data/code/ouyanghao/oyh/textop-reproduction-20261024`, executed once verbatim:

```text
python3 scripts/supervise_quick_pilot.py --stage smoke --run quick_pilot_smoke_document_v1
```

The actual SSH/supervisor invocation exited **0**, and the evaluator child exited **0**, was reaped, with supervision_passed=true, timed_out=false, no supervisor/sample errors, and no supervisor shutdown signals. Ten resource samples include before_launch, after_launch and after_exit. Total command observation was119.25s; evaluator supervision was117.97s.

All **8 trials closed and naturally completed;0 failed**. There were63 vectorized control calls and252 scene physics steps,8 environments,0 optimizer updates. Private records contain458 total pose samples, lengths[52,53,54,56,58,60,61,64], covering allT−1 required control transitions for each whole selected original. All8 succeeded, so the actual failure branch was not exercised.

Independent audit read private binary/NPZ arrays in place on the server using only the Python standard library: all64 pose/action/target arrays match the binary exactly, values are finite, NPZ CRC passes,29 joint names agree,50Hz times are exact, and reference sequence is initial0 then consumed0..T−2. All8 terminal states were captured before autoreset. No partial files or owned open private output descriptors remain. Raw per-motion arrays were neither downloaded nor written under publicBASE/results.

output_writers_closed=true. **native_close_returned=false; teardown_fallback_used=true**. A bounded fallback exit after closed writers is evidence of successful controlled teardown, not native SDK-close success. The final reference frameT−1 endpoint is not separately compared. This is an early5000-checkpoint diagnostic on8 shortest complete originals, not full-library filtering, accepted corpus, generator training or paper-complete quality.

Before and after: teacher checkpoint SHA cfb443ab9ac80a615fdffdd881ede38b6e1997a0993752ff2a82a9f7c9939fae; original trainer PID3812580 absent; common lock available; no active personal SDK worker. GPU1free40726MiB, CPU affinity[0,1,2,3], Torch limit0.65 from the unchanged spec/code; peak Torch allocation8822339584bytes. Existing paused PID3237858 and all other workloads were preserved. All23 unique bound runtime/source hashes, quick spec and teacher provenance artifacts remained unchanged.

## Preserved preflight rule decision

Initial v1 made0 SDK launches and stopped at the generic skill memory.used<500MiB criterion. Its scripts, exact traceback, failure_state.json and verification.json remain unchanged. GPU1used7823MiB was attributable to an existing stopped process; free40726MiB already satisfied the documented supervisor.

The parent task supplied trusted evidence of prior explicit user authorization “允许共享，先做限额训练验证” with shared/headlessGPU1, CPU0–3 andTorch0.65. This supersedes the skill's default empty-GPU criterion. shared_gpu_authorization_v2.md and separate preflight_v2 files record this applicability decision. V2 retained every real supervisor threshold, source/config byte and no-overlap check. No documented SDK failure was retried; the original uncreated run name was invoked once only.

## Document versus reality

The documented invocation required no flag, dependency, configuration, threshold, source or path workaround. No invocation divergence. The document explicitly disclosed native-close fallback and incomplete final-endpoint validation; observed reality agrees. The initial generic skill/shared-GPU applicability issue is preserved above rather than hidden.

## Bound server evidence

receipt SHA256:4b3d537d3f299c7909ad46b2dec8c3d96155c3cd8b1f1969d82a9675a1ede3a1

exit SHA256:192741862d250b4b0459d8e21049a82bdf7e76e1f75295d8290e0c8b88395120

process_exit SHA256:b832144f70443cfd2dfcc2fd7a95efcc7c91a2e5de890f811cc6517f2a3ffe80

resources SHA256:00295e482ddfedee625a31321332444fb499bd1ac7ee115b6320cac99a6ff1b1

Exact UTF-8 server receipt/exit/process_exit/resources bytes are saved as *.server.json/jsonl. document_execution.json retains real transport exit/log hashes; document_audit.json and document_verification.json retain the independent aggregate audit. Parent owns private archival and any later sample gating. No Git/publish or shared source/config/results mutations performed by this witness.
'''
with (W/'REPORT.md').open('x',encoding='utf-8') as target:target.write(report)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in W.iterdir() if p.is_file() and p.name!='witness_sha256.json'}
with (W/'witness_sha256.json').open('x',encoding='utf-8') as target:json.dump(manifest,target,indent=2,ensure_ascii=False)
print(json.dumps({'status':summary['status'],'report':str(W/'REPORT.md'),'witness_files':len(manifest),'raw_server_evidence_hashes_match':True}))
