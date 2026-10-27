from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,time,datetime,sys
base=Path('D:/具身智能/PhyGile复现/Unitree框架适配_20261022')
witness=base/'production_active_doc_witness_v3'
output=base/'production_active_validation_v3'
work=Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work')
scripts=work/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts'
unitree=scripts.parent
local=work/'phygile_unitree_20261022'
assets=base/'private_assets_contact_v3'
command="& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/validate_production_active_v3.py' --scripts 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/production_active_validation_v3'"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    (witness/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not output.exists(),'Fresh production output already exists: do not rerun'
invocation=local/'ACTIVE_EXPORT_INVOCATION_V3.md'
doc=invocation.read_text(encoding='utf-8')
assert command in doc,'Command must match invocation verbatim'
specpath=unitree/'configs/environment_plan_v2.json'
ledgerpath=unitree/'docs/COMPUTE_LEDGER.md'
spec=json.loads(specpath.read_text(encoding='utf-8'))
canonical=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
ledger=ledgerpath.read_text(encoding='utf-8')
assert f'canonical_sha256: {canonical}' in ledger,'Warm environment spec differs from ledger'
(witness/'environment_canonical_hash.txt').write_text(canonical+'\n',encoding='utf-8')
(witness/'exact_command.txt').write_text(command+'\n',encoding='utf-8')
tracked=[scripts/n for n in ['export_active_policy.py','frozen_policy.py','active_policy_artifact.py']]+[local/'validate_production_active_v3.py',assets/'encoder_actor_critic.py',assets/'contract.json',assets/'parent_checkpoint.pt',specpath,ledgerpath,invocation,Path('C:/Users/Pende/.codex/skills/run-experiment/SKILL.md'),Path('C:/Users/Pende/.codex/skills/shared-references/compute-env-contract.md')]
for prior in ['production_active_doc_witness_v2','production_active_validation_v2']:
    tracked.extend(sorted(p for p in (base/prior).rglob('*') if p.is_file()))
tracked.append(base/'production_batch_diagnosis_v2.json')
before={str(p):sha(p) for p in tracked}
save('before_hashes.json',before)
archive=witness/'source_snapshot';archive.mkdir()
snapshot_map={}
for i,p in enumerate(tracked[:12]):
    if p.name=='parent_checkpoint.pt': continue
    target=archive/f'{i:02d}_{p.name}'
    shutil.copyfile(p,target)
    snapshot_map[str(p)]=str(target)
save('source_snapshot_map.json',snapshot_map)
ps51=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
wrapper=witness/'invoke_exact.ps1'
wrapper_text="$ErrorActionPreference = 'Continue'\n[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)\n"+command+"\n$witnessActualExit = $LASTEXITCODE\n[System.IO.File]::WriteAllText('D:/具身智能/PhyGile复现/Unitree框架适配_20261022/production_active_doc_witness_v3/actual_exit.txt', [string]$witnessActualExit, (New-Object System.Text.UTF8Encoding($false)))\nexit $witnessActualExit\n"
wrapper.write_text(wrapper_text,encoding='utf-8-sig')
assert wrapper.read_bytes().startswith(b'\xef\xbb\xbf')
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
start=time.monotonic()
with (witness/'full_stdout.txt').open('xb') as so,(witness/'full_stderr.txt').open('xb') as se:
    process=subprocess.Popen([str(ps51),'-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(wrapper)],stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
    (witness/'process_id.txt').write_text(str(process.pid),encoding='ascii')
    print(json.dumps({'launched_once':True,'pid':process.pid,'powershell':str(ps51),'wrapper_utf8_bom':True}),flush=True)
    timed_out=False
    while process.poll() is None:
        remaining=540-(time.monotonic()-start)
        if remaining<=0:
            timed_out=True;process.kill();break
        try: process.wait(timeout=min(30,remaining))
        except subprocess.TimeoutExpired: print('Witness running; no retry or repair.',flush=True)
    outer_exit=process.wait(timeout=30)
finished=datetime.datetime.now(datetime.timezone.utc).isoformat()
actualpath=witness/'actual_exit.txt'
actual_exit=int(actualpath.read_text(encoding='utf-8')) if actualpath.exists() else None
save('process_receipt.json',{'started_utc':started,'finished_utc':finished,'elapsed_seconds':time.monotonic()-start,'powershell_process_exit_code':outer_exit,'actual_native_exit_code':actual_exit,'timed_out':timed_out,'documented_command_executions':1,'powershell_executable':str(ps51),'wrapper_has_utf8_bom':wrapper.read_bytes().startswith(b'\xef\xbb\xbf')})
after={str(p):{'before':before[str(p)],'after':sha(p),'unchanged':before[str(p)]==sha(p)} for p in tracked}
save('after_hash_checks.json',after)
sourcechecks={str(p):{'original':str(p),'snapshot':s,'before':before[str(p)],'snapshot_sha256':sha(s),'unchanged':sha(s)==before[str(p)]==sha(p)} for p,s in [(Path(p),s) for p,s in snapshot_map.items()]}
save('source_archive_hash_checks.json',sourcechecks)
proofpath=output/'production_proof.json'
checks={'actual_exit_zero':actual_exit==0 and outer_exit==0,'all_tracked_original_bytes_unchanged':all(x['unchanged'] for x in after.values()),'all_witness_source_snapshot_bytes_match':all(x['unchanged'] for x in sourcechecks.values()),'warm_spec_matches_ledger':True,'v2_preserved':all(x['unchanged'] for p,x in after.items() if 'production_active_' in p and '_v2' in p),'command_invoked_exactly_once':True}
proof=None
if proofpath.exists():
    proof=json.loads(proofpath.read_text(encoding='utf-8'))
    shutil.copyfile(proofpath,witness/'production_proof.json')
    batch=proof['batch_checks']
    checks['strict_same_batch_bitwise_full_vs_compact_all_four_sizes']=sorted(x['batch'] for x in batch)==[1,7,128,1024] and all(x['same_batch_full_vs_compact_bitwise_equal'] and x['max_abs_action_delta']==0.0 for x in batch)
    checks['batch1024_fixture_bitwise_equal']=any(x['batch']==1024 and x['cross_batch_fixture_delta']==0.0 for x in batch)
    checks['cross_batch_fixture_within_existing_1e_5']=all(x['cross_batch_fixture_delta']<=1e-5 and x['cross_batch_fixture_tolerance']==1e-5 for x in batch)
    checks['omitted_branches_forbidden_hooks']=proof['dropped_branches_guarded'] is True
    checks['no_optimizer_or_physics_updates']=proof['optimizer_or_physics_updates']==0
    checks['synthetic_proxy_only']=proof['evaluation_type']=='synthetic_proxy'
    archived={}
    for p,digest in proof['source_files_sha256'].items():
        snap=output/'source_snapshot'/Path(p).name
        archived[p]={'reported_sha256':digest,'before_sha256':before[p],'current_sha256':sha(p),'archived_sha256':sha(snap),'unchanged':digest==before[p]==sha(p)==sha(snap)}
    save('production_source_archive_hash_checks.json',archived)
    checks['all_executed_source_snapshots_match_original_bytes']=all(x['unchanged'] for x in archived.values())
    exportproof=output/'export/artifact_proof.json'
    if exportproof.exists():
        shutil.copyfile(exportproof,witness/'artifact_proof.json')
        ep=json.loads(exportproof.read_text(encoding='utf-8'))
        checks['exporter_passed_with_bitwise_parity']=ep['status']=='BITWISE_ACTIVE_INFERENCE_PARITY' and ep['max_abs_action_delta']==0.0
        checks['full_resume_checkpoint_preserved']=ep['source_checkpoint_sha256']==before[str(assets/'parent_checkpoint.pt')]==sha(assets/'parent_checkpoint.pt')
        checks['compact_resume_disabled']=ep['training_resume_supported'] is False
    for name in ['export_stdout.txt','export_stderr.txt']:
        shutil.copyfile(output/name,witness/name)
status='PASS' if proof is not None and all(checks.values()) else 'FAIL'
report={'status':status,'checks':checks,'batch_checks':proof['batch_checks'] if proof else None,'evidence_dir':str(witness),'production_output_dir':str(output),'scope':'synthetic_proxy; CPU inference correctness only; no training, optimizer, physics, environment changes or server operations','exact_command':command,'actual_exit':actual_exit,'powershell_process_exit':outer_exit,'canonical_environment_sha256':canonical,'preserved_v2_failure_and_diagnosis':checks['v2_preserved']}
save('witness_report.json',report)
(witness/'witness_report.md').write_text('# Fresh active-export document witness V3\n\n'+status+'\n\nExact documented invocation ran once in Windows PowerShell 5.1 using a UTF-8 BOM wrapper.\n\n'+''.join(f'- {k}: {v}\n' for k,v in checks.items())+'\nScope: synthetic_proxy, CPU inference correctness only; no training or physics.\n',encoding='utf-8')
manifest={str(p.relative_to(witness)):sha(p) for p in witness.rglob('*') if p.is_file() and p.name!='evidence_hashes.json'}
save('evidence_hashes.json',manifest)
print(json.dumps(report,ensure_ascii=False),flush=True)
sys.exit(0 if status=='PASS' else 1)