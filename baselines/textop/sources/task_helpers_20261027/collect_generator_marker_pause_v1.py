import base64,hashlib,json,pathlib,shutil,subprocess,sys
TASK=pathlib.Path(__file__).resolve().parent;BASE=TASK.parent/'phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(BASE/'provider_tools'));from session import remote,SSH
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
code=r'''import base64,hashlib,json,pathlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');o=r/'generator_pause_20261027/generator8278_to_filtered_order_v1'
receipt=json.loads((o/'receipt.json').read_text());assert receipt['status']=='USER_PAUSED_GENERATOR_CHECKPOINT_PRIVATE_BACKUP_PENDING'
audit=json.loads((o/'checkpoint_cpu_audit.json').read_text());assert audit['status']=='OWN_TRAINER_FULL_STATE_CPU_AUDIT_PASSED'
assert audit['original_file_unchanged'] and audit['model_exact_cpu_reload'] and audit['adam_exact_cpu_reload'] and not audit['cuda_initialized']
names=['receipt.json','checkpoint_cpu_audit.json','marker_control.json','original_receipt_before_pause.json']
proof={n:base64.b64encode((o/n).read_bytes()).decode() for n in names}
print(json.dumps({'receipt':receipt,'audit':audit,'proof':proof}))
'''
j=json.loads(remote(code,timeout=30));ck=j['audit']['checkpoint']
dest=D/'checkpoints/private/generator8278_prefilter_20261027';dest.mkdir(parents=True,exist_ok=True)
f=dest/pathlib.Path(ck['path']).name
assert not f.exists() and not f.with_suffix('.partial').exists(),'Preserve any previous download; inspect rather than duplicate'
assert shutil.disk_usage(D).free>ck['bytes']+2*1024**3
partial=f.with_suffix('.partial')
cmd=['scp']+SSH[1:-1]+[SSH[-1]+':'+ck['path'],str(partial)]
copied=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
assert copied.returncode==0,copied.stderr.decode('utf-8',errors='replace')
assert partial.stat().st_size==ck['bytes'] and sha(partial)==ck['sha256'];partial.replace(f)
backup={'status':'PREFILTER_GENERATOR_PAUSED_CHECKPOINT_PRIVATE_BACKUP_VERIFIED','server_path':ck['path'],'local_private_path':str(f),'bytes':ck['bytes'],'sha256':ck['sha256'],
 'completed_attempts':ck['manager_step'],'optimizer_updates':ck['actual_optimizer_updates'],'skips':ck['skipped_updates'],'source_deleted':False,'training_resumed':False,
 'cpu_audit':j['audit'],'real_worker_exited':True,'common_gpu_sdk_lock_released':True,'prefilter_comparison_only':True,'whole_library_filtering_complete':False}
out=BASE/'results/generator_pause_20261027';private=D/'diagnostics/generator_pause_20261027';private.mkdir(parents=True,exist_ok=True)
for n,enc in j['proof'].items():
 data=base64.b64decode(enc);(out/('production_'+n)).write_bytes(data);(private/n).write_bytes(data)
(out/'private_backup_verified.json').write_text(json.dumps(backup,indent=2),encoding='utf-8');(private/'private_backup_verified.json').write_bytes((out/'private_backup_verified.json').read_bytes())
encoded=base64.b64encode((out/'private_backup_verified.json').read_bytes()).decode()
receipt_code=r'''import base64,hashlib,json,pathlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');data=base64.b64decode(ENC)
p=r/'receipts/generator8278_user_pause_verified_20261027_v1.json';assert not p.exists();p.write_bytes(data)
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(data).hexdigest()}))
'''.replace('ENC',repr(encoded))
accepted=json.loads(remote(receipt_code,timeout=30));(out/'final_acceptance_location.json').write_text(json.dumps(accepted,indent=2),encoding='utf-8')
print(json.dumps({'backup':{k:v for k,v in backup.items() if k!='cpu_audit'},'acceptance':accepted,'D_free':shutil.disk_usage(D).free}))
