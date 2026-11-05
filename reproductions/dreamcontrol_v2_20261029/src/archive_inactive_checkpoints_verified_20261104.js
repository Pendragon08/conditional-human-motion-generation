// Retain every byte of inactive checkpoints in verified local cold storage.
// Current diffusion/Pick final models, all raw data, logs and reports stay put.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{execFileSync}=require('child_process');
const L=path.resolve(__dirname,'..'),C=path.join(L,'checkpoint_archive_20261104'),D='D:/DreamControl_v2_checkpoint_archive_20261104';
const HOST='10.147.17.32',ROOT='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029';
fs.mkdirSync(C,{recursive:true});fs.mkdirSync(D,{recursive:true});
const planFile=path.join(C,'manifest.json');
function remote(script){return execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python -'],{input:script,encoding:'utf8',timeout:60000});}
function hash(file){return new Promise((resolve,reject)=>{const h=crypto.createHash('sha256'),s=fs.createReadStream(file);s.on('data',b=>h.update(b));s.on('end',()=>resolve(h.digest('hex')));s.on('error',reject);});}
function save(j){const t=planFile+'.tmp';fs.writeFileSync(t,JSON.stringify(j,null,2));fs.renameSync(t,planFile);}
const script=String.raw`import hashlib,json,os
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';S=R/'stage3_pick29_20261103'
paths=[R/'stage2_prior29_20261103/runs/grab_omni29_mdm_baseline_v1'/n for n in ['step_000000100.pt','step_000005000.pt']]
paths += [D/'reports'/n for n in ['physx_model_author_smoke_v1/integration_checkpoint.pt','physx_model_doc_witness_v1/integration_checkpoint.pt',
 'online_doc_v1/prefix/resume_partial_72.pt','online_doc_v1/resumed/resume_final.pt','online_author_v1/prefix/resume_partial_72.pt','online_author_v1/resumed/resume_final.pt']]
old=S/'runs/pick29_8912_3000_resume_v3/ppo'
paths += sorted([p for p in old.glob('model_*.pt') if int(p.stem.split('_')[1])<2999],key=lambda p:int(p.stem.split('_')[1]))[:50]
assert len(paths)==58
protected=[R/'stage2_prior29_20261103/runs/grab_omni29_mdm_baseline_v1/step_000010000.pt',old/'model_2999.pt']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
rows=[]
for p in paths:
 p=p.resolve(strict=True);assert p.is_relative_to(R.resolve()) and p not in protected
 stat=p.stat();rows.append({'remote_path':str(p),'relative_path':str(p.relative_to(R)),'bytes':stat.st_size,
  'sha256':sha(p),'inode':stat.st_ino,'device':stat.st_dev,'mtime_ns':str(stat.st_mtime_ns),
  'class':'inactive_diagnostic_checkpoint' if p.is_relative_to(D) else 'inactive_intermediate_checkpoint',
  'destination_drive':'D' if p.is_relative_to(D) else 'C'})
print(json.dumps({'schema':1,'scope':'verified cold archive; all checkpoint bytes retained, only inactive owned project intermediates/diagnostics',
 'rows':rows,'total_bytes':sum(r['bytes'] for r in rows),'protected_final_models':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in protected]}))
`;
(async()=>{
 let j=fs.existsSync(planFile)?JSON.parse(fs.readFileSync(planFile,'utf8')):JSON.parse(remote(script));
 if(j.status==='archived_verified_complete'){console.log(JSON.stringify({status:j.status,bytes:j.total_bytes}));return;}
 for(const row of j.rows){
  const base=row.destination_drive==='D'?D:C,target=path.resolve(base,...row.relative_path.split('/'));
  if(!target.startsWith(path.resolve(base)+path.sep))throw Error('archive destination outside owned directory');
  row.local_path=target;fs.mkdirSync(path.dirname(target),{recursive:true});
 }
 j.status='copying_before_any_remote_removal';save(j);
 for(const row of j.rows){
  if(fs.existsSync(row.local_path)&&fs.statSync(row.local_path).size===row.bytes&&await hash(row.local_path)===row.sha256){row.local_verified=true;save(j);continue;}
  if(fs.existsSync(row.local_path))throw Error('incomplete or mismatched copy retained; inspect '+row.local_path);
  const stat=fs.statfsSync(path.parse(row.local_path).root),free=stat.bavail*stat.bsize;
  if(free-row.bytes<1024**3+100*1024**2)throw Error('local archive disk reserve; no remote deletion');
  execFileSync('scp',['-q','-p','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST+':'+row.remote_path,row.local_path],{stdio:'inherit',timeout:300000});
  const fd=fs.openSync(row.local_path,'r+');fs.fsyncSync(fd);fs.closeSync(fd);
  if(fs.statSync(row.local_path).size!==row.bytes||await hash(row.local_path)!==row.sha256)throw Error('archive hash mismatch; retain remote');
  row.local_verified=true;save(j);console.log(JSON.stringify({copied_verified:row.relative_path,bytes:row.bytes}));
 }
 j.status='all_copies_verified_before_remote_archival';save(j);
 // Perform a second complete read of local copies before releasing server space.
 for(const row of j.rows)if(!row.local_verified||await hash(row.local_path)!==row.sha256)throw Error('second local archive verification failed');
 const encoded=Buffer.from(JSON.stringify(j)).toString('base64');
 const finish=String.raw`import base64,hashlib,json,os,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');j=json.loads(base64.b64decode('PAYLOAD'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
assert all(r['local_verified'] and r['local_path'] for r in j['rows'])
for final in j['protected_final_models']:assert sha(Path(final['path']))==final['sha256']
for row in j['rows']:
 p=Path(row['remote_path']).resolve(strict=True);assert p.is_relative_to(R.resolve())
 assert '/data/GRAB/' not in str(p) and '/data/OmniRetarget_Dataset/' not in str(p)
 assert p.name not in ['step_000010000.pt','model_2999.pt']
 st=p.stat();assert (st.st_size,st.st_ino,st.st_dev)==(row['bytes'],row['inode'],row['device']) and sha(p)==row['sha256']
 # Old v1 JSON encoded ns as an integer, exceeding JavaScript's exact range.
 # Only the timestamp representation may differ by sub-microsecond rounding;
 # size/inode/device and the entire cryptographic content remain exact checks.
 assert abs(st.st_mtime_ns-int(row['mtime_ns']))<=1024
 row['mtime_ns']=str(st.st_mtime_ns)
 # No file currently opened by any accessible process may be archived.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:fds=list((proc/'fd').iterdir())
  except (FileNotFoundError,PermissionError):continue
  for fd in fds:
   try:target=os.readlink(fd)
   except (FileNotFoundError,PermissionError):continue
   assert target!=str(p),'checkpoint currently open by process '+proc.name
index=R/'checkpoint_archival_20261104';index.mkdir(exist_ok=True)
manifest=index/'manifest.json';assert not manifest.exists()
j['status']='all_local_copies_verified_remote_release_in_progress';manifest.open('x').write(json.dumps(j,indent=2))
before=shutil.disk_usage('/data').free
for row in j['rows']:
 p=Path(row['remote_path']);assert sha(p)==row['sha256']
 marker=p.with_suffix(p.suffix+'.archived.json');assert not marker.exists()
 marker.open('x').write(json.dumps({'status':'verified_cold_archive','original_path':str(p),'bytes':row['bytes'],'sha256':row['sha256'],
  'local_archive_path':row['local_path'],'archive_manifest':str(manifest),'restore':'restore byte-identical file from local archive before any access; original checkpoint content is retained'},indent=2))
 p.unlink();row['remote_archived']=True
 tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2));tmp.replace(manifest)
for final in j['protected_final_models']:assert sha(Path(final['path']))==final['sha256']
j['status']='archived_verified_complete';j['server_free_before_bytes']=before;j['server_free_after_bytes']=shutil.disk_usage('/data').free
tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2));tmp.replace(manifest)
print(json.dumps(j))
`.replace('PAYLOAD',encoded);
 j=JSON.parse(remote(finish));save(j);
 fs.writeFileSync(path.join(C,'归档说明.md'),'# 本项目非活跃检查点校验归档\n\n所有58个中间/诊断检查点的完整文件均已保留在本机C与D盘，逐文件两次SHA256校验后释放服务器对应副本；原始人体/机器人数据、当前扩散最终模型、旧Pick最终模型、所有日志报告和参考轨迹留在服务器。manifest.json逐项记录原路径、本机路径、尺寸、SHA256和恢复位置。恢复时先核本机SHA，再上传至原路径，核服务器SHA后即可使用。\n\n服务器归档索引：'+ROOT+'/checkpoint_archival_20261104/manifest.json。\n');
 console.log(JSON.stringify({status:j.status,files:j.rows.length,bytes:j.total_bytes,server_free_bytes:j.server_free_after_bytes,manifest:planFile}));
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
