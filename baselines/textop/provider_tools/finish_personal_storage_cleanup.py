"""Verify complete GitHub recovery assets before personal storage cleanup.

No raw motion/model assets are published. The backup contains only an idle
third-party Python environment and fixed build/download caches with licenses.
"""
import datetime,hashlib,html,json,os,pathlib,shlex,shutil,subprocess,sys,time
from session import REPO,SSH,GH,remote
from restore_indexed_cache import sha
BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
DEST=BASE/'results/storage_cleanup_20261025';DEST.mkdir(parents=True,exist_ok=True)
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');TASK=REPO.parent.parent/'textop_reproduction_20261024'
WORK='/dev/shm/ouyanghao_storage_20261025';ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
TAG='inactive-engineai-environment-cache-20261025';REPOSITORY='Pendragon08/conditional-human-motion-generation'
ENV=os.environ.copy();ENV.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',PYTHONIOENCODING='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
def state(status,**values):
 p=DEST/'continuation.json';j=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
 j.update(status=status,time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**values)
 p.write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps(j),flush=True)
def gh(*args):return subprocess.check_output([GH,*args],env=ENV,text=True,encoding='utf-8')
def publish():
 p=subprocess.run([sys.executable,str(HERE/'publish.py')],env=ENV,capture_output=True,text=True,encoding='utf-8',errors='replace')
 if p.returncode:raise RuntimeError(p.stderr[-1800:])
 return json.loads((HERE/'github_sync.json').read_text(encoding='utf-8'))['verified_remote_head']
def read(path):
 try:return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
 except json.JSONDecodeError:return {}
def stream_part_to_github(item,release_id):
    """Stream SSH bytes to the official upload API; no local disk staging.

    Account token exists only in process memory and is sent only to the
    hardcoded official uploads.github.com endpoint. Never log/store it.
    """
    import http.client,ssl,urllib.parse
    token=gh('auth','token').strip();assert token
    command=[*SSH,'cat '+shlex.quote(WORK+'/'+item['name'])]
    error=(TASK/('storage_'+item['name']+'_ssh_stderr.log')).open('ab')
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=error)
    class SizedPipe:
        def __init__(self):self.count=0;self.hash=hashlib.sha256()
        def __len__(self):return item['bytes']
        def read(self,size=-1):
            block=child.stdout.read(1048576 if size<0 else size);self.count+=len(block);self.hash.update(block);return block
    body=SizedPipe()
    connection=http.client.HTTPSConnection('127.0.0.1',7897,timeout=900,context=ssl.create_default_context(),blocksize=1048576)
    connection.set_tunnel('uploads.github.com',443)
    try:
        connection.request('POST','/repos/'+REPOSITORY+'/releases/'+str(release_id)+'/assets?'+urllib.parse.urlencode({'name':item['name']}),body=body,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/octet-stream',
                     'Content-Length':str(item['bytes']),'X-GitHub-Api-Version':'2022-11-28'})
        assert child.wait(timeout=60)==0 and body.count==item['bytes'] and body.hash.hexdigest()==item['sha256']
        response=connection.getresponse();payload=response.read()
        assert response.status==201,'GitHub backup upload HTTP'+str(response.status)
        return json.loads(payload)
    finally:
        del token;connection.close();child.stdout.close();error.close()
        if child.poll() is None:child.terminate();child.wait(timeout=30)
def free_bmlrub_duplicate():
 receipt=read(BASE/'results/data_preparation_20261025/bmlrub_resume.json')
 if receipt.get('status')!='LOCAL_ORIGINAL_ARCHIVE_VERIFIED':return False
 source=TASK/'private_assets/incoming_amass/BMLrub_smplx_g.tar.bz2'
 partial=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives/BMLrub_smplx_g.tar.bz2.part')
 if not partial.exists():return True
 assert receipt['compressed_crc_and_tar_headers_verified'] and sha(source)==receipt['sha256']
 size=partial.stat().st_size;assert size==3555048892 and partial.is_file() and not partial.is_symlink()
 digest=sha(partial);h=hashlib.sha256();remaining=size
 with source.open('rb') as f:
  while remaining:
   b=f.read(min(8388608,remaining));assert b;h.update(b);remaining-=len(b)
 assert h.hexdigest()==digest
 plan={'status':'DUPLICATE_PARTIAL_RECOVERY_VERIFIED','target':str(partial.resolve()),'bytes':size,'sha256':digest,
       'retained_complete_archive':str(source.resolve()),'retained_archive_sha256':receipt['sha256'],
       'restore':'first3555048892 bytes of the retained full CRC-verified original archive','authoritative_datasets_deleted':0}
 p=DEST/'bmlrub_duplicate_recovery.json';p.write_text(json.dumps(plan,indent=2),encoding='utf-8');commit=publish()
 gh('api','repos/'+REPOSITORY+'/git/commits/'+commit)
 for rel in ['provider_tools/finish_personal_storage_cleanup.py','results/storage_cleanup_20261025/bmlrub_duplicate_recovery.json']:
  backed=subprocess.check_output(['git','-C',str(REPO),'show',commit+':baselines/textop/'+rel]);current=(BASE/rel).read_bytes()
  assert (json.loads(backed)==json.loads(current)) if rel.endswith('.json') else backed==current
 assert partial.resolve()==pathlib.Path(plan['target']) and pathlib.Path('D:/具身智能').resolve() in partial.resolve().parents
 assert sha(partial)==digest;before=shutil.disk_usage('D:/').free;partial.unlink()
 plan.update(status='REDUNDANT_PARTIAL_REMOVED_COMPLETE_ORIGINAL_RETAINED',backed_up_commit=commit,D_free_before=before,D_free_after=shutil.disk_usage('D:/').free)
 (DEST/'bmlrub_duplicate_cleanup.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');state('REDUNDANT_LOCAL_PARTIAL_REMOVED')
 return True
def main():
 assert not (DEST/'continuation.json').exists(),'Existing cleanup continuation retained; refuse duplicate'
 state('WAITING_FOR_FULL_ARCHIVE_VERIFICATION')
 for _ in range(240):
  proof=json.loads(remote("import pathlib; p=pathlib.Path('"+ROOT+"/receipts/storage_archive_20261025.json');print(p.read_text() if p.exists() else '{}')"))
  if proof:break
  time.sleep(15)
 else:raise TimeoutError('Archive not complete in1h; no deletion performed')
 assert proof['status']=='ARCHIVE_CRC_AND_EVERY_FILE_SHA_VERIFIED'
 (DEST/'archive_index.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
 for name in ['requirements-engineai.txt','pyvenv.cfg']:
  subprocess.run(['scp','-q','-o','BatchMode=yes','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+WORK+'/'+name,str(DEST/name)],check=True)
 notes='Complete recovery archive of the inactive personal EngineAI Python environment and regenerable caches before cleanup. Includes dependency licenses; no raw AMASS/SMPL/BABEL assets, experiment datasets or experiment checkpoints. Restore ordered archive pieces, verify hashes, concatenate, and extract at original paths. Current TextOp/GMR/IsaacSim environments are preserved.'
 note=TASK/'inactive_environment_release_notes.txt';note.write_text(notes,encoding='utf-8')
 result=subprocess.run([GH,'api','repos/'+REPOSITORY+'/releases/tags/'+TAG],env=ENV,capture_output=True,text=True,encoding='utf-8')
 if result.returncode:
  gh('release','create',TAG,'--repo',REPOSITORY,'--target','codex/textop-reproduction-20261024','--title','Recovery archive before personal server cleanup20261025','--notes-file',str(note))
 parts=[]
 for item in proof['parts']:
  release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG));old=next((x for x in release['assets'] if x['name']==item['name']),None)
  if old is None:
   state('STREAMING_BACKUP_TO_GITHUB_WITHOUT_LOCAL_DISK_COPY',part=item['name'])
   stream_part_to_github(item,release['id'])
   release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG));old=next(x for x in release['assets'] if x['name']==item['name'])
   assert old['state']=='uploaded' and old['size']==item['bytes'] and old.get('digest')=='sha256:'+item['sha256']
  assert old['size']==item['bytes'] and old.get('digest')=='sha256:'+item['sha256']
  parts.append({**item,'github_asset_id':old['id'],'github_digest':old['digest']})
  state('BACKUP_PART_VERIFIED',verified_parts=len(parts),total_parts=len(proof['parts']))
 index=DEST/'archive_index.json';gh('release','upload',TAG,str(index),str(DEST/'requirements-engineai.txt'),str(DEST/'pyvenv.cfg'),'--repo',REPOSITORY)
 release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG))
 auth={'all_asset_digests_verified':True,'release_url':release['html_url'],'parts':parts,
       'archive_sha256':proof['archive_sha256'],'manifest_sha256':proof['manifest_sha256']}
 (DEST/'github_backup_verified.json').write_text(json.dumps(auth,indent=2),encoding='utf-8');commit=publish()
 gh('api','repos/'+REPOSITORY+'/git/commits/'+commit)
 for rel in ['scripts/storage_recovery_cleanup.py','results/storage_cleanup_20261025/archive_index.json','results/storage_cleanup_20261025/github_backup_verified.json']:
  content=subprocess.check_output(['git','-C',str(REPO),'show',commit+':baselines/textop/'+rel]);current=(BASE/rel).read_bytes()
  assert (json.loads(content)==json.loads(current)) if rel.endswith('.json') else content==current
 auth['code_and_inventory_backed_up_commit']=commit
 authorized=TASK/'cleanup_apply_authorization.private.json';authorized.write_text(json.dumps(auth),encoding='utf-8')
 subprocess.run(['scp','-q','-o','BatchMode=yes','-i','C:/Users/Pende/.ssh/id_ed25519',str(authorized),'ouyanghao@10.147.17.32:'+WORK+'/github_backup_verified.json'],check=True)
 state('FULL_GITHUB_BACKUP_VERIFIED_APPLYING_FIXED_CLEANUP',commit=commit,release_url=release['html_url'])
 print(remote("import subprocess; p=subprocess.run(['python3','"+ROOT+"/scripts/storage_recovery_cleanup.py','apply'],capture_output=True,text=True,timeout=600);print(p.stdout);print(p.stderr[-3000:]);assert p.returncode==0",timeout=650),flush=True)
 state('IDLE_ENVIRONMENT_CACHES_CLEARED_RELOCATING_PRESERVED_ASSETS')
 print(remote("import subprocess; p=subprocess.run(['python3','"+ROOT+"/scripts/storage_recovery_cleanup.py','relocate'],capture_output=True,text=True,timeout=600);print(p.stdout);print(p.stderr[-3000:]);assert p.returncode==0",timeout=650),flush=True)
 data=json.loads(remote("import pathlib,json; r=pathlib.Path('"+ROOT+"/receipts');print(json.dumps({k:json.loads((r/('storage_'+k+'_20261025.json')).read_text()) for k in ['cleanup','relocation']}))"))
 free_bmlrub_duplicate()
 for k,j in data.items():(DEST/(k+'.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
 cleanup=data['cleanup'];move=data['relocation'];before=cleanup['free_before'];after=move['free_after']
 report=f'''# 服务器个人目录空间清理核验

完成时间：{move['completed_beijing']}（北京时间）。只操作ouyanghao个人目录，未停止训练/预处理进程。

| 磁盘 | 清理前可用 | 清理并迁移后可用 |
|---|---:|---:|
| 数据盘 | {before['/data']/1024**3:.2f}GiB | {after['/data']/1024**3:.2f}GiB |
| 系统盘 | {before['/home/ouyanghao']/1024**3:.2f}GiB | {after['/home/ouyanghao']/1024**3:.2f}GiB |

完整归档闲置EngineAI Python环境及指定旧编译、着色器、pip/安装包缓存，共{proof['source_bytes']/1024**3:.2f}GiB逻辑文件。逐个归档文件SHA与源文件一致、完整gzip/tar核验通过，GitHub分片大小/SHA256全部核对，再执行清理。恢复归档：{release['html_url']} 。该旧环境已归档，后续EngineAI任务需先恢复；框架源码、当前TextOp/GMR环境和共享IsaacSim平台均保留。

系统盘的TextOp资源与PhyGile保留资源已完整迁至个人数据盘retained-system-assets-20261025。逐文件SHA256一致，原/home路径保留为链接。数据集、人体模型、预训练模型、实验检查点与唯一实验记录的内容均保留；没有公开发布受限原始资产。

约17GiB旧CLoSD下载目录实际承载数据/预训练权重，未按缓存名误删。额外清理的本地BMLrub旧断点，仅在C盘完整原始包CRC/SHA与前缀核验、GitHub恢复证据备份后执行。

清理名单、依赖锁、恢复索引、GitHub资产校验及逐文件迁移证明保存在本目录。迁移释放系统盘但占用数据盘，两盘净变化以采样值为准，不将迁移量算作整体新增空间。
'''
 (BASE/'reports/personal_storage_cleanup_20261025.md').write_text(report,encoding='utf-8')
 for name,j in data.items():(LOCAL/('个人目录空间清理_'+name+'_20261025.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
 (LOCAL/'个人目录空间清理报告_20261025.md').write_text(report,encoding='utf-8')
 (LOCAL/'个人目录空间清理报告_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>个人目录空间清理报告</title><style>body{font:16px/1.8 system-ui;max-width:1000px;margin:30px auto;padding:24px}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
 state('CLEANUP_AND_ASSET_RELOCATION_VERIFIED',free_before=before,free_after=after,release_url=release['html_url'])
 for name in ['record_asset_authorization.py','collect_data_preparation.py','collect_storage_cleanup.py','collect_tracker_progress.py']:
  p=subprocess.run([sys.executable,str(HERE/name)],env=ENV,capture_output=True,text=True,encoding='utf-8',errors='replace')
  if p.returncode:raise RuntimeError(name+': '+p.stderr[-1500:])
 publish()
if __name__=='__main__':
 try:main()
 except BaseException as error:state('STOPPED_WITH_ERROR_RECOVERY_ASSETS_AND_SOURCE_RETAINED',error=repr(error));raise
