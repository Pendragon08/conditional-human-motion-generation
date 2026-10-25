"""Complete one new software-only backup; never repeats EngineAI cleanup."""
import base64,datetime,hashlib,json,os,pathlib,subprocess,sys,time
from session import REPO,SSH,GH,remote

BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
DEST=BASE/'results/idle_dreamcontrol_20261026'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
WORK='/dev/shm/ouyanghao_dreamcontrol_backup_20261026'
REPOSITORY='Pendragon08/conditional-human-motion-generation'
TAG='inactive-dreamcontrol-overlay-20261026'
ENV=os.environ.copy();ENV.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',PYTHONIOENCODING='utf-8')

def state(status,**values):
 p=DEST/'continuation.json';j=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
 j.update(status=status,time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**values)
 p.write_text(json.dumps(j,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'status':status,**values},ensure_ascii=True),flush=True)

def gh(*args):
 for attempt in range(3):
  r=subprocess.run([GH,*args],env=ENV,capture_output=True,text=True,encoding='utf-8',timeout=120)
  if r.returncode==0:return r.stdout
  if attempt==2 or args[0]!='api':raise RuntimeError('GitHub operation failed; source retained. '+r.stderr[-500:])
  time.sleep(2**attempt)

def read_server(path):
 return json.loads(remote('from pathlib import Path\np=Path('+repr(path)+');print(p.read_text() if p.exists() else "{}")'))

def publish():
 r=subprocess.run([sys.executable,str(HERE/'publish.py')],env=ENV,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
 (DEST/'publish.log').open('a',encoding='utf-8').write(r.stdout+'\n'+r.stderr+'\n')
 assert r.returncode==0,'Code/inventory publication failed; source retained'
 return json.loads((HERE/'github_sync.json').read_text())['verified_remote_head']

def reports(proof,cleanup=None):
 lines=['# 闲置 DreamControl 软件依赖的备份与空间回收','',
        '仅处理个人目录的 dreamcontrol-retarget 软件依赖目录。它不包含本次动作数据集、模型检查点或实验结果；当前 TextOp、GMR、IsaacSim 和 IsaacLab 依赖保留。', '',
        '**PBHC training 环境保留**：检查发现它是当前 TextOp Python 的基础环境，并被暂停的旧训练进程加载，因此未纳入清理。', '',
        '清理前核对当前进程映射、打开文件、环境变量，以及其他环境的解释器链接和运行时依赖。所有目标文件属于本人。', '',
        f"完整软件归档：{proof['archive_bytes']} 字节，逐文件校验 {proof['verified_files']} 个文件；归档 SHA256：`{proof['archive_sha256']}`。", '',
        f'[完整恢复包](https://github.com/{REPOSITORY}/releases/tag/{TAG}) 包含全部有序分片、逐文件清单、软件版本和许可证。只有全部 GitHub 分片摘要、大小及代码备份核验通过，才允许清理原软件副本。', '',
        '恢复：下载并按编号拼接 idle_dreamcontrol_overlay.tar.gz.part001 起的全部分片，核对各分片和完整包 SHA256，再解压到服务器原路径。原路径若已存在新环境，不得覆盖。恢复后需重新验证对应软件任务，不能仅凭解压成功称环境可用。', '']
 if cleanup:
  lines+=['已完成软件目录回收；数据集、检查点和实验记录删除数均为 0。',
          f"数据盘可用空间由 {cleanup['free_before']/1024**3:.2f} GiB 增至 {cleanup['free_after']/1024**3:.2f} GiB。", '',
          '此容量改善不能替代后续训练的完整磁盘预算。正式训练仍需数据物理筛选、训练集验收和资源检查。','']
 else:lines+=['当前仅完成本地归档内容验证，原软件目录仍保留。最终删除状态以 cleanup.json 为准。','']
 text='\n'.join(lines)
 (BASE/'DREAMCONTROL_SOFTWARE_RECOVERY.md').write_text(text,encoding='utf-8')
 (LOCAL/'闲置DreamControl软件备份与空间回收_20261026.md').write_text(text,encoding='utf-8')
 import html
 (LOCAL/'闲置DreamControl软件备份与空间回收_20261026.html').write_text('<!doctype html><meta charset="utf-8"><title>软件备份与空间回收</title><style>body{max-width:1000px;margin:40px auto;font:17px/1.8 sans-serif}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>',encoding='utf-8')

def main():
 DEST.mkdir(exist_ok=True)
 assert not (DEST/'continuation.json').exists(),'Existing operation retained; do not start duplicate coordinator'
 state('WAITING_FOR_NEW_SOFTWARE_ARCHIVE',datasets_deleted=0,checkpoints_deleted=0,records_deleted=0)
 for _ in range(180):
  proof=read_server(ROOT+'/receipts/idle_dreamcontrol_20261026/archive.json')
  if proof:break
  time.sleep(20)
 else:raise TimeoutError('New archive not complete; source retained')
 assert proof['status']=='ARCHIVE_ALL_MEMBERS_AND_FILE_HASHES_VERIFIED'
 (DEST/'archive_index.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
 packages=read_server(WORK+'/packages.json');(DEST/'packages.json').write_text(json.dumps(packages,indent=2),encoding='utf-8')
 reports(proof);publish()
 notes=DEST/'release_notes.txt'
 notes.write_text('Complete recovery backup of one inactive personal DreamControl Python package overlay, before authorized storage cleanup. Includes package licenses, full source inventory and hashes. No user motion datasets, body models, experiment checkpoints or results. Download every numbered part, verify hashes, concatenate, and restore only into an absent original path. Current TextOp/GMR/IsaacSim and PBHC base environment are retained.',encoding='utf-8')
 r=subprocess.run([GH,'api','repos/'+REPOSITORY+'/releases/tags/'+TAG],env=ENV,capture_output=True,text=True,encoding='utf-8',timeout=120)
 if r.returncode:
  assert '404' in r.stderr,'Could not establish release absence; no duplicate release or cleanup'
  gh('release','create',TAG,'--repo',REPOSITORY,'--target','codex/textop-reproduction-20261024','--title','Idle DreamControl software recovery 20261026','--notes-file',str(notes))
 release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG))
 source=(BASE/'scripts/upload_idle_dreamcontrol.py').read_bytes();encoded=base64.b64encode(source).decode()
 remote('from pathlib import Path\nimport base64\np=Path('+repr(ROOT+'/scripts/upload_idle_dreamcontrol.py')+');data=base64.b64decode('+repr(encoded)+')\nassert not p.exists() or p.read_bytes()==data\np.write_bytes(data)\nprint("UPLOADER_INSTALLED")')
 state('UPLOADING_NEW_SOFTWARE_RECOVERY',release_url=release['html_url'])
 token=gh('auth','token').strip();assert token
 payload=json.dumps({'token':token,'release_id':release['id']}).encode();del token
 try:
  r=subprocess.run([*SSH,'python3 '+ROOT+'/scripts/upload_idle_dreamcontrol.py'],input=payload,capture_output=True,timeout=7200)
 finally:del payload
 (DEST/'upload.log').write_bytes(r.stdout+r.stderr)
 (DEST/'upload_exit.json').write_text(json.dumps({'actual_exit_code':r.returncode,'credentials_stored':False}),encoding='utf-8')
 assert r.returncode==0,'Upload failed; source retained'
 verified=read_server(ROOT+'/receipts/idle_dreamcontrol_20261026/upload_verified.json')
 assert verified['status']=='ALL_GITHUB_SOFTWARE_ASSET_DIGESTS_VERIFIED'
 release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG))
 for item in verified['items']:
  asset=next(x for x in release['assets'] if x['name']==item['name'])
  assert asset['state']=='uploaded' and asset['size']==item['bytes'] and asset['digest']=='sha256:'+item['sha256']
 assert next(x for x in verified['items'] if x['name']=='source_manifest.json')['sha256']==proof['manifest_sha256']
 (DEST/'github_assets_verified.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
 commit=publish();gh('api','repos/'+REPOSITORY+'/git/commits/'+commit)
 for rel in ['scripts/archive_idle_dreamcontrol.py','scripts/upload_idle_dreamcontrol.py','provider_tools/finish_idle_dreamcontrol.py']:
  backed=subprocess.check_output(['git','-C',str(REPO),'show',commit+':baselines/textop/'+rel])
  assert backed.replace(b'\r\n',b'\n')==(BASE/rel).read_bytes().replace(b'\r\n',b'\n')
 auth={'repository':REPOSITORY,'tag':TAG,'release_url':release['html_url'],'all_digests_verified':True,
       'code_inventory_commit':commit,'archive_sha256':proof['archive_sha256'],'manifest_sha256':proof['manifest_sha256'],'parts':proof['parts']}
 (DEST/'cleanup_authorization.json').write_text(json.dumps(auth,indent=2),encoding='utf-8')
 state('FULL_GITHUB_BACKUP_VERIFIED_RECHECKING_IDLE_SOURCE',commit=commit)
 command='''
import pathlib,json,subprocess
work=pathlib.Path(%r);root=pathlib.Path(%r)
(work/'github_verified.json').write_text(%r)
out=root/'receipts/idle_dreamcontrol_20261026'
with (out/'apply.log').open('xb') as log:
 result=subprocess.run(['python3',str(root/'scripts/archive_idle_dreamcontrol.py'),'apply'],stdout=log,stderr=subprocess.STDOUT,timeout=240)
(out/'apply_exit.json').write_text(json.dumps({'actual_exit_code':result.returncode}))
assert result.returncode==0,'Cleanup failed; inspect retained evidence'
print((out/'cleanup.json').read_text())
'''%(WORK,ROOT,json.dumps(auth))
 cleanup=json.loads(remote(command,timeout=280));assert cleanup['status']=='IDLE_DREAMCONTROL_SOFTWARE_BACKED_UP_AND_REMOVED'
 for name in ['cleanup','apply_exit']:
  value=read_server(ROOT+'/receipts/idle_dreamcontrol_20261026/'+name+'.json')
  (DEST/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
 reports(proof,cleanup)
 state('IDLE_SOFTWARE_RECOVERY_AND_CLEANUP_VERIFIED',free_after=cleanup['free_after'],release_url=release['html_url'],datasets_deleted=0,checkpoints_deleted=0,records_deleted=0)
 publish()

if __name__=='__main__':
 try:main()
 except BaseException as error:
  if DEST.exists():state('FAILED_NO_FURTHER_DELETION',error_type=type(error).__name__,error=str(error)[-700:])
  raise
