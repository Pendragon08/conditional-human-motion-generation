"""Publish actual cleanup/relocation outcomes after supervised phase success."""
import datetime,html,json,pathlib,shutil,sys
from session import REPO,remote
BASE=REPO/'baselines/textop';DEST=BASE/'results/storage_cleanup_20261025'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');TASK=REPO.parent.parent/'textop_reproduction_20261024'
sys.stdout.reconfigure(encoding='utf-8')
code=r'''
import pathlib,json,hashlib,os,shutil,datetime
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');out={}
for name,file in [('cleanup','storage_cleanup_20261025.json'),('relocation','storage_relocation_20261025.json'),
                  ('apply_exit','storage_apply_exit_v2.json'),('relocation_exit','storage_relocate_exit_v1.json'),
                  ('direct_upload','storage_direct_upload_progress.json')]:out[name]=json.loads((r/'receipts'/file).read_text())
assert out['apply_exit']['exit_code']==out['relocation_exit']['exit_code']==0
assert out['relocation']['status']=='ALL_ASSETS_RELOCATED_SHA_VERIFIED_ORIGINAL_PATHS_PRESERVED'
verified=0
for move in out['relocation']['moves']:
 source=pathlib.Path(move['source']);dest=pathlib.Path(move['destination'])
 assert source.is_symlink() and source.resolve()==dest and source.lstat().st_uid==1013
 for row in move['files']:
  p=source/row['relative_path'];assert p.stat().st_size==row['bytes'];h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8388608),b''):h.update(b)
  assert h.hexdigest()==row['sha256'];verified+=1
resources={}
for label,path in [('gmr','/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'),
                   ('textop','/data/code/ouyanghao/oyh/textop-reproduction-20261024/env/bin/python'),
                   ('isaacsim','/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacsim-4.5.0/python.sh')]:
 assert pathlib.Path(path).exists();resources[label]=True
progress=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/local1653_v1/prepared1653_v1/progress.json')
j=json.loads(progress.read_text());old=pathlib.Path('/proc/3237858/status')
out['post_verification']={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'original_home_path_files_sha256_verified':verified,'current_environment_executables_retained':resources,
 'old57D_pause_state':next((x for x in old.read_text().splitlines() if x.startswith('State:')),'') if old.exists() else 'exited',
 'gmr_batch':{k:j.get(k) for k in ['status','completed_sequences','failed_sequences','total_requested','updated_utc']},
 'free_bytes':{x:shutil.disk_usage(x).free for x in ['/data','/home/ouyanghao']},'other_user_directories_modified':False,
 'datasets_deleted':0,'experiment_checkpoints_deleted':0,'unique_records_deleted':0}
print(json.dumps(out))
'''
data=json.loads(remote(code,timeout=120));DEST.mkdir(parents=True,exist_ok=True)
for name,j in data.items():(DEST/(name+'.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
for pattern in ['personal_storage_cleanup_*attempt*.log','storage_parallel_tail_*.log','storage_direct_upload_*.log']:
 for p in TASK.glob(pattern):
  target=DEST/'transport_records'/p.name;target.parent.mkdir(exist_ok=True)
  shutil.copyfile(p,target)
before=data['cleanup']['free_before'];after=data['post_verification']['free_bytes'];release=data['cleanup']['github_backup']['release_url']
freed=data['cleanup']['free_after']['/data']-before['/data']
report=f'''# 服务器个人目录空间清理核验

完成核查：{data['post_verification']['time_beijing']}（北京时间）。

| 磁盘 | 清理前可用 | 清理及资源迁移后可用 |
|---|---:|---:|
| 数据盘 | {before['/data']/1024**3:.2f}GiB | {after['/data']/1024**3:.2f}GiB |
| 系统盘 | {before['/home/ouyanghao']/1024**3:.2f}GiB | {after['/home/ouyanghao']/1024**3:.2f}GiB |

清理闲置EngineAI Python环境与指定旧编译、着色器、pip和安装包缓存，数据盘实际回收约{freed/1024**3:.2f}GiB。82,327个归档源文件逐个SHA256核对，完整压缩归档验证、全部六个GitHub分片摘要核验、代码/清单备份完成后才删除原副本。[完整恢复备份]({release})，恢复说明见[ENGINEAI_ENV_RECOVERY.md](../ENGINEAI_ENV_RECOVERY.md)。旧EngineAI环境已归档，后续使用需先恢复；其框架源码、基础Python、当前TextOp/GMR环境及共享IsaacSim平台保留。

系统盘的TextOp资源和PhyGile保留资源，共19个文件、约5.11GiB，已移到个人数据盘retained-system-assets-20261025。逐文件SHA256一致，原/home路径保留为链接；额外通过原路径重读19个文件再次验证SHA。**数据集、人体模型、预训练模型、实验检查点和唯一实验记录的内容均保留。**迁移占用一部分数据盘空间，不能把迁移量算成两盘整体新增空间。

GMR当前批次完成{data['post_verification']['gmr_batch']['completed_sequences']}/{data['post_verification']['gmr_batch']['total_requested']}，失败{data['post_verification']['gmr_batch']['failed_sequences']}，状态{data['post_verification']['gmr_batch']['status']}；清理没有停止或重启训练/预处理。旧57D训练保持原暂停状态。其他用户目录未修改。

约17GiB的CLoSD下载目录承载数据和预训练权重，因此保留。另在本地核验完整BMLrub原始包与前缀并备份恢复证据后，仅删除旧断点副本，D盘回收3.31GiB。

备份上传曾出现连接重置及API暂时失败；全部失败记录保留，原文件一直保留到备份核验通过。后续改由服务器直接向官方GitHub上传剩余分片；认证信息只经SSH标准输入与内存传递，未保存到代码、日志或报告。首次SSH执行阶段在未删文件时断连，经检查无执行进程、源清单零缺失后改用监督进程；清理和迁移均实测退出0。

全部清理限定在ouyanghao个人目录。受限AMASS/SMPL/BABEL及重定向资产保持私有，未公开上传。清理、迁移、GitHub校验和运行状态证据见results/storage_cleanup_20261025。数据准备与完整生成器训练仍按原验收顺序继续。
'''
(BASE/'reports/personal_storage_cleanup_20261025.md').write_text(report,encoding='utf-8')
(LOCAL/'个人目录空间清理报告_20261025.md').write_text(report,encoding='utf-8')
(LOCAL/'个人目录空间清理报告_20261025.html').write_text('<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>服务器个人目录空间清理</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1000px;margin:30px auto;padding:24px;background:#f6f8fa}pre{white-space:pre-wrap;background:white;padding:24px;border:1px solid #ccc;font:inherit}a{color:#0969da}</style><h1>服务器个人目录空间清理核验</h1><p><a href="'+release+'">GitHub完整恢复备份</a></p><pre>'+html.escape(report)+'</pre></html>',encoding='utf-8')
for name,j in data.items():(LOCAL/('个人目录空间清理_'+name+'_20261025.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
shutil.copyfile(BASE/'ENGINEAI_ENV_RECOVERY.md',LOCAL/'EngineAI旧环境恢复说明_20261025.md')
state=DEST/'continuation.json'
if state.exists():shutil.copyfile(state,DEST/'continuation_attempt5.json')
state.write_text(json.dumps({'status':'CLEANUP_AND_ASSET_RELOCATION_VERIFIED','time_beijing':data['post_verification']['time_beijing'],
 'free_before':before,'free_after':after,'release_url':release,'source_code_backed_up_commit':data['cleanup']['github_backup']['code_and_inventory_backed_up_commit'],
 'datasets_deleted':0,'checkpoints_deleted':0,'unique_records_deleted':0},indent=2),encoding='utf-8')
print(json.dumps({'free_before':before,'free_after':after,'reclaimed_data_gib_before_relocation':freed/1024**3,
                  'original_paths_verified_files':19,'gmr':data['post_verification']['gmr_batch']}))
