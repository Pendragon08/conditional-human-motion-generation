"""Read/copy public code and aggregate inventory only; no experiment launch."""
from pathlib import Path
import json,hashlib,shutil,subprocess,sys,re,datetime
T=Path(__file__).resolve().parent;R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation'
B=R/'baselines/textop';D=Path('D:/具身智能/TextOp复现_20261024')
private=D/'diagnostics/github_sync_inventory_20261027'
O=B/'results/github_sync_inventory_20261027';O.mkdir(exist_ok=True)
H=B/'sources/task_helpers_20261027';H.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'provider_tools'));from session import remote
text=remote("from pathlib import Path\nprint(Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/audit_candidate_collision_v1.py').read_text(),end='')",timeout=45)
assert hashlib.sha256(text.encode()).hexdigest()=='a4d18d7c1031ca72d11fc7b9fbf1e63d2ed31efae6a8c86d277e7b845509bc74'
target=B/'scripts/audit_candidate_collision_v1.py'
assert not target.exists()
target.write_bytes(text.encode())
patterns=[r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}',r'hf_[A-Za-z0-9]{30,}']
rows=[]
# Original84 one-off helpers only. This inventory helper is not a past experiment.
helpers=[p for p in sorted(T.glob('*.py')) if p.name not in ['prepare_public_sync_inventory_20261027.py','sync_public_inventory_20261027.py']]
assert len(helpers)==84
for p in helpers:
 b=p.read_bytes();s=b.decode('utf-8-sig')
 assert not any(re.search(x,s) for x in patterns),p.name
 q=H/p.name;assert not q.exists();q.write_bytes(b)
 assert hashlib.sha256(q.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()
 rows.append({'original_workspace_path':'work/textop_reproduction_20261024/'+p.name,'archived_path':q.relative_to(B).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'executed_during_sync':False})
(H/'.gitattributes').write_text('*.py -text\n',encoding='utf-8')
(H/'README.md').write_text("""# Historical local task helpers — source archive only

These84 exact-byte files preserve one-off deployment, inspection, collection and
publication helpers used during this TextOp reproduction. They are archived
source, NOT current launch commands. Some encode old57D, oldgenerator-first,
cleanup, signal or previous-run paths and priorities. Do not execute them merely
because they are archived here. No helper was run as part of copying this archive.

Use ../../CONTINUATION.md and the current stage environment document for active
instructions. Runtime scripts/configs live under ../../scripts and ../../configs;
the original environments, failures, current teacher and saved checkpoints are
unchanged. Original workspace paths and SHA256 values are in manifest.json.
This directory disables Git line-ending conversion for .py to preserve bytes.
No licensed arrays, per-source annotations, checkpoint contents or credentials
are included. The scripts may reference private local files without embedding them.
""",encoding='utf-8')
(H/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
server=json.loads((private/'server_code_inventory.private.json').read_text(encoding='utf-8'))
for x in server['code_files']:
 b=(B/x['path']).read_bytes()
 if hashlib.sha256(b).hexdigest()==x['sha256']:x['repository_match']='exact_bytes'
 else:
  assert hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()==x['sha256']
  x['repository_match']='CRLF_to_LF_reconstruction_exact'
server.update(inspected_code_only=True,server_files_changed=False,active_experiment_changed=False)
(O/'server_code_coverage.json').write_text(json.dumps(server,indent=2),encoding='utf-8')
releases=json.loads((private/'github_releases_metadata.json').read_text(encoding='utf-8'))
(O/'existing_release_assets.json').write_text(json.dumps(releases,indent=2),encoding='utf-8')
stats=json.loads((private/'local_private_categories.json').read_text(encoding='utf-8'))
scope={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository_visibility':'public','scope':'TextOp current reproduction; unrelated paused baselines not modified',
'historical_helpers_added':len(rows),'historical_helper_bytes':sum(x['bytes'] for x in rows),'missing_server_script_added':target.relative_to(B).as_posix(),
'deployed_server_code_files':len(server['code_files']),'deployed_files_exact':sum(x['repository_match']=='exact_bytes' for x in server['code_files']),
'deployed_files_lf_reconstructable':sum(x['repository_match']!='exact_bytes' for x in server['code_files']),
'private_local_categories_metadata_only':stats,'private_data_upload_performed':False,'large_release_assets_reuploaded':False,
'public_release_metadata_only_verified':True,'active_training_unchanged':True,
'privacy_rule':'AMASS/SMPL/BABEL arrays, source/text indices, per-trial raw traces and complete optimizer/per-source checkpoints stay private; publish aggregate evidence and prior recovery receipts only.'}
(O/'scope.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
names=subprocess.check_output(['git','-C',str(R),'ls-files','-z','--','baselines/textop']).decode('utf-8').split('\0')
files=[]
for n in names:
 if not n:continue
 p=R/n
 if p.is_file():files.append({'path':p.relative_to(B).as_posix(),'bytes':p.stat().st_size,'sha256_worktree_bytes':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'existing_public_files.json').write_text(json.dumps({'snapshot_note':'Existing tracked public files before this inventory commit. Git CRLF normalization may change checkout bytes; runtime byte bundles remain authoritative. Does not hash private datasets or live model states.','files':files},indent=2),encoding='utf-8')
report="""# TextOp GitHub同步范围与数据恢复索引

本次按用户要求清点并增量同步TextOp复现代码、配置、报告和可公开聚合证据。
仓库为公开仓库，分支codex/textop-reproduction-20261024。
同步不改变当前训练，也不启动、停止或恢复任何实验。

同步前已有1844个版本控制文件，工作树合计约42.29MB：
scripts150份、provider_tools136份、configs58份、reports45份、
results1107份、upstream264份，以及环境文档和来源记录。
这些既有文件由Git增量机制复用，不重新上传整套内容。
新增84份历史本地辅助脚本共173603字节（只归档，不执行），
并补入服务器此前漏同步的一份2688字节碰撞审计源码。
服务器198份脚本/配置全部已覆盖：195份字节相同，3份可通过CRLF转LF精确恢复。
活动代码没有变更。逐文件摘要见results/github_sync_inventory_20261027。

## 已经在GitHub的资产

已通过GitHub API核对已有Release的资产尺寸和摘要元数据，无重新下载或上传：
- textop-filter-policy29d-20261026：1000/5000/10000轮的纯推理导出，约90.62MB合计。
- textop-quick-tracking-video-20261026：SFU科研跟踪视频与致谢，约7.00MB。
- textop-mvae-initial-20261024：历史57D检查点及说明，约513.35MB，不是当前69D模型。
- inactive-engineai-environment-cache-20261025、inactive-dreamcontrol-overlay-20261026：已有软件恢复归档，未重传。

具体链接、资产名、大小、摘要见existing_release_assets.json。
已有SHA核验收据及恢复说明保留；本次仅查API元数据，不冒称重新下载全量逐字节复验。

## 数据和私有结果保存范围

依据此前确定的许可与隐私要求，下列内容不向公开仓库传输：
AMASS/SMPL/BABEL原始或重定向数组、逐条来源/文本索引、真实逐试验轨迹、
含优化器及逐动作采样状态的完整检查点。
这些保存在原服务器本人目录与D:/具身智能/TextOp复现_20261024的私有备份目录。
仓库保存构建/恢复代码、聚合数量、备份摘要和验收收据；这不等于原始数据已上传GitHub。

本次仅做文件元数据清点：D盘checkpoints共15文件/3088144491字节，
datasets共140文件/7096959038字节，diagnostics共891文件/266856995字节，
videos共244文件/28658838字节。分类可能包含公开文件的私有留存副本；
这些数字不是重新校验所有私有内容的结果，也不是一次新归档。

主要恢复证据入口：
- 预处理：PREPROCESSING_BACKUP_AUDIT.md、reports/preprocessing_complete_20261026.md与after_kit_v1验收。
- 教师完整状态：FORMAL_CHECKPOINT_BACKUP.md及10k/恢复训练相应收据；完整状态私有、纯推理权重Release。
- 快速VAE/LDM：QUICK_VAE_ENVIRONMENT.md、QUICK_LDM_ENVIRONMENT.md及完整保存/私有备份收据。
- 已暂停全源VAE8100：GENERATOR_PAUSE_MARKER_ENVIRONMENT.md与results/generator_pause_20261027/private_backup_verified.json。
- 未来15k：FILTER_RECOVERY_CHECKPOINT_ENVIRONMENT.md及新run绑定工具；尚不存在时不能计为已同步。

## VPN与增量传输

本次不重传数据集、视频、模型或数GB软件归档。只上传缺失源码、最新聚合状态和小型清单。
提交前统计新Git对象包大小；该值是内容传输估计，不是VPN账单，
TLS/协议开销、代理路由和重试可能带来额外流量。最终实核commit和包大小记录于同步收据。
""";
(B/'SYNC_SCOPE_AND_DATA_RECOVERY.md').write_text(report,encoding='utf-8')
print(json.dumps({'helpers':len(rows),'helper_bytes':scope['historical_helper_bytes'],'server_code_coverage':len(server['code_files']),'existing_tracked_files':len(files),'private_assets_uploaded':False},ensure_ascii=False))
