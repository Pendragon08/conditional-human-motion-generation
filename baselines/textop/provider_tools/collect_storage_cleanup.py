"""Collect cleanup/durability proof and current data progress for user reports."""
import datetime,html,json,pathlib,shutil,sys
from session import REPO,remote
BASE=REPO/'baselines/textop';DEST=BASE/'results/data_preparation_20261025'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
sys.stdout.reconfigure(encoding='utf-8')

def read(name):
    p=DEST/(name+'.json');return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

def main():
    cache=read('redundant_cache_cleanup');render=read('render_cache_cleanup');compression=read('retained_record_compression');cmu=read('cmu_resume')
    duplicates=read('duplicate_copy_cleanup');bmlrub=read('bmlrub_resume');kit=read('kit_resume')
    personal_root=BASE/'results/storage_cleanup_20261025'
    personal=json.loads((personal_root/'cleanup.json').read_text(encoding='utf-8')) if (personal_root/'cleanup.json').exists() else {}
    relocated=json.loads((personal_root/'relocation.json').read_text(encoding='utf-8')) if (personal_root/'relocation.json').exists() else {}
    babel=read('babel_alignment');babel_install=read('babel_server_install')
    babel_note=('BABEL官方原始标注已下载并安装服务器，8,291条发布运动、50,004条分段标注的ID、来源路径、文本、类别和起止时间核对一致；原始JSON及逐条配对索引不公开上传GitHub。' if babel.get('mismatch_sequences')==0 and babel_install else 'BABEL原始标注独立下载或核查尚未完成；当前候选文本来自官方发布包内字段。')
    status=json.loads(remote("""
import pathlib,json,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');ram=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/local1653_v1')
p=ram/'prepared1653_v1/progress.json'
j=json.loads(p.read_text()) if p.exists() else {}
pid=3319567;ps=pathlib.Path('/proc')/str(pid)/'status'
print(json.dumps({'data_free_bytes':shutil.disk_usage(r).free,'home_free_bytes':shutil.disk_usage('/home/ouyanghao').free,'batch1653':{k:j.get(k) for k in ['completed_sequences','failed_sequences','status']},'gmr_process_exists':ps.exists()}))
""",timeout=50))
    result={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'local_free_bytes':{x:shutil.disk_usage(x+':/').free for x in ['C','D']},
            'server':status,'cache_cleanup':cache,'private_recovery_kit':read('cache_residency'),'render_cleanup':{k:v for k,v in render.items() if k!='files'},
            'retained_record_compression':{k:v for k,v in compression.items() if k!='details'},'cmu_download':cmu,
            'verified_duplicate_cleanup':duplicates,'bmlrub_download':bmlrub,'kit_download':kit,
            'personal_idle_environment_cleanup':personal,'personal_system_assets_relocation':relocated,
            'datasets_deleted':0,'checkpoints_deleted':0,'unique_experiment_records_deleted':0,
            'full_generator_training_started':False,'physical_filter_executed':False}
    (DEST/'storage_cleanup_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (LOCAL/'空间清理与数据续传_20261025.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    total=643+(status['batch1653'].get('completed_sequences') or 0)
    report=f'''# TextOp 空间清理与数据续传

更新时间：{result['time_beijing']}。

已继续清理可恢复的重复缓存，数据集、已处理数据、检查点、唯一实验记录和运行环境均保留；PhyGile、ReactiveBFM仍保持暂停。

## 已验证的清理

- C盘：清理1,653个重复解压缓存，释放约{cache.get('previous_cache_physical_bytes',0)/1024**3:.2f}GiB。D盘完整原始压缩包逐一SHA256核验、服务器1,653份工作副本全量核验，并实际从原始包恢复一条文件。私有索引在C、D盘均保存并核对SHA256，可按需恢复全部缓存；D盘恢复工具包位于`datasets/recovery1653_private`，逐条索引未上传GitHub。恢复代码及证据已先备份至GitHub提交`{cache.get('recovery_code_and_evidence_backed_up_github_commit','')}`。
- 服务器数据盘：清理71个无人使用的着色器、派生数据和纹理缓存，实际增加约{(render.get('data_free_after',0)-render.get('data_free_before',0))/1024**3:.2f}GiB。清理名单和代码已先备份，保留所有日志、模型、环境及原始资产。
- D盘：对13个保留文件做NTFS透明压缩，物理占用减少约{compression.get('physical_bytes_saved',0)/1024**3:.2f}GiB，逐文件SHA256不变，未删除实验记录。磁盘可用量还受其他活动影响，不把物理占用变化直接当作整盘净增量。
- D盘重复副本：完整CMU原始包CRC/SHA验证后清理其旧断点副本；CLoSD服务器原始数据集核验不变，四个本地备份分片和GitHub Release均保留，只清理此前临时恢复验证副本。恢复证据先保存到GitHub；完整记录见duplicate_copy_cleanup.json。所有权威数据、检查点和唯一实验记录保留。
- 本轮个人目录清理：闲置EngineAI Python环境及指定旧缓存状态`{personal.get('status','BACKUP_OR_VERIFICATION_PENDING')}`；系统盘资源迁移状态`{relocated.get('status','PENDING')}`。完整软件/缓存归档在GitHub Release逐片摘要核验、代码及清单备份后才删除原副本；数据和模型资源迁移后原路径保留为链接。详见[个人目录空间清理核验](personal_storage_cleanup_20261025.md)及[EngineAI旧环境恢复说明](../ENGINEAI_ENV_RECOVERY.md)。当前TextOp/GMR环境与共享IsaacSim平台保留。迁移不计作两盘整体新增空间。

当前C盘余{result['local_free_bytes']['C']/1024**3:.2f}GiB、D盘余{result['local_free_bytes']['D']/1024**3:.2f}GiB；服务器数据盘余{status['data_free_bytes']/1024**3:.2f}GiB、系统盘余{status['home_free_bytes']/1024**3:.2f}GiB。容量是本次采样值，下载过程中会变化。

## 数据与训练

- 已重建{total}条29DoF/69D候选运动，另保存对应镜像；新增1,653条批次当前完成{status['batch1653'].get('completed_sequences',0)}条、失败{status['batch1653'].get('failed_sequences',0)}条。**尚未进行论文要求的动态物理筛选，不是最终可训练数据集。**
- CMU下载状态：`{cmu.get('status','NOT_STARTED')}`。完整原始包已通过bzip2 CRC/tar校验，旧D盘断点在完整包校验和GitHub恢复证据备份后清理。选定导出覆盖1,330条所需输入，另13条缺失未冒充完成。
- BMLrub状态`{bmlrub.get('status','NOT_STARTED')}`；KIT状态`{kit.get('status','NOT_STARTED')}`。两项已按授权启动官网续传，分别保留所在盘6GiB余量；完成后必须CRC/tar验证。TotalCapture继续跳过。
- {babel_note}
- 69D生成器核心的VAE/LDM前向反向、条件输入、五步扩散和历史2→未来8帧自回归数值验证均通过；这是未训练的数值验证，0次参数更新，不代表动作质量或闭环成功。
- 完整训练仍需补齐原始输入/标注、训练特权跟踪策略并执行宽松和严格两轮物理筛选、完成最终语料归一化与加载集成。随后按论文VAE20万、LDM30万步配置训练。当前未启动完整69D生成器，也未恢复旧57D训练。

原始动作、重定向输出、逐条文本和人体模型受许可限制，保存在个人持久目录；GitHub仅同步代码、恢复工具、哈希、聚合结果和报告。服务器RAM仅用于可从持久原始包重建的工作副本，处理后结果由现有队列归档到D盘并校验。
'''
    (BASE/'reports/storage_cleanup_20261025.md').write_text(report,encoding='utf-8')
    (LOCAL/'空间清理与数据续传_20261025.md').write_text(report,encoding='utf-8')
    page='<!DOCTYPE html><meta charset="utf-8"><title>TextOp空间与数据进度</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1000px;margin:30px auto;padding:24px;background:#f6f8fa}pre{white-space:pre-wrap;background:white;padding:24px;border:1px solid #ccc;font:inherit}a{color:#0969da}</style><h1>TextOp空间清理与数据续传</h1><p><a href="数据准备核查报告_20261025.html">数据准备记录</a> · <a href="69D核心适配与数值验证_20261025.html">69D核心验证</a></p><pre>'+html.escape(report)+'</pre>'
    (LOCAL/'空间清理与数据续传_20261025.html').write_text(page,encoding='utf-8')
    print(json.dumps({'candidate_sequences':total,'new_batch':status['batch1653'],'cmu':cmu.get('status'),'free_bytes':result['local_free_bytes'],'server_free_bytes':status['data_free_bytes']}),flush=True)

if __name__=='__main__':main()
