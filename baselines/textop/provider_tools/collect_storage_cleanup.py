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
            'server':status,'cache_cleanup':cache,'render_cleanup':{k:v for k,v in render.items() if k!='files'},
            'retained_record_compression':{k:v for k,v in compression.items() if k!='details'},'cmu_download':cmu,
            'datasets_deleted':0,'checkpoints_deleted':0,'unique_experiment_records_deleted':0,
            'full_generator_training_started':False,'physical_filter_executed':False}
    (DEST/'storage_cleanup_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (LOCAL/'空间清理与数据续传_20261025.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    total=643+(status['batch1653'].get('completed_sequences') or 0)
    report=f'''# TextOp 空间清理与数据续传

更新时间：{result['time_beijing']}。

已继续清理可恢复的重复缓存，数据集、已处理数据、检查点、唯一实验记录和运行环境均保留；PhyGile、ReactiveBFM仍保持暂停。

## 已验证的清理

- C盘：清理1,653个重复解压缓存，释放约{cache.get('previous_cache_physical_bytes',0)/1024**3:.2f}GiB。D盘完整原始压缩包逐一SHA256核验、服务器1,653份工作副本全量核验，并实际从原始包恢复一条文件。私有索引保留，可按需恢复全部缓存。恢复代码及证据已先备份至GitHub提交`{cache.get('recovery_code_and_evidence_backed_up_github_commit','')}`。
- 服务器数据盘：清理71个无人使用的着色器、派生数据和纹理缓存，实际增加约{(render.get('data_free_after',0)-render.get('data_free_before',0))/1024**3:.2f}GiB。清理名单和代码已先备份，保留所有日志、模型、环境及原始资产。
- D盘：对13个保留文件做NTFS透明压缩，物理占用减少约{compression.get('physical_bytes_saved',0)/1024**3:.2f}GiB，逐文件SHA256不变，未删除实验记录。磁盘可用量还受其他活动影响，不把物理占用变化直接当作整盘净增量。

当前C盘余{result['local_free_bytes']['C']/1024**3:.2f}GiB、D盘余{result['local_free_bytes']['D']/1024**3:.2f}GiB；服务器数据盘余{status['data_free_bytes']/1024**3:.2f}GiB、系统盘余{status['home_free_bytes']/1024**3:.2f}GiB。容量是本次采样值，下载过程中会变化。

## 数据与训练

- 已重建{total}条29DoF/69D候选运动，另保存对应镜像；新增1,653条批次当前完成{status['batch1653'].get('completed_sequences',0)}条、失败{status['batch1653'].get('failed_sequences',0)}条。**尚未进行论文要求的动态物理筛选，不是最终可训练数据集。**
- CMU官网续传状态：`{cmu.get('status','NOT_STARTED')}`。使用独立内存会话；旧D盘断点保留，复制到C盘后校验SHA256，续传前核对服务器对应边界字节。完整下载后必须通过bzip2 CRC及tar头校验。不得把当前字节数计作已完成下载。
- BMLrub、KIT仍保留断点，后续按可用空间安排；未启动其新增续传，不重复全量下载已拥有的原始包，TotalCapture继续跳过。
- 用户已报告BABEL登录完成并接受许可；当前应用内BABEL页面仍显示登录表单，独立原始标注下载尚未完成。已提供在相应页面登录或将标注包保存到本地BABEL目录的接续方式。当前候选文本来自官方TextOp发布包内BABEL派生字段。
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
