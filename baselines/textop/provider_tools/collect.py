import sys,pathlib,json,subprocess,shutil,hashlib,datetime,html,re
TASK=pathlib.Path(__file__).resolve().parent
from session import remote,REPO
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
result=json.loads(remote(rf'''
import pathlib,json,shutil,subprocess,time,re
r=pathlib.Path('{ROOT}')
names=['download_complete.json','kernel_witness.json','doc_witness_pass.json','training_progress.json','training_launch.json','training_exit.json','import_probe.json','numerical_failure.json','disk_guard_stop.json']
out={{'collected_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'receipts':{{}},'logs':{{}},'free_bytes':{{str(p):shutil.disk_usage(p).free for p in [r,pathlib.Path('/home/ouyanghao')]}}}}
for n in names:
    p=r/'receipts'/n
    if p.exists():out['receipts'][n]=json.loads(p.read_text())
for n in ['mvae_metrics.jsonl','gradient_skips.jsonl','build_attempt1_failed.log','env_finalize.log','build.log','download.log','mvae.log','gate.log']:
    p=r/'logs'/n
    if p.exists():
        if n.endswith('.jsonl'):
            raw=p.read_text()
            out['logs'][n]=raw if raw.endswith('\n') else raw.rsplit('\n',1)[0]+'\n'
        else:
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));out['logs'][n]=f.read().decode(errors='replace')
if (r/'receipts/training_failure.txt').exists():out['logs']['training_failure.txt']=(r/'receipts/training_failure.txt').read_text()
launch=out['receipts'].get('training_launch.json',{{}})
if launch.get('pid'):
    p=pathlib.Path('/proc')/str(launch['pid'])/'status'
    if p.exists():out['training_process_state']=next((x for x in p.read_text().splitlines() if x.startswith('State:')),'')
if (r/'logs/mvae.log').exists():
    clean=re.sub(r'\x1b\[[0-9;]*m','',(r/'logs/mvae.log').read_text(errors='replace'))
    counts=[[int(a),int(b)] for a,b in re.findall(r'Found (\d+) valid samples out of (\d+)',clean)]
    out['receipts']['dataset_actual_counts.json']={{'train':counts[0] if counts else None,'val':counts[1] if len(counts)>1 else None,'interpretation':'[valid windows, total sequences], official loader filter with history2 future8 primitives4'}}
for n in ['pip-freeze.txt','mvae_resolved_config.yaml']:
    p=r/'receipts'/n
    if p.exists():out['logs'][n]=p.read_text()
print(json.dumps(out))
''',timeout=50))
(BASE/'results/latest_status.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
for n,data in result['receipts'].items():(BASE/'results'/n).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
for n,data in result['logs'].items():
    p=BASE/'results/logs'/n;p.parent.mkdir(exist_ok=True);p.write_text(data,encoding='utf-8')
progress=result['receipts'].get('training_progress.json',{})
dl=result['receipts'].get('download_complete.json',{})
backup=json.loads((BASE/'results/initial_checkpoint_backup.json').read_text()) if (BASE/'results/initial_checkpoint_backup.json').exists() else None
failed=result['receipts'].get('training_exit.json',{}).get('returncode',0)!=0
status='首轮启动检查停止，已保留失败记录，正在修正重启' if failed else ('已发生真实优化器更新' if progress else ('训练进程已启动、正在装载完整数据' if 'training_launch.json' in result['receipts'] else '下载与环境验证进行中，尚未发生真实训练更新'))
if 'T (stopped)' in result.get('training_process_state',''):status='23DoF/57D基线训练现场已暂停；优先准备论文29DoF/69D数据'
report=f'''# TextOp 下载与训练启动记录

更新时间：{result['collected_utc']}。当前状态：**{status}**。

## 已固定的官方材料

- 代码：[TeleHuman/TextOp](https://github.com/TeleHuman/TextOp/tree/ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22)，提交 ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22。
- 数据与预训练权重：[Yochish/TextOp-Data](https://huggingface.co/datasets/Yochish/TextOp-Data/tree/bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9)，提交 bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9。
- 本批清单100个文件，共7,653,009,670字节：完整公开 RobotMDAR 配对数据、描述文件、官方 MVAE/DAR 与 Tracker 预训练权重。已完成校验文件数：{len(dl.get('files',[])) if dl else '等待最终汇总'}。未完成下载不能计为可用数据。
- 官方 DATASET.md：RobotMDAR共8,285序列，train6,209/val2,076，总时长105,395秒。实际加载计数以训练日志为准；短于窗口的序列由官方加载器过滤。

## 本次启动配置

先训练 MVAE，再训练依赖 MVAE 的 DAR。采用官方 USAGE.md 的公开数据命令：MVAE三个阶段100000/50000/50000，共200000次管理器步；batch512，4个连续primitive，每段history2/future8，50fps、57维特征，weighted_sample=true、use_rollout=true，seed0，Adam初始学习率1e-4并线性退火。网络为官方AutoMldVae：9层，隐藏512，前馈1024，4头，latent[1,128]，约4275万参数。损失权重保持官方配置。

服务器使用GPU1 A6000、4 CPU线程、CUDA分配上限65%。Python3.8.20、PyTorch1.13.1+cu117、NumPy1.24.4、MuJoCo3.2.3、Hydra1.3.2、OmegaConf2.3.0。新venv以只读方式继承原环境，安装单独覆盖层；未更换共享环境。完整安装规格见configs/env-spec.json。

官方文档中的action_statistics路径与发布文件位置不同，已修正为同一公开数据目录内的action_statistics.json。训练算法不变；包装器分别统计循环步数、有效优化器更新及官方跳过非有限梯度的次数，保留每次异常，增加step1000附加检查点和1GiB磁盘保护。仅20次连续跳过才保存诊断并停止。首轮在第644循环步完成643次有效更新时，启动检查错误地把官方允许的单次跳过当作致命异常；已完整归档，并修正检查逻辑后从seed0重新开始，未冒称从检查点恢复。安装时发现isaac_utils路径和旧setuptools元数据问题，均已修复并保留首次失败日志。TensorRT只用于后续部署/测速，当前训练路径未导入，暂不安装。

## 实际训练证据

```json
{json.dumps(progress,ensure_ascii=False,indent=2) if progress else '尚无实际梯度更新记录；请勿将环境检查当作训练结果。'}
```

环境见证仅用固定随机种子与小批合成输入验证官方MVAE前向/反向，已得到[2,8,57]输出及有限梯度。这不构成论文训练或生成质量评价。训练进程启动记录及逐步指标见results/。

实际数据加载计数（[有效序列数, 原始序列数]）：{json.dumps(result['receipts'].get('dataset_actual_counts.json',{}),ensure_ascii=False)}。

固定HF版本实测train6,213/val2,078，总计8,291，与官方DATASET.md的6,209/2,076相差6段（train+4、val+2）。本次保留该官方发布版本的完整数据，未删减成文档数字；文档总时长105,395秒未当作本版本的实测时长。

首个检查点备份：{('step1000已保存在服务器、本地checkpoints文件夹及[GitHub预发布]('+backup['github_release']+')，三端SHA256一致：'+backup['sha256']) if backup else '等待step1000检查点与三端哈希校验；当前梯度更新记录不等同于检查点已完成备份。'}。

后续检查点采用后台自动归档：逐个验证模型/优化器有限性、服务器和本地SHA256、GitHub资产摘要；只有本地与GitHub均验证一致后，才释放该版本的旧服务器副本，服务器保留本轮最新可恢复检查点。所有版本保留在本地和GitHub，数据集、官方权重与其他实验不进入清理范围。归档清单见results/checkpoint_sync_state.json。同步依赖本地电脑保持联网；连接或空间不足时保留服务器文件，训练仍受1GiB磁盘保护约束。

## 复现范围与后续

本阶段完成公开配对数据的下载验证并启动完整公开数据MVAE训练。DAR、Tracker再训练、生成—执行闭环、成功率评价和结果视频尚未完成。官方预训练模型使用包含未公开私有数据的训练混合，不能宣称与作者完整训练数据完全相同；本次从头训练使用公开BABEL统计量，预训练推理必须用作者PRIVATE-DATA统计量。

官方README还注明部分最新代码和数据尚未完全更新，因此本轮对齐的是上述固定公开版本，不能宣称已经覆盖最新论文的所有实现细节。

2026-10-25补充核对：论文表VI为69维、附录A为29DoF和0.2m接触高度阈值；当前发布数据/旧处理脚本为23DoF、57维和0.08m。此前“官方配置”仅指固定公开代码的配置，并非已经验证与论文最终版完全一致。当前保留该实验，暂停参数更新，转为先准备29DoF论文数据；详细状态见数据准备报告。

Tracker官方9780段训练动作另需9,900,102,760字节，尚未全量下载；已保存完整文件名、大小和哈希清单，本次下载包含其预训练权重。完整Tracker需要Isaac Lab2.1.0和官方rsl_rl分支，不能用当前MVAE环境代替。ROS2/Unitree MuJoCo实时部署需后续独立配置。

原始/重定向数据及模型大文件留在服务器个人目录；GitHub保存代码、固定版本、哈希、配置和结果证据，避免公开受限AMASS/SMPL资产。数据目录分布在/data/code/ouyanghao/oyh/textop-reproduction-20261024/assets与/home/ouyanghao/textop-assets-20261024，由符号链接提供统一官方目录。未删除已有数据。PhyGile和ReactiveBFM保持暂停。
'''
(BASE/'reports/启动报告_20261024.md').write_text(report,encoding='utf-8')
(BASE/'README.md').write_text('# TextOp 复现\n\n'+status+'。\n\n优先查看 [数据准备核查与重建报告](reports/dataset_preparation_20261025.md)。另见 [启动报告](reports/启动报告_20261024.md)、[环境说明](ENVIRONMENT.md)、[官方版本](upstream/README.md)。\n\n官方代码以固定提交镜像保存于upstream/，下载数据不入Git。scripts/为本次下载、隔离环境安装、训练启动与指标记录代码。官方算法源文件未修改；论文29DoF重建部分为独立自实现并复用GMR核心，尚待真实动态筛选。\n',encoding='utf-8')
for p in [BASE/'reports/启动报告_20261024.md',BASE/'results/latest_status.json',BASE/'ENVIRONMENT.md',BASE/'SERVER_NOTES.md']:shutil.copy2(p,LOCAL/p.name)
def render_inline(t):
    t=html.escape(t)
    t=re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',r'<a href="\2">\1</a>',t)
    t=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',t)
    return t
parts=[];in_code=False;in_list=False
for line in report.splitlines():
    if line.startswith('```'):
        if in_list:parts.append('</ul>');in_list=False
        parts.append('</pre>' if in_code else '<pre>');in_code=not in_code;continue
    if in_code:parts.append(html.escape(line)+'\n');continue
    if line.startswith('- '):
        if not in_list:parts.append('<ul>');in_list=True
        parts.append('<li>'+render_inline(line[2:])+'</li>');continue
    if in_list:parts.append('</ul>');in_list=False
    m=re.match(r'^(#{1,3}) (.+)',line)
    if m:parts.append(f'<h{len(m[1])}>'+render_inline(m[2])+f'</h{len(m[1])}>')
    elif line.strip():parts.append('<p>'+render_inline(line)+'</p>')
if in_list:parts.append('</ul>')
page='<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp 下载与训练启动</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1040px;margin:36px auto;padding:24px 40px;background:#f6f8fa;color:#17202a}h1{font-size:30px}h2{font-size:21px;margin-top:36px;border-top:1px solid #d0d7de;padding-top:24px}a{color:#0969da;overflow-wrap:anywhere}p,li{overflow-wrap:anywhere}pre{font-size:13px;line-height:1.6;white-space:pre-wrap;background:white;padding:20px;border:1px solid #d0d7de;border-radius:10px;overflow-wrap:anywhere}strong{color:#095b45}li{margin:8px 0}</style><main><p><a href="数据准备核查报告_20261025.html">查看最新数据准备核查报告</a></p>'+''.join(parts)+'</main></html>'
(LOCAL/'TextOp下载与训练报告.html').write_text(page,encoding='utf-8')
print(status,json.dumps(progress,ensure_ascii=False))
