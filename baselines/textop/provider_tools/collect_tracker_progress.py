"""Collect only aggregate diagnostic evidence, with explicit training limits."""
import datetime,html,json,pathlib,shutil
from session import REPO,remote
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
DEST=BASE/'results/tracker_backend_20261025';DEST.mkdir(parents=True,exist_ok=True)
code=r'''
import pathlib,json,subprocess,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');files={}
for label in ['author_v1','author_v2','author_v3','author_v4','author_v5','document_v1','document_v2','environment_probe_v1','ppo_author_v1','ppo_author_v2','ppo_document_v1',
              'capacity_author_v1','capacity_document_v1','initialization_author_v1','initialization_document_v1',
              'filter_statistics_author_v1','filter_statistics_author_v2','filter_statistics_document_v1']:
 for leaf in ['receipt.json','audit_correction.json','native_teardown_stack.txt']:
  p=r/'tracker_validation'/label/leaf
  if p.exists():files[label+'/'+leaf]=p.read_text(errors='replace')
 for prefix in ['tracker_kernel_','tracker_job_','tracker_capacity_','tracker_initialization_','filter_policy_']:
  p=r/'receipts'/(prefix+label+'_exit.json')
  if p.exists():files[label+'/supervisor_'+prefix+'.json']=p.read_text()
  p=r/'logs'/(prefix+label+'.log')
  if p.exists():files[label+'/'+prefix+'.log']=p.read_text(errors='replace')
for label in ['export_author_v1','export643_v1','export1653_v1']:
 p=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/label/'receipt.json'
 if p.exists():files[label+'/receipt.json']=p.read_text()
 p=r/'receipts'/('tracker_job_'+label+'_exit.json')
 if p.exists():files[label+'/supervisor.json']=p.read_text()
 p=r/'logs'/('tracker_job_'+label+'.log')
 if p.exists():files[label+'/export.log']=p.read_text(errors='replace')
for leaf in ['reference643_dynamics_audit.json','filter_statistics_author_v1_failure_stop.json']:
 p=r/'receipts'/leaf
 if p.exists():files['additional/'+leaf]=p.read_text()
print(json.dumps({'files':files,'space':{x:shutil.disk_usage(x).free for x in ['/data','/home/ouyanghao','/dev/shm']},
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader'],text=True)}))
'''
data=json.loads(remote(code,timeout=90))
for name,value in data.pop('files').items():
    p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8')
TASK=REPO.parent.parent/'textop_reproduction_20261024'
for folder in ['tracker_doc_witness_v1','tracker_doc_witness_v2','tracker_ppo_doc_witness_v1','tracker_capacity_doc_witness_v1','tracker_initialization_doc_witness_v1','filter_statistics_doc_witness_v1']:
    p=TASK/folder/'REPORT.md'
    if p.exists():
        target=DEST/folder/'REPORT.md';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
def read(label):
    p=DEST/label/'receipt.json';return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
latest=json.loads((BASE/'results/data_preparation_20261025/latest.json').read_text(encoding='utf-8'))
batches=[latest['receipts'].get(k,{}) for k in ['batch500','batch143','batch1653','batchCMU','batch_bmlrub','batch_kit']]
prepared=sum(x.get('completed_sequences',0) for x in batches)
ppo=read('ppo_author_v2');independent=read('ppo_document_v1');export=read('export643_v1')
capacity=read('capacity_author_v1');capacity_doc=read('capacity_document_v1')
initialization=read('initialization_author_v1');statistics=read('filter_statistics_author_v2');statistics_doc=read('filter_statistics_document_v1')
export1653=read('export1653_v1')
backup_path=BASE/'results/data_preparation_20261025/reference643_backup.json'
backup=json.loads(backup_path.read_text(encoding='utf-8')) if backup_path.exists() else {}
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
summary={'time_beijing':now,'prepared_source_sequences':prepared,'physical_filter_executed':False,
         'full_generator_training_started':False,'backend_author_passed':read('author_v5').get('status'),
         'backend_document_passed':read('document_v2').get('status'),'actual_mdp_probe':read('environment_probe_v1').get('status'),
         'ppo_author':ppo,'ppo_document':independent,'reference_export643':export,'reference_export643_backup':backup,'resources':data,
         'capacity643_author':capacity,'capacity643_document':capacity_doc,'initialization_audit':initialization,
         'filter_statistics_author':statistics,'filter_statistics_document':statistics_doc,'reference_export1653':export1653,
         'unresolved':['8192-env memory/stability with ALL future references; current643-source capacity alone passed','full40k policy and two-round physical filtering',
                       'accepted corpus loader/statistics/text cache','fullVAE200k and LDM300k','native SDK stop/close hang; bounded diagnostic fallback disclosed']}
(DEST/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(LOCAL/'跟踪筛选器集成验证_20261025.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
report=f'''# TextOp 数据筛选器集成与生成器训练进度

更新时间：{now}。

**完整69D生成器尚未启动：现有候选动作必须先通过真实物理筛选。** 用户已授权数据验收后继续完整训练，无需再次确认是否训练。

## 已验证的实际进展

- 29DoF/69D源运动已重建{prepared}条，另存镜像；原始BABEL的8,291条运动、50,004条标签已核对。所有新候选仍是physical_filter_passed=false。
- 真实IsaacSim4.5.0 / IsaacLab项目2.1.0（包标识0.36.21）运行通过。29关节、合并固定关节后的30个刚体使用名称映射，不混用GMR的38个框架顺序。
- 作者与独立按文档执行的底层FK/物理诊断均退出0。实际跟踪MDP也完成8环境、24次控制调用，观测/奖励有限；actor和critic均557维，无域随机化，预训练期间移除高度终止项，严格阶段恢复骨盆0.25m、末端0.5m阈值。
- 真实PPO更新及断点恢复：作者状态`{ppo.get('status','PENDING')}`，实际Adam更新{ppo.get('optimizer_updates',0)}次；模型、Adam、两套观测归一化的数值/类型/形状逐一核对，恢复后继续更新。独立按文档复核状态`{independent.get('status','PENDING')}`。这是8环境、2轮的功能诊断，**不是论文8192环境、40,000轮的过滤训练，更不是动作通过率**。
- 首批643条源动作及镜像的完整参考导出状态`{export.get('status','PENDING')}`：完成{export.get('exported_source_sequences',0)}/643条源动作、{export.get('exported_trajectories',0)}条轨迹。实际Isaac FK、50Hz速度、WXYZ姿态均保存；导出不前进动力学，也不代表物理通过。私有持久备份状态`{backup.get('status','NOT_STARTED')}`，备份同时核查KIT剩余下载量和D盘6GiB余量，不删除服务器工作副本。
- 8,192环境容量：作者`{capacity.get('status','PENDING')}`、独立文档执行`{capacity_doc.get('status','PENDING')}`；各完成196,608次环境转换和20次Adam更新，Torch峰值分配约{capacity.get('peak_torch_allocated_bytes',0)/1024**3:.2f}GiB。范围为643条源动作及镜像，不等于全部未来运动库容量已通过。随机初始策略的价值损失约409万、接触力尖峰约111万N均保留，不能视为稳定收敛。
- 独立初始化核查已完成：1,286条轨迹各从第0帧开始，根位置最大误差约1.19×10⁻⁷m，466个初始姿态触及软关节裁剪；20步直接参考PD出现37次终止和接触冲击。说明根坐标映射吻合，但没有证明动作可由训练后控制器稳定跟踪。诊断零重置扰动属于对照设置，未替换正式训练的官方重置分布。
- 新增筛选统计与恢复层：作者`{statistics.get('status','PENDING')}`、独立`{statistics_doc.get('status','PENDING')}`，40次真实更新后逐项恢复模型、Adam、两套归一化、EMA失败统计及独立真实尝试计数；初始重置不计为真实成功。正式40,000轮训练入口已实现，但当前仍被数据与完整库验收条件禁用。
- 新1,653条源动作已完成重定向并逐条SHA校验私有备份，CMU1,330条可用源动作已接续处理。其仿真参考导出当前状态`{export1653.get('status','PENDING')}`，已完成{export1653.get('exported_source_sequences',0)}/1,653条；这是参考构建，没有实施动态筛选。

## 保留的失败与差异

SDK原生stop/close卡住；日志、超时和栈均保留。仅在诊断结束、检查点/结果落盘、输出写入关闭后使用20秒有界退出，进程实际退出由外层监督程序核验。native_close_returned=false，不能写成原生退出问题已彻底修好。

PPO首轮已做20次Adam更新，之后比较Adam标量的CPU/GPU设备时报错。原始失败记录和其未及时刷新的0计数均保留，并附audit_correction.json；修正比较设备后，后续诊断完整通过40次更新。没有修改优化器训练目标来绕过错误。

作者机器人资产哈希、最终69D几何损失刚体子集、私有403条运动未公开，不能声称与私有实现完全等价。CMU所选SMPL-X G导出缺13条索引，逐项清单保留；BMLrub、KIT已完成原始包CRC/tar/SHA校验及索引核对，分别2,139/2,139、2,513/2,513条精确匹配，零歧义。可用公开配对源范围是8,278/8,291条。它与论文跟踪器的12,296条公共运动及403条私有运动范围不同，不能把当前BABEL索引子集写成作者全量数据。

新增统计恢复层首轮在20次更新后，写回由官方推断模式创建的EMA张量时报错。原始失败、检查点和仅停止该失败诊断进程的退出记录全部保留；恢复函数显式进入推断模式后，作者与独立执行均完整通过40次更新。恢复参数和统计的数值精确一致，物理仿真回合在进程恢复后重新开始，不声称仿真轨迹位级连续。

## 已安排的后续

1. 现有原始包下载已核验完成；GMR队列当前为CMU1330条 → BMLrub2139条 → KIT2513条，前一批持久备份成功后才进入下一批；不重复备份、不并发抢占GMR资源。
2. 完成所有可获得候选的参考导出，检查8192环境及完整运动库的显存容量，启动论文40,000轮特权过滤策略。保留失败记录，再执行宽松失败概率>0.05剔除及严格任何失败剔除。
3. 从真实保留动作构建训练/验证分离语料，仅用训练集统计归一化并补全CLIP文本缓存。加载和断点恢复验证通过后，新建29DoF/69D训练：VAE200,000步 → 冻结VAE的LDM300,000步；batch512、lr1e-4、历史2帧→未来8帧及论文网络配置。
4. 生成器训练完成后还需训练最终非特权431D actor/557D critic全身控制器（仿真实验50,000轮），加入生成运动增强，再验证文本切换、长时自回归和生成—执行闭环，保存结果视频、失败案例和报告。特权筛选策略不是最终部署控制器。

已安排后台持续推进。PhyGile、ReactiveBFM及旧57D TextOp继续暂停；权威数据集、检查点和唯一记录保留。受限原始/重定向资产及逐条文本索引保持私有，GitHub同步代码、哈希、聚合结果和报告。

详见[数据准备核查](dataset_preparation_20261025.md)、[论文Step2配置](step2_motion_generator_20261025.md)、[物理/PPO原始证据](../results/tracker_backend_20261025/summary.json)。
'''
(BASE/'reports/tracker_integration_20261025.md').write_text(report,encoding='utf-8')
(LOCAL/'跟踪筛选器集成验证_20261025.md').write_text(report,encoding='utf-8')
page='<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp数据与完整训练进度</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1000px;margin:30px auto;padding:24px;background:#f6f8fa}pre{white-space:pre-wrap;background:white;padding:24px;border:1px solid #ccc;font:inherit}a{color:#0969da}</style><h1>TextOp数据准备与完整训练进度</h1><p><a href="数据准备核查报告_20261025.html">数据准备</a> · <a href="Step2运动生成器训练安排_20261025.html">训练配置</a> · <a href="空间清理与数据续传_20261025.html">存储与下载</a></p><pre>'+html.escape(report)+'</pre></html>'
for name in ['跟踪筛选器集成验证_20261025.html','TextOp下载与训练报告.html']:(LOCAL/name).write_text(page,encoding='utf-8')
print(json.dumps({'prepared_sources':prepared,'ppo_actual_updates':ppo.get('optimizer_updates'),'ppo_document':independent.get('status'),'exported_sources':export.get('exported_source_sequences'),'full_generator_started':False}))
