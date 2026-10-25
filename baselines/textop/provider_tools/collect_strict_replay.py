"""Collect only public strict-replay diagnostics and aggregate queue state."""
import datetime,html,json,pathlib,shutil
from session import REPO,remote
b=REPO/'baselines/textop';local=pathlib.Path('D:/具身智能/TextOp复现_20261024')
dest=b/'results/strict_replay_20261025';dest.mkdir(parents=True,exist_ok=True)
code=r'''
import pathlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');files={}
for label in ['strict_replay_author_v1','strict_replay_author_v2','strict_replay_author_v3','strict_replay_document_v1','strict_replay_document_v2']:
 for leaf in ['receipt.json','native_teardown_stack.txt']:
  p=r/'tracker_validation'/label/leaf
  if p.exists():files[label+'/'+leaf]=p.read_text(errors='replace')
 for folder,leaf in [('receipts','strict_replay_'+label+'_exit.json'),('logs','strict_replay_'+label+'.log')]:
  p=r/folder/leaf
  if p.exists():files[label+'/'+leaf]=p.read_text(errors='replace')
p=r/'receipts/reference_export_queue_v1.json'
if p.exists():files['reference_export_queue.json']=p.read_text()
print(json.dumps(files))
'''
files=json.loads(remote(code,timeout=60))
for name,value in files.items():
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8')
task=REPO.parent.parent/'textop_reproduction_20261024'
for label in ['strict_replay_doc_witness_v1','strict_replay_doc_witness_v2']:
 p=task/label/'REPORT.md'
 if p.exists():(dest/(label+'.md')).write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
def read(name):
 p=dest/name;return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
author=read('strict_replay_author_v3/receipt.json');document=read('strict_replay_document_v2/receipt.json')
summary={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'author':author,'independent_document':document,'reference_queue':read('reference_export_queue.json'),
 'physical_filter_executed':False,'accepted_dataset_created':False,'full_generator_training_started':False}
(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
report=f'''# TextOp 严格回放执行器验证

更新时间：{summary['time_beijing']}。

已实现实际IsaacSim动力学回放、逐动作首次失败或整段覆盖记录、40,000轮策略凭据与EMA统计绑定。**正式两轮数据筛选仍未执行，完整生成器仍未启动。** 本轮使用的检查点只有2轮训练，用于验证执行器运行与计数，不能评价最终跟踪效果。

- 作者状态：{author.get('status','PENDING')}；独立按文档执行：{document.get('status','PENDING')}。最新固定源码及规格哈希在原始凭据中保存，外层记录真实进程退出。
- 两条85/90帧原动作的回放均失败，完整原动作通过数为0。作者实际{author.get('actual_vectorized_control_calls',0)}次向量控制调用、{author.get('actual_scene_physics_steps',0)}次场景物理步，0次参数更新。
- 另外使用明确缩短为2帧的独立控制流夹具，执行1步真实仿真，验证SDK自然结束信号；该夹具不加入完整原动作通过数，原始参考数组和原动作失败记录不变。完整原动作的自然完成结果仍待真实训练后的策略验证。
- 修复首步高度误判：官方命令的相对参考缓冲区在重置后仍为零值或旧值，直到终止判定后的命令更新才刷新。现用同一官方变换在重置后刷新，未推进运动时间、改变EMA或物理状态、放宽阈值。修复前作者首轮2条均首步终止，源代码及原始记录已保留；修复后初始跟踪刚体最大误差约0.02697m，原动作分别触发真实高度终止。
- 严格骨盆高度阈值0.25m、末端0.5m、姿态终止均保留。默认10秒终止替换为精确运动结束，失败后新回合不拼接到原回合。确定性第0帧、无重置扰动和单次重复属于已声明的实现选择。
- SDK原生关闭仍卡住；结果落盘、写入关闭后使用20秒有界退出。实际退出0与原生关闭成功明确区分，原始栈与资源采样保留。

新增参考导出串行工作者状态：{summary['reference_queue'].get('status','PENDING')}，当前等待{summary['reference_queue'].get('cohort','UNKNOWN')}。它在每批GMR实际完成且资源充足后，接续CMU→BMLrub→KIT的参考导出；不重复GMR/备份，不启动训练，不删除数据。参考导出不等于物理筛选。

下一步：完成全部可用候选和参考库、全库资源与稳定性验证、真实40,000轮特权策略，再执行实际两轮过滤；最终语料验收通过后自动接续VAE20万→LDM30万和闭环验证。受限逐动作数据、标签和私有日志保留在个人目录，GitHub仅发布聚合证据、代码和报告。
'''
(b/'reports/strict_replay_20261025.md').write_text(report,encoding='utf-8')
(local/'严格回放执行器验证_20261025.md').write_text(report,encoding='utf-8')
(local/'严格回放执行器验证_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>TextOp严格回放执行器验证</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.8 system-ui}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
print(json.dumps({'author':author.get('status'),'document':document.get('status'),'full_original_passes':0,'physical_filter_executed':False}))
