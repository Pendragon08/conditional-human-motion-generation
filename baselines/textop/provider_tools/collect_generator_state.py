"""Collect aggregate bounded state-diagnostic evidence; private checkpoints stay put."""
import datetime,html,json,pathlib
from session import REPO,remote
b=REPO/'baselines/textop';local=pathlib.Path('D:/具身智能/TextOp复现_20261024')
dest=b/'results/generator_state_20261025';dest.mkdir(parents=True,exist_ok=True)
files=json.loads(remote(r'''
import pathlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');files={}
for run in ['generator_state_author_v1','generator_state_document_v1']:
 for folder,leaf in [('generator_validation/'+run,'receipt.json'),('receipts','generator_state_'+run+'_exit.json'),('logs','generator_state_'+run+'.log')]:
  p=r/folder/leaf
  if p.exists():files[run+'/'+leaf]=p.read_text(errors='replace')
 for stage in ['vae','ldm']:
  for name in ['before_resume_metrics.jsonl','after_resume_metrics.jsonl']:
   p=r/'generator_validation'/run/stage/name
   if p.exists():files[run+'/'+stage+'/'+name]=p.read_text()
print(json.dumps(files))
''',timeout=90))
for name,value in files.items():
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8')
independent=REPO.parent.parent/'textop_reproduction_20261024/generator_state_doc_witness_v1/REPORT.md'
if independent.exists():(dest/'independent_REPORT.md').write_text(independent.read_text(encoding='utf-8'),encoding='utf-8')
def read(name):
 p=dest/name;return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
author=read('generator_state_author_v1/receipt.json');doc=read('generator_state_document_v1/receipt.json')
proofs={run:read(run+'/generator_state_'+run+'_exit.json') for run in ['generator_state_author_v1','generator_state_document_v1']}
passed=all(x.get('child_exit_code')==0 and not x.get('timed_out',True) and x.get('numerical_status')=='TEXTOP_GENERATOR_STATE_WITNESS_PASSED' for x in proofs.values())
summary={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'author':author,'independent_document':doc,'exit_proofs':proofs,'both_processes_passed':passed,
 'dataset_accepted':False,'physical_filter_executed':False,'full_generator_training_started':False}
(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
stages=author.get('stages',{})
rows='\n'.join('| '+name+' | '+str(value.get('actual_optimizer_updates',0))+' | '+value.get('optimizer','?')+' | '+f"{value.get('peak_torch_allocated_bytes',0)/1024**3:.3f}GiB"+' | '+str(value.get('checkpoint',{}).get('bytes',0))+' |' for name,value in stages.items())
report=f'''# TextOp 69D 优化器与断点恢复验证

更新时间：{summary['time_beijing']}。**仅完成未过滤候选数据上的短状态诊断，完整VAE/LDM训练尚未开始。**

作者状态：`{author.get('status','PENDING')}`。独立按文档状态：`{doc.get('status','PENDING')}`。两次真实子进程退出均通过：{passed}。实际退出、超时、资源采样及逐次损失保留在公开聚合证据中；私有检查点留在服务器，没有公开训练动作或文本。

| 阶段 | 本次实际更新数 | 优化器 | 作者Torch峰值分配 | 保存的检查点字节数 |
| --- | ---: | --- | ---: | ---: |
{rows}

使用固定2,296源的候选视图、真实CLIP缓存与候选训练集统计，batch512、history2/future8、每批4个primitive，VAE9层、LDM8层。每阶段先做4次真实优化，落盘检查点，再创建新的模型/管理器恢复，完成原本待执行的4个验证片段，然后继续4次优化；总计每次运行16次优化。验证不增加优化器计数。

实际核对模型、Adam/AdamW、EMA、manager步数/真实更新/跳过数、Python/NumPy/Torch/CUDA随机状态与显式批次生成器，以及待评估字段。恢复后下一批数据摘要及随机探针一致；再次恢复后才执行真实验证/续训。特征维度绑定错误的检查点被拒绝。LDM冻结4步诊断VAE，冻结参数保持不变，不能据此宣称训练好了潜空间。

EMA仅用于这次状态压力诊断，固定官方版本正式配置仍为关闭；验证间隔4与4个验证片段也仅为诊断设置。正式训练维持论文/官方配置并如实记录差异。只保存每阶段第4步检查点，运行到第8步用于验证续训更新；没有产出正式生成器检查点。此检查覆盖第一阶段状态恢复，后期自回归课程、最终合格语料和全程收敛仍需实际验证。

与先前零更新数值检查不同，本次真正分配了优化器和EMA状态并完成磁盘保存/读取。显存为本进程Torch峰值，不是整个共享GPU占用；不等于最终全语料训练一定通过。每次运行约0.99GB检查点留在个人目录，启动前要求额外预算并保留至少2GiB空间，没有清理权威数据、旧检查点或其他用户文件。

现有GMR、私有备份和参考导出串行队列保持运行。全部可得动作处理及备份验收、真实宽松/严格筛选、最终语料/统计/CLIP验收仍是完整VAE200k→LDM300k和闭环的前置条件。
'''
(b/'reports/generator_state_20261025.md').write_text(report,encoding='utf-8')
(local/'69D优化器与断点恢复验证_20261025.md').write_text(report,encoding='utf-8')
(local/'69D优化器与断点恢复验证_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>TextOp断点恢复验证</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.8 system-ui}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
print(json.dumps({'author':author.get('status'),'document':doc.get('status'),'both_processes_passed':passed,'full_generator_training_started':False}))
