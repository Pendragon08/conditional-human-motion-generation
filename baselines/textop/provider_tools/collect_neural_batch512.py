"""Public zero-update neural diagnostics only; no private arrays or model files."""
import datetime,html,json,pathlib
from session import REPO,remote

b=REPO/'baselines/textop'
local=pathlib.Path('D:/具身智能/TextOp复现_20261024')
dest=b/'results/neural_batch512_20261025';dest.mkdir(parents=True,exist_ok=True)
files=json.loads(remote(r'''
import pathlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');files={}
for run in ['neural512_author_v1','neural512_document_v1']:
 for folder,leaf in [('generator_validation/'+run,'receipt.json'),('receipts','neural_batch512_'+run+'_exit.json'),('logs','neural_batch512_'+run+'.log')]:
  p=r/folder/leaf
  if p.exists():files[run+'/'+leaf]=p.read_text(errors='replace')
print(json.dumps(files))
''',timeout=90))
for name,value in files.items():
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8')
task=REPO.parent.parent/'textop_reproduction_20261024'
p=task/'neural_batch512_doc_witness_v1/REPORT.md'
if p.exists():(dest/'independent_REPORT.md').write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
def read(name):
 p=dest/name
 return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
author=read('neural512_author_v1/receipt.json');doc=read('neural512_document_v1/receipt.json')
summary={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'author':author,'independent_document':doc,'optimizer_updates':0,'dataset_accepted':False,
 'physical_filter_executed':False,'full_generator_training_started':False,
 'optimizer_and_ema_states_actually_allocated':False}
(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
vae_peak=max((x['peak_allocated_bytes'] for x in author.get('vae',[])),default=0)
ldm_peak=author.get('ldm',{}).get('peak_allocated_bytes',0)
report=f'''# TextOp 69D完整网络 batch512 数值与显存验证

更新时间：{summary['time_beijing']}。当前完成的是**完整网络尺寸的短数值验证，优化器更新0次**。没有完成真实物理筛选、没有接纳正式训练语料、没有启动完整生成器训练，也没有生成—执行闭环结果。

## 实际运行范围

服务器GPU1、CPU4、Torch显存比例0.65，复用已验证的隔离环境，不修改共享依赖和官方源码。使用2,296条已完成运动建立的未过滤候选视图及其真实CLIP缓存、仅训练集统计；视图来源和哈希固定。模型为随机初始化的论文尺寸：VAE 9层、LDM 8层，hidden512、FFN1024、4头、latent128，batch512、history2/future8、4个连续primitive。

直接调用固定官方版本的MVAEManager.calc_loss与DARManager.calc_loss，保留Huber重建、KL和几何损失。VAE分别用真实历史、停止梯度的自生成历史，各执行4次反向传播；LDM冻结随机VAE，用干净潜变量预测目标及停止梯度历史执行4次反向传播。所有损失及梯度有限且非零，模型参数运行前后SHA一致。另执行两段5步CFG5数值采样，每个样本16帧；这只是有限值检查，不代表动作语义质量、物理可执行性或单流实时速度。

作者运行状态`{author.get('status','PENDING')}`；独立按文档运行`{doc.get('status','PENDING')}`。直接退出和子进程退出、超时与资源采样保存在聚合证据目录；本验证不用Isaac SDK，不涉及SDK退出兼容层。

## 显存的实际含义

作者运行中Torch峰值分配：VAE {vae_peak:,}字节（{vae_peak/1024**3:.3f}GiB），LDM {ldm_peak:,}字节（{ldm_peak/1024**3:.3f}GiB）。这是模型前向/反向及几何损失的峰值，**未创建优化器，未实际分配Adam/AdamW状态或EMA副本**，也没有正式全语料与检查点写入的开销。因此不能称为完整训练显存验收。设备总显存采样还包含其他已有进程，不等于本进程的Torch峰值。

代码给出的Adam+EMA额外字节数是按参数量计算的估计，不是实测。后续仍需完成真实优化器/EMA/恢复验证和最终合格语料运行检查，才能启动正式训练。随机VAE的latent scale1不能代替训练好的VAE或证明潜空间质量。

## 训练步数与恢复配置核对

固定官方训练循环对每个primitive执行一次optimizer.step和manager.post_step，4个primitive共用一次加载器批次。因此论文VAE200,000、LDM300,000个manager步分别约对应50,000、75,000次四片段批次抽样；不能误乘4，把它们变为80万、120万次优化。出现官方非有限梯度跳过时，manager步仍递增，必须同时报告实际优化器更新数和跳过数。VAE使用Adam，LDM使用AdamW；LDM冻结训练完成的69D VAE，明确绑定其检查点，不搜索旧57D目录。

当前已有57D训练、PhyGile和ReactiveBFM继续暂停。原有GMR重定向、私有备份和串行FK导出工作者保持运行。真正完成全部可得源、宽松/严格物理筛选和最终语料/统计/CLIP/运行验收后，已获授权自动接续新69D VAE200k→LDM300k及实际闭环，不等待再次确认。
'''
(b/'reports/neural_batch512_20261025.md').write_text(report,encoding='utf-8')
(local/'69D完整网络batch512数值验证_20261025.md').write_text(report,encoding='utf-8')
(local/'69D完整网络batch512数值验证_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>TextOp完整网络batch512验证</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.8 system-ui}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
print(json.dumps({'author':author.get('status'),'document':doc.get('status'),'optimizer_updates':0,'dataset_accepted':False}))
