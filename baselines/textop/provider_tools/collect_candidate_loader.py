"""Aggregate loader/scope evidence only; never copy private candidate data."""
import datetime,html,json,pathlib
from session import REPO,remote
b=REPO/'baselines/textop';local=pathlib.Path('D:/具身智能/TextOp复现_20261024')
dest=b/'results/candidate_loader_20261025';dest.mkdir(parents=True,exist_ok=True)
files=json.loads(remote(r'''
import pathlib,json,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');files={}
for label in ['candidate_loader_author_v1','candidate_loader_author_v2','candidate_loader_document_v1']:
 p=r/'generator_validation'/label/'receipt.json'
 if p.exists():files[label+'/receipt.json']=p.read_text()
 for folder,leaf in [('receipts','candidate_loader_'+label+'_exit.json'),('logs','candidate_loader_'+label+'.log')]:
  p=r/folder/leaf
  if p.exists():files[label+'/'+leaf]=p.read_text(errors='replace')
for name in ['candidate_cross_split_collision_v1.json','candidate_cross_split_raw_comparison_v1.json','source_scope_v1_20261025.json']:
 p=r/'receipts'/name
 if p.exists():files['additional/'+name]=p.read_text()
p=r/'logs/source_scope_v1_20261025.log'
if p.exists():files['additional/source_scope_regression.log']=p.read_text()
p=r/'scripts/audit_candidate_collision_v1.py'
if p.exists():files['historical_source/audit_candidate_collision_v1.py']=p.read_text()
profile=json.loads((r/'configs/paper-step2.json').read_text())['data_scope'];folder=r/'data_preparation_20261025/source_scope_inputs_v1'
for leaf,key in [('textop_original_verified_pairs.private.json','original_index_sha256'),('cmu_missing_selected_export.private.json','unavailable_index_sha256')]:
 assert hashlib.sha256((folder/leaf).read_bytes()).hexdigest()==profile[key]
print(json.dumps(files))
''',timeout=90))
for name,value in files.items():
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8')
task=REPO.parent.parent/'textop_reproduction_20261024';p=task/'candidate_loader_doc_witness_v1/REPORT.md'
if p.exists():(dest/'independent_REPORT.md').write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
def read(name):
 p=dest/name;return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
author=read('candidate_loader_author_v2/receipt.json');doc=read('candidate_loader_document_v1/receipt.json');audit=read('additional/candidate_cross_split_collision_v1.json')
summary={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'author':author,'independent_document':doc,
 'collision_audit':audit,'source_scope_tests':read('additional/source_scope_v1_20261025.json'),
 'physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False}
(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
report=f'''# TextOp 69D候选语料加载与覆盖范围验收

更新时间：{summary['time_beijing']}。本轮是未过滤候选语料的加载器验证，**没有完成真实物理筛选，没有接纳正式训练集，也没有开始完整生成器训练**。

## 实际加载验证

已读取全部2,296条完成批次的29DoF源运动及镜像。排除跨划分重复的1条训练源及其镜像后，候选视图4,590条（训练3,440、验证1,150）；原始资产全部保留，验证集未改。作者状态`{author.get('status','PENDING')}`，独立按文档执行`{doc.get('status','PENDING')}`。

复用官方加载器分别执行训练均匀采样、验证均匀采样、训练加权采样，batch512、history2/future8、4个连续primitive。每批形状[4,512,10,69]，CLIP文本[512,512]，相邻片段2帧重叠误差0、缺失文本嵌入0、归一化有限。冻结已有ViT-B/32权重计算2,999条不同文本的真实嵌入；本轮没有模型梯度或优化器。该峰值是CLIP/加载器用量，不能用来估计完整VAE/LDM显存。

仅用保留的训练运动拟合69D统计，float64总体统计、std下限1e-6；本轮没有维度触及下限。向量化编码时剔除每个跨源连接处，统计2,389,768个有效特征帧，并用真实片段检查与分源计算相同。最终物理筛选后必须重新拟合，不能把候选缓存直接当正式统计。

修复公开采样器两个边界问题：按有效源权重映射索引；允许最后合法起点及恰好35姿态帧的唯一窗口。官方片段提取、69D编码和文本选择规则保留，原仓库文件未改。

## 保留的失败与数据质量处理

首轮在跨划分检查处退出1，未导出训练语料。发现1条训练源与1条验证源（含镜像形成2组）重建后的275帧运动完全相同；源文件SHA不同，但人体动作数值差异最大约9.6×10⁻¹³，betas/性别/帧率相同。机器人动作并非静止或IK塌缩：关节最大运动范围约1.76rad、根位移范围约5.37m。逐条清单保持私有，失败、原始代码和聚合审计均保留。

采用明确的数据质量选择：保持验证集和权威数据不变，在候选训练视图中隔离该训练源的两个变体，并在隔离后重新拟合训练统计。正式过滤后重新检查重复关系和隔离清单，不直接套用本轮候选排除结果。原公开23DoF裁剪视图的既往无重复检查与本次全长原始动作检查是不同对象，不能互相替代。

## 覆盖范围检查

已明确固定公开索引8,291条、所选导出缺失13条、可用目标8,278条；403条作者私有运动不计入已获取数据，也不声称覆盖论文全部12,296条公共跟踪运动。source_scope.py核对私有索引/缺失清单的固定SHA、全部准备清单和实际过滤决策；任何可用源尚未处理或评估都会阻止启动。10项合成覆盖范围测试实际退出0，不等于数据验收。

完整训练门槛仍要求真正合格语料的独立加载/cache验证，候选视图dataset_accepted=false不能通过。已授权在真实验收后自动接续VAE20万→LDM30万和实际闭环；不增加人工确认。代码、聚合结果和报告同步GitHub、本地，受限索引、候选PKL及文本缓存保持私有。
'''
(b/'reports/candidate_loader_20261025.md').write_text(report,encoding='utf-8')
(local/'69D候选语料与覆盖范围验证_20261025.md').write_text(report,encoding='utf-8')
(local/'69D候选语料与覆盖范围验证_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>TextOp69D候选语料与范围验证</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.8 system-ui}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
print(json.dumps({'author':author.get('status'),'document':doc.get('status'),'candidate_variants':author.get('motion_variants'),'dataset_accepted':False}))
