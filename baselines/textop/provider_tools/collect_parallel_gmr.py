"""Public compatibility/timing evidence only; no licensed payload transfer."""
import datetime,html,json,pathlib
from session import REPO,remote
b=REPO/'baselines/textop';local=pathlib.Path('D:/具身智能/TextOp复现_20261024');dest=b/'results/parallel_gmr_20261025';dest.mkdir(parents=True,exist_ok=True)
files=json.loads(remote(r'''
import pathlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');files={}
for run in ['author_v1','author_v2','author_v3','author_v4','document_v1']:
 for leaf in ['receipt.json','serial.log','parallel.log']:
  p=r/'parallel_validation'/run/leaf
  if p.exists():files[run+'/'+leaf]=p.read_text(errors='replace')
p=r/'configs/gmr-parallel-activation.json'
if p.exists():files['activation.json']=p.read_text()
print(json.dumps(files))
''',timeout=90))
for name,value in files.items():
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8')
independent=REPO.parent.parent/'textop_reproduction_20261024/parallel_gmr_doc_witness_v1/REPORT.md'
if independent.exists():(dest/'independent_REPORT.md').write_text(independent.read_text(encoding='utf-8'),encoding='utf-8')
def read(name):
 p=dest/name;return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
receipts={r:read(r+'/receipt.json') for r in ['author_v1','author_v2','author_v3','author_v4','document_v1']}
summary={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'receipts':receipts,
 'activation':read('activation.json'),'physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False}
(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
rows=[]
for name,value in receipts.items():
 times=value.get('runs',{});s=times.get('serial',{}).get('elapsed_seconds');p=times.get('parallel',{}).get('elapsed_seconds')
 rows.append('| '+name+' | '+value.get('status','PENDING')+' | '+(f'{s:.2f}' if s else '待定')+' | '+(f'{p:.2f}' if p else '待定')+' | '+(f'{s/p:.2f}×' if s and p else '待定')+' |')
report=f'''# TextOp GMR 四进程并行验证

更新时间：{summary['time_beijing']}。正式批次调度模式：`{summary['activation'].get('mode','UNKNOWN')}`。

在已授权的4个CPU核范围内，让4个独立进程分别处理不同完整动作；每条动作内部仍顺序调用同一个GMR实例。没有把逐帧IK改成GPU算法，没有增加到16个CPU核，没有改动原始动作、IK约束或69D数学定义。当前配置保留SMPL-X的Torch/MKL4线程，仅把NumPy/SciPy OpenBLAS降为1线程，所有进程共同限制在CPU0–3。

## 同一批8条完整动作的实际对照

| 运行 | 状态 | 串行秒数 | 并行秒数 | 观测速比 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

每次对照8条完整CMU动作、合计5,040个50Hz姿态帧（镜像另计）。比较重定向轨迹、接触、镜像、69D特征和时间数组，并核对文本分段/划分等元数据；还与此前存档的同源CMU结果对照。浮点容差事先固定为atol1e-5、rtol1e-6，接触/时间/标注必须精确一致。诊断样本不增加正式数据进度，也不代表物理筛选通过。

第一版把Torch和BLAS线程都降为1，出现0.00016665的关节轨迹差异，未通过原有精度门槛，失败和源代码保留。第二版保留全部4线程，结果精确一致但仅约1.08倍加速。第三版调整OpenBLAS线程仍只有约1.07倍。第四版进一步让空闲OpenMP辅助线程休眠，保持数值计算配置不变，作者测试获得约2.41倍加速且数组差异为0。仅在作者和独立文档执行均通过后，启用BMLrub/KIT的并行门禁；未放宽精度条件来制造通过。

正式分片按声明帧数平衡，每条源动作仅归属一个进程；父进程核对完成文件SHA后汇集到唯一规范目录。工作进程的临时清单位于备份目录外，避免备份重复收录；最终仍是一份动作清单、一份汇总进度和每源一个NPZ。已有上传、批次间持久备份条件和后续FK/物理筛选流程保留。跨批次仍按备份依赖接续，批内采用并行。

小样本加速比包含进程启动和模型加载开销，不保证大批次同样倍数。CPU并行不能消除原始输入传输、备份和校验耗时；完成时间应结合后续正式批次的实际速率更新。数据、检查点、失败记录均保留；仅代码和聚合证据公开，受限资产继续保存在个人私有目录。
'''
(b/'reports/parallel_gmr_20261025.md').write_text(report,encoding='utf-8')
(local/'GMR并行处理验证_20261025.md').write_text(report,encoding='utf-8')
(local/'GMR并行处理验证_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>GMR并行验证</title><style>body{max-width:1040px;margin:40px auto;font:16px/1.8 system-ui}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
print(json.dumps({'author':receipts['author_v4'].get('status'),'document':receipts['document_v1'].get('status'),'mode':summary['activation'].get('mode')}))
