"""Summarize read-only private-backup audit receipts; no restricted contents."""
import datetime,html,json,pathlib
from session import REPO

b=REPO/'baselines/textop';dest=b/'results/preprocessing_backup_audit_20261025'
local=pathlib.Path('D:/具身智能/TextOp复现_20261024')
runs={}
for p in sorted(dest.glob('*/receipt.json')):
 runs[p.parent.name]=json.loads(p.read_text(encoding='utf-8'))
task=REPO.parent.parent/'textop_reproduction_20261024'
independent=task/'preprocessing_backup_doc_witness_v1/REPORT.md'
if independent.exists():(dest/'independent_REPORT.md').write_text(independent.read_text(encoding='utf-8'),encoding='utf-8')
assert runs,'No actual backup-audit run found'
latest=max(runs.values(),key=lambda x:x['time_beijing'])
summary={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'runs':runs,
 'latest_verified_sources':latest.get('verified_sources',0),'preprocessing_complete':latest.get('preprocessing_complete',False),
 'physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False}
(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
report=f'''# TextOp 预处理持久备份验收

更新时间：{summary['time_beijing']}。本页核查的是已保存到本地磁盘的私有预处理备份，不是服务器正在计算中的动作数量，也不是物理筛选通过数量。

最新实际状态`{latest['status']}`。已重新从持久压缩包读取并验证{latest.get('verified_sources',0):,}条源动作及其镜像，尚有{latest.get('remaining_available_sources',0):,}条可获取源未通过这项完整备份验收；当前预处理完成标志为`{latest.get('preprocessing_complete',False)}`。CMU处理中的部分结果不会提前计作完整持久备份。

| 批次 | 持久备份中逐条核查的源动作数 | 状态 |
| --- | ---: | --- |
'''
for row in latest.get('cohorts',[]):report+=f"| {row['id']} | {row['per_motion_sha_verified']:,} | SHA与数组检查通过 |\n"
for name in latest.get('pending_cohorts',[]):report+=f'| {name} | 0 | 等待完整处理和持久备份 |\n'
report+='''
核查项目包括整个压缩包SHA、每个NPZ与备份内清单的SHA对应、源ID和训练/验证划分、原始及镜像数组形状与有限值、50Hz时间轴、单位四元数、二值接触、69D中的接触/根高度/关节位置/关节差分字段、镜像根变换，以及文本分段的帧边界和时间一致性。这里没有重新计算所有FK或证明物理可执行；完整编码/镜像FK检查仍以准备阶段实际执行记录为依据。被拒绝的无效时间标注保留在记录中，不把它们算成整条动作失败。

只有6个批次全部完成、全部压缩包及逐条内容验证通过、源集合恰好等于8,291条原始索引扣除13条所选CMU导出缺失，并且划分一致，才能产生“全部可获取源预处理和备份完成”的通知条件。403条作者私有运动仍不可得，不能声称作者全量数据复现。

此程序只读私有备份，不解压到新的数据目录，不删除/改写原始数据，不控制服务器任务，不启动筛选或训练，也不自行通知。逐条源列表和任何完整错误细节只存本地私有目录；GitHub保存代码、哈希和聚合结果。完成预处理后还要完成真实宽松/严格物理筛选、合格语料加载和完整训练验收；原授权自动接续仍有效。
'''
(b/'reports/preprocessing_backup_audit_20261025.md').write_text(report,encoding='utf-8')
(local/'预处理持久备份验收_20261025.md').write_text(report,encoding='utf-8')
(local/'预处理持久备份验收_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>TextOp预处理备份验收</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.8 system-ui}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['latest_verified_sources','preprocessing_complete','physical_filter_executed']}))
