"""Whitelist aggregate synthetic-protocol checks; never label them physics."""
import html,json,pathlib
from session import REPO,remote
b=REPO/'baselines/textop';local=pathlib.Path('D:/具身智能/TextOp复现_20261024')
dest=b/'results/filter_protocol_20261025';dest.mkdir(parents=True,exist_ok=True)
data=json.loads(remote("import pathlib,json;r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');print(json.dumps({'receipt':(r/'receipts/filter_evidence_v2_20261025.json').read_text(),'log':(r/'logs/filter_evidence_v2_20261025.log').read_text()}))"))
for key,leaf in [('receipt','receipt.json'),('log','regression.log')]:(dest/leaf).write_text(data[key],encoding='utf-8')
receipt=json.loads(data['receipt']);assert receipt['exit_code']==0 and not receipt['physical_rollouts_executed']
report='''# TextOp 物理筛选验收规则核查

已补上完整轨迹覆盖计数，以及训练凭据、实际更新数、退出状态、检查点和参考库指纹的一致性检查。17项合成逻辑测试在本地与服务器均通过，服务器Python3.8.10实际退出0。

10秒超时不能算完整长动作通过；没有真实尝试、漏掉严格评估、短PPO诊断冒充40,000轮、检查点被替换等记录都会被拒绝。初始重置不计为成功；最后一帧同时触发失败时按失败处理。

**这仅是验收规则的代码测试，没有运行物理评估，也没有动作被接纳。** 真实全轨迹仿真评估器仍需实现并验证；候选数据继续由现有GMR队列处理。完整生成器仍等待实际两轮过滤与数据验收。

原始动作、逐条文本和人体模型保持私有。详见仓库FILTER_EVIDENCE_PROTOCOL.md及results/filter_protocol_20261025。
'''
(b/'reports/filter_evidence_protocol_20261025.md').write_text(report,encoding='utf-8')
(local/'物理筛选验收规则_20261025.md').write_text(report,encoding='utf-8')
(local/'物理筛选验收规则_20261025.html').write_text('<!DOCTYPE html><meta charset="utf-8"><title>TextOp物理筛选验收规则</title><style>body{max-width:950px;margin:40px auto;font:16px/1.8 system-ui}pre{white-space:pre-wrap;font:inherit}</style><pre>'+html.escape(report)+'</pre>',encoding='utf-8')
print(json.dumps({'test_cases':receipt['test_cases'],'exit_code':receipt['exit_code'],'physical_filter_executed':False}))
