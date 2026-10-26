"""Report effective quick budget separately from the frozen40k paper protocol."""
import datetime
import html
import json
import pathlib

B=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
plan=json.loads((B/'configs/quick-reproduction-plan-20261026.json').read_text(encoding='utf-8'))
from quick_progress_snapshot import snapshot
progress,state,connection_note=snapshot(B)
text='''# TextOp快速复现：先在5,000轮节点做小规模验证

更新：{time}

根据用户“我想快速复现，训练可以先不训练40000轮吗？”的要求，短期目标调整为 **5,000轮跟踪策略＋小规模真实跟踪评估**。所有结果标为快速验证，论文完整版40,000轮的结果口径和原始配置保留。

|项目|快速版安排|
|---|---|
|跟踪策略训练预算|5,000轮／100,000次优化更新，为论文轮数的12.5%|
|当前完整采样|{rounds:,}轮／{updates:,}次更新|
|自动切换启用|{enabled}；协调器状态：{state}|
|策略参数与仿真|沿用现有网络、PPO、G1、地面接触设置、GPU1和CPU0–3|
|检查点与记录|原服务器保留完整检查点；本地私有完整备份；GitHub仅公开推理权重及聚合结果|
|下一步|先小样本实际跟踪，判断早期策略可用性；保留所有动作，避免直接用弱策略批量拒绝数据|

原程序仍保持论文40,000轮配置，外部协调器在真实第5,000轮检查点稳定、完整私有备份及公开推理权重摘要核验成功后，才请求结束这次运行。它通过原始启动记录、用户目录、命令、进程启动时间和pidfd绑定唯一训练进程；不会向同用户其他任务或其他用户任务发送信号。

在切换实际发生前，当前训练继续。若备份或连接失败，协调器保存错误并停止自动操作，训练保持原状；不会为了提早退出而牺牲唯一检查点。少量检查点之后的内存更新可能无法恢复，后续续训从完整5,000轮快照开始。

原训练收据可能因受控中断记录KeyboardInterrupt/FAILED及非零退出，这些原始信息会保留，并由单独的用户授权提前结束记录解释；不会改写成40,000轮已完成。

**快速版不代表论文同规模复现。** 早期策略失败不能直接证明动作物理不可行。先评估小规模动作，效果不足时从检查点续训。真实快速回放和SDK验证仍需单独执行，以实际收据为准；既有要求40,000轮的正式筛选门槛没有被放宽。生成器VAE/LDM训练步数尚未因本次变更自动缩减，后续会明确数据范围和实验预算。

自动切换协调器运行在本地，训练仍在原服务器；本地需保持联网以完成备份和切换。约6.36秒/轮的既有实测只能用于估算，到达节点时间随共享资源而变。

核验记录位于results/quick_reproduction_20261026：11项CPU保护检查通过，独立v1文档预检通过；只读审查发现的数值PID复用竞态已在启用前改为pidfd＋启动时间绑定。3项实际CPU子进程pidfd检查和服务器11项保护检查通过；这些不是实际SDK退出或物理筛选完成的证明。独立v2文档预检、激活收据及真实切换结果分别保留。
'''.format(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),
           rounds=progress['completed_learning_iterations'],updates=progress['optimizer_updates'],
           enabled=plan['early_stop_enabled'],state=state['status'])
if connection_note:
    text=text.replace('更新：', '**'+connection_note+'**\n\n更新：', 1)
if (B/'results/quick_pilot_candidate_20261026/trace_semantics_v2.json').exists():
    text+='''

## 小规模回放候选准备记录

独立quick入口已形成禁用候选，新版本本地及服务器31项CPU检查通过；真实SDK回放尚未运行。已用实际私有清单核对8条完整短动作和跨10个数据集的64条代表动作选择，不使用失败EMA批量淘汰动作。记录器在自动重置前保存实际物理姿态；参考末帧端点尚未独立比较，不能将当前目标误差当作论文指标。详见本地《TextOp快速跟踪候选与数据范围_20261026.html》及仓库reports/quick_pilot_candidate_20261026.md。此准备记录不代表真实筛选、生成器或闭环完成。
'''
(B/'reports/quick_reproduction_20261026.md').write_text(text,encoding='utf-8')
(D/'TextOp快速复现安排_20261026.md').write_text(text,encoding='utf-8')
try:
    import markdown
    body=markdown.markdown(text,extensions=['tables'])
except ImportError:body='<pre>'+html.escape(text)+'</pre>'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp快速复现安排</title><style>body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:9px}th{background:#edf3f8}pre{white-space:pre-wrap}</style>'+body+'</html>'
(D/'TextOp快速复现安排_20261026.html').write_text(page,encoding='utf-8')
print(json.dumps({'quick_target':5000,'enabled':plan['early_stop_enabled'],'coordinator_status':state['status']}))
