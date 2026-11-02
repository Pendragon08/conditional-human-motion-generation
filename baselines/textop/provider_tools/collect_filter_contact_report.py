"""Publish only aggregate CPU readiness evidence, never private motion records."""
import datetime,html,json,pathlib
B=pathlib.Path(__file__).resolve().parents[1];R=B/'results/filter_contact_protocol_20261026'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
s=json.loads((R/'server_cpu_v1.json').read_text(encoding='utf-8'))
l=json.loads((R/'local_cpu_v3.json').read_text(encoding='utf-8'))
f=json.loads((B/'results/full_contact_20261026/formal_launch/progress.json').read_text(encoding='utf-8'))
text=f"""# TextOp：物理筛选接触一致性与退出后验收准备

更新：{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}

服务器特权跟踪策略仍在正式训练。采样收据已完成 **{f['completed_learning_iterations']:,} / 40,000 轮、{f['optimizer_updates']:,} 次实际优化更新**，价值损失为 {f['latest_losses']['value_function']:.8f}，尚无退出收据。当前只是早期训练，不能据此认定长期收敛或筛选通过。8,278 条可得源动作的预处理、私有备份已完成；完整生成器尚未启动。

## 本轮补齐了什么

为后续真实物理筛选准备独立代码，要求训练与评估使用同一已验证地面接触设置：静态箱体平地，保留自碰撞。旧 Plane 执行器和历史证据保留，新方案未修改正在运行的 16 个源码及配置文件，部署前后哈希均相同。

后续每段完整回放的结果将增量保存，保留失败试验；进程真正退出后才核对检查点、地面设置、动作覆盖、执行文件、退出状态及逐次回放记录。严格的骨盆 0.25 m、末端 0.5 m 阈值保持不变，未删除难动作或截断奖励。

|实际完成的检查|结果|范围|
|---|---|---|
|本地合成验收逻辑|{l['checks_passed']} 项通过|仅人工构造的协议测试，不是机器人回放|
|服务器 Python 3.8.10 合成验收逻辑|{s['synthetic_check_receipt']['checks_passed']} 项通过|CPU 0–3；零 SDK、物理步和优化更新|
|当前真实训练收据|被后续正式筛选入口拒绝|100 轮未满足 40,000 轮完成条件|
|新回放监督器禁用检查|按预期拒绝启动|没有新 GPU 工作或新回放目录|
|活动训练文件复核|16 份哈希一致|现有策略继续运行|

新回放器、监督器及退出后收集器**尚未做真实 SDK 验证，保持禁用**。CPU 测试通过不表示物理筛选已执行。后续还需补齐独立的有界实际验证入口、连续资源采样与异常退出处理，再完成实际验证；当前没有绕过这些条件。

## 还没有完成

- 40,000 轮特权跟踪策略与完整检查点备份。
- 真实宽松及严格两轮物理筛选、最终合格 69D 训练语料验收。
- 完整 VAE 200,000 → LDM 300,000 训练、最终 WBC、生成增强及生成—执行视频。

沿用既有自动接续授权：真实筛选和最终语料、归一化、CLIP 缓存、断点及资源验收通过后直接开始完整生成器训练，无需再次确认。作者未公开的精确碰撞资产、403 条私有动作及13条所选CMU缺失仍明确列为差异。

## 保留的证据与限制

公开聚合证据位于 results/filter_contact_protocol_20261026；精确候选代码字节另存恢复包，防止 Git 换行转换破坏核对。原始运动、逐条文本/来源索引和真实回放日志保持私有。

本地收集器在服务器检查已成功后出现过一次目录层级错误；已保留失败说明并只更正本地收集路径，没有重复服务器验证、重启训练或删除文件。较早20项和22项合成检查记录也保留。

内部一致的哈希和日志不能自行证明真实物理执行，实际执行与独立观察仍是必要条件。没有新结果视频，因为本轮未进行新的物理回放或闭环实验。
"""
(B/'reports/filter_contact_protocol_20261026.md').write_text(text,encoding='utf-8')
(D/'物理筛选接触一致性与退出后验收准备_20261026.md').write_text(text,encoding='utf-8')
try:
 import markdown
 body=markdown.markdown(text,extensions=['tables'])
except ImportError:body='<pre>'+html.escape(text)+'</pre>'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp 后续筛选准备</title><style>body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:9px}th{background:#edf3f8}pre{white-space:pre-wrap}</style>'+body+'</html>'
(D/'物理筛选接触一致性与退出后验收准备_20261026.html').write_text(page,encoding='utf-8')
print(json.dumps({'local':str(D/'物理筛选接触一致性与退出后验收准备_20261026.html'),'physical_filter_executed':False,'new_sdk_runs':0},ensure_ascii=False))
