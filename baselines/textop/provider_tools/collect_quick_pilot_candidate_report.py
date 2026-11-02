"""Report the new disabled pilot separately from real physical results."""
import datetime, html, json, pathlib
B=pathlib.Path(__file__).resolve().parents[1];R=B/'results/quick_pilot_candidate_20261026'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
stage=json.loads((R/'stage.json').read_text(encoding='utf-8'))
v2=json.loads((R/'trace_semantics_v2.json').read_text(encoding='utf-8'))
selection=json.loads((R/'selection_metadata.json').read_text(encoding='utf-8'))
spec=json.loads((B/'configs/quick-pilot-spec.json').read_text(encoding='utf-8'))
progress=json.loads((B/'results/full_contact_20261026/formal_launch/progress.json').read_text(encoding='utf-8'))
doc=R/'quick_pilot_document_preflight_v2.json'
doc=json.loads(doc.read_text(encoding='utf-8')) if doc.exists() else None
sm=selection['selected']['smoke'];rep=selection['selected']['sample']
text=f'''# TextOp快速版：小规模真实跟踪入口准备

更新：{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}

**本次完成的是独立回放候选代码和CPU核验，尚未执行机器人跟踪试验。** 服务器跟踪策略采样为{progress['completed_learning_iterations']:,}/5,000轮、{progress['optimizer_updates']:,}次实际更新。现有协调器继续独占5,000轮检查点备份与受控退出，不重复启动或提前发信号。

|阶段|实际状态|
|---|---|
|预处理|8,278原始动作和镜像、文本对齐、69D、FK及私有备份已完成；本次未重复处理|
|快速teacher|有效预算5,000轮；原40,000轮参数及16份运行文件保留|
|新quick回放候选|单独schema/程序/收据；要求真实5,000快照、备份和受控退出证据|
|真实smoke / 64动作pilot|尚未运行，当前两项启动标志为{spec['pilot_launch_enabled']} / {spec['sample_launch_enabled']}|
|正式全库物理筛选、生成器、闭环视频|尚未完成；不能用候选CPU结果替代|

原40,000轮v3/v4验收条件保持原样，没有伪造40,000轮或800,000次更新。新入口固定标注`QUICK_REPRODUCTION_NOT_PAPER_COMPLETE`，不输出正式语料接纳文件，也不按早期teacher的5%失败门槛批量淘汰8,278动作。

## 已核验的新代码与实际数据范围

第一版本地及原服务器CPU各28项检查通过；独立文档预检实际退出0、15项通过。随后根据独立审查补充参考帧时序说明和记录断言，第二版本地及原服务器各31项检查通过。新版本独立只读文档预检：{str(doc['check_count'])+'项通过' if doc else '待收集'}。这些都是CPU/元数据核查，新增SDK、真实动作试验、优化更新和训练信号均为0。

真实私有清单的选择核查使用同一8,278条原始来源顺序，指纹为`{selection['reference_fingerprint']}`。8条smoke和64条pilot选择不读取失败EMA、无相互重叠、没有据此删除或拒绝数据。逐来源映射在服务器和D盘私有目录按SHA一致保存，GitHub仅公开聚合统计。

|回放范围|完整动作数|帧数|实际来源分布|当前物理试验|
|---|---:|---|---|---|
|短smoke|8|{sm['minimum_frames']}–{sm['maximum_frames']}|KIT5、CMU1、DFaust2|0|
|代表性pilot|64|{rep['minimum_frames']}–{rep['maximum_frames']}|10个数据集；22/23/19条分别位于50–250/251–750/751–2000帧区间|0|

64条pilot的训练/验证来源标注分别为47/17；teacher训练使用原参考库，因此这不是未见动作泛化测评。pilot时8,214条其他原始动作仍未尝试，超出时长范围的330条也没有被判断为不可执行。

## 实际执行记录与尚未验证的部分

新记录器在IsaacLab的终止计算内部、自动reset之前复制物理根位姿、29关节状态、14个跟踪身体位置及当前参考，避免把终止后的重置姿态误写成执行结果。每步写入私有二进制日志，完整/失败试验另存压缩数组；超时或不完整轨迹保留但不判通过。存储上限256MiB，写入和压缩前检查余量。真实运行监督器每15秒采样并保存实际退出和失败记录，沿用原有原生关闭/有界fallback的区别。

独立审查指出：原后端在终止/重置之后才推进参考时间。T帧动作的T−1次物理调用按零阶保持覆盖名义时长，但当前参考索引依次为0..T−2，不能冒充另行核验了末帧T−1端点。第二版已断言和记录实际消费的帧序列，并明确端点比较尚未完成。误差仅为当前控制目标的自定义跟踪诊断，不标为论文最终指标。

8条smoke要求至少一条自然结束；若观察到失败就完整保留。若全部成功，不能声称真实失败分支也已覆盖。当前没有实际SDK smoke，原有合成失败夹具只是CPU证据。后续在原训练真正退出、锁和资源重新验收后串行进行作者及fresh独立实际smoke，再开启64条pilot。不要中断当前训练抢占SDK。

真实执行视频仍待物理轨迹生成及渲染；参考动画不能替代。VAE200k/LDM300k原配置没有被本次静默缩短；快版后续语料、训练预算和生成—执行结果仍需明确记录。作者精确碰撞资产不可得、SMPL/AMASS/BABEL私有许可范围及13条CMU/403条作者私有数据缺失等差异继续保留。

复核入口：`QUICK_PILOT_ENVIRONMENT.md`。证据：`results/quick_pilot_candidate_20261026`。既有数据集、检查点、唯一实验记录和其他用户目录均未删除或改动。
'''
(B/'reports/quick_pilot_candidate_20261026.md').write_text(text,encoding='utf-8')
(D/'TextOp快速跟踪候选与数据范围_20261026.md').write_text(text,encoding='utf-8')
try:
 import markdown
 body=markdown.markdown(text,extensions=['tables'])
except ImportError:body='<pre>'+html.escape(text)+'</pre>'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp快速跟踪候选</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:9px}th{background:#edf3f8}pre{white-space:pre-wrap}</style>'+body+'</html>'
(D/'TextOp快速跟踪候选与数据范围_20261026.html').write_text(page,encoding='utf-8')
print(json.dumps(dict(status='QUICK_PILOT_CANDIDATE_REPORT_UPDATED',real_trials=0,teacher_completed=progress['completed_learning_iterations'])))
