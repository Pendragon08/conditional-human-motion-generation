"""Report bounded-validation preparation; never promote CPU checks to physics."""
import datetime
import html
import json
import pathlib

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
evidence = json.loads((B / 'results/contact_replay_v4_20261026/server_cpu_author_v1.json').read_text(encoding='utf-8'))
progress = json.loads((B / 'results/full_contact_20261026/formal_launch/progress.json').read_text(encoding='utf-8'))
text = '''# TextOp：有界物理回放验证入口准备

更新时间：{time}

服务器正式跟踪策略继续训练，最新完整采样为 **{rounds:,}/40,000轮、{updates:,}次实际优化更新**。第1,000轮检查点已备份；当前还没有完成真实两轮物理筛选或生成器训练。

本轮补齐了后续回放的独立验证模式，存为新的执行器版本4。已有16份活动训练文件及9份旧候选文件均未改变，新入口没有抢占正在训练的SDK任务。

|准备内容|实际结果|
|---|---|
|独立validation模式|只回放8条1–10秒的完整原始动作，不裁剪片段；上限1,024次控制调用、600秒监督时间|
|训练完成条件|验证模式和正式筛选都要求真实40,000轮策略及成功退出；当前运行中的策略被拒绝|
|模式隔离|验证结果不生成正式筛选决策；启用验证开关不能开启正式筛选|
|本地与服务器CPU检查|各13项通过；包含不完整策略拒绝、源动作选择、正式验收拒绝验证收据|
|真实来源的元数据准备|8,278条来源顺序一致，可选择8条完整短动作，长度范围52–64帧；未运行物理仿真|
|当前启动状态|验证、正式筛选、真实SDK验证完成三个标志均为false；两种入口都按预期拒绝派发|

**本轮完成的是代码和CPU准备，真实SDK验证尚未执行。** 服务器训练持有共同互斥锁，后续回放要等当前训练完成后进行。没有将合成测试里的40,000轮数字当成真实训练，也没有用8条短动作冒充全量物理筛选。

下一步是在真实40,000轮检查点和退出凭据具备后，按文档分别完成作者执行及独立文档执行，核对实际退出、资源、完整动作结束与失败记录。实际验证输出的验收和激活凭据还需要补齐；验证通过后才启用全部动作的正式回放与退出后验收。

正式筛选与语料验收通过后，继续按既有授权直接接续VAE200,000 → LDM300,000及最终控制、生成增强和真实闭环视频。近期基于每轮约6.36秒给出的3–4天是包含后续验收的暂定排期，严格全量回放仍没有实测耗时，不能据此承诺完成日期。

精确代码字节和聚合证据保存在 results/contact_replay_v4_20261026；具体动作索引/源文件映射仅在服务器个人私有目录。旧失败、旧候选、原始数据和所有检查点保留。
'''.format(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),
           rounds=progress['completed_learning_iterations'], updates=progress['optimizer_updates'])
(B / 'reports/contact_replay_v4_20261026.md').write_text(text, encoding='utf-8')
(D / '有界物理回放验证入口准备_20261026.md').write_text(text, encoding='utf-8')
try:
    import markdown
    body = markdown.markdown(text, extensions=['tables'])
except ImportError:
    body = '<pre>' + html.escape(text) + '</pre>'
page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp 有界回放验证准备</title><style>body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:9px}th{background:#edf3f8}pre{white-space:pre-wrap}</style>' + body + '</html>'
(D / '有界物理回放验证入口准备_20261026.html').write_text(page, encoding='utf-8')
print(json.dumps({'report_saved': True, 'new_sdk_runs': 0,
                  'cpu_checks': evidence['test_receipt']['checks_passed'], 'actual_sdk_validation_completed': False}))
