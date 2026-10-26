"""Report an actual verified checkpoint without exposing per-motion statistics."""
import datetime
import html
import json
import pathlib

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
state = json.loads((B / 'results/formal_filter_checkpoints_20261026/round_1000/backup.json').read_text(encoding='utf-8'))
assert state['status'] == 'FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED'
progress = json.loads((B / 'results/full_contact_20261026/formal_launch/progress.json').read_text(encoding='utf-8'))
audit = state['audit']
text = '''# TextOp：首个正式跟踪策略检查点已保存并备份

更新时间：{time}

正式特权跟踪策略已保存第 **1,000轮** 检查点，其中实际优化更新 **20,000次**。当前采样进度为 {rounds:,}/40,000轮，服务器继续训练；这不表示训练已收敛、物理筛选完成或动作生成器已经开训。

|保存内容|位置与核验|
|---|---|
|完整恢复检查点，90,799,598字节|服务器原文件保留；本地D盘私有目录备份；两端SHA256一致|
|推理权重，30,208,218字节|服务器、本地D盘、GitHub Release三端大小与SHA256一致|
|训练代码、聚合检查结果、失败记录|继续保存到GitHub分支及本地报告目录|

完整检查点包含模型、优化器、归一化状态和逐动作自适应采样统计。为了保持逐动作信息私有，GitHub发布的是单独的推理版本，保留相同模型及归一化张量和训练轮数；它不含优化器和逐动作统计，不能代替完整恢复检查点。

实际CPU核查确认：所有Adam状态计数均为20,000；80个张量有限；公开导出的25个模型/归一化张量重新加载后逐项完全相同。557维特权观测、29维动作输出与当前策略一致。原服务器文件在核查、下载和上传后哈希未变。

**所有数据集、已有检查点和实验记录均保留，没有停止或重启训练。** 本轮没有启动新的SDK、物理回放或优化更新。当前40,000轮策略仍在训练，真实两轮筛选、合格语料、VAE/LDM、最终WBC及闭环视频尚未完成。

首轮CPU备份读取曾因旧Torch的weights-only解析器不支持序列化浮点操作码而失败，记录已保留。随后只对已核验来源和SHA的本次自生成检查点使用兼容CPU读取完成检查，没有安装或更改依赖。该问题与正式训练异常不同；实际训练仍使用既有SDK环境。这也不等于已经进行真实SDK恢复训练试验。

完整检查点SHA256：`{private_sha}`。

公开推理权重SHA256：`{public_sha}`。

[GitHub推理权重与说明]({release})

本地完整检查点：`{private_path}`。

下一个计划检查点为第5,000轮。继续训练期间保留当前全部文件；新检查点出现后检查工作者、容量和哈希，再进行备份。后续真实物理筛选及最终语料验收通过后，按已有授权自动接续完整生成器训练。
'''.format(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),
           rounds=progress['completed_learning_iterations'], private_sha=audit['full_checkpoint']['sha256'],
           public_sha=audit['public_inference']['sha256'], release=state['github_release'],
           private_path=state['local_paths']['full_checkpoint'])
(B / 'reports/formal_checkpoint_backup_20261026.md').write_text(text, encoding='utf-8')
(D / '首个正式策略检查点与备份_20261026.md').write_text(text, encoding='utf-8')
try:
    import markdown
    body = markdown.markdown(text, extensions=['tables'])
except ImportError:
    body = '<pre>' + html.escape(text) + '</pre>'
page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp 首个正式策略检查点</title><style>body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:9px}th{background:#edf3f8}pre,code{overflow-wrap:anywhere;white-space:pre-wrap}</style>' + body + '</html>'
(D / '首个正式策略检查点与备份_20261026.html').write_text(page, encoding='utf-8')
print(json.dumps({'checkpoint_round': 1000, 'private_full_verified': True,
                  'github_inference_verified': True, 'data_deleted': False}))
