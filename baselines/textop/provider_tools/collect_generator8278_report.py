"""Publish aggregate generator8278 report without rewriting older evidence."""
import html,json,pathlib,re
BASE=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
plan=json.loads((BASE/'configs/generator8278-priority-plan.json').read_text(encoding='utf-8'))
p=BASE/'results/generator8278_20261027/generator8278_vae200k_train_v1/progress.json'
progress=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {};r=progress.get('receipt') or {}
state=r.get('status','CORPUS_VERIFIED_NEURAL_ACCEPTANCE_PENDING');steps=r.get('completed_primitive_attempts',0);updates=r.get('optimizer_updates',0)
text=f"""# TextOp 全来源生成器训练（物理筛选前）

用户已选择先用8278条预处理来源扩大语义覆盖。2个TRAIN源族因与验证集精确姿态重复被隔离，原资产保留；实际8276源族/16552含镜像变体，93632标注对。归一化只用TRAIN，真实CLIP缓存和三种512×4批次加载已验收；完整物理筛选尚未完成。

当前状态：{state}。VAE尝试{steps}/200000，实际Adam更新{updates}；跳过{r.get('skipped_updates',0)}。当前实验不是筛选数据论文复现通过。LDM目标300000步，须冻结这次新训练VAE后独立验收正式full-sample训练入口再接续；尚未启动。

| 配置 | VAE | LDM |
|---|---|---|
| 总步数/阶段 | 200000 /100000+50000+50000 | 300000 /100000×3 |
| Batch/初始学习率 |512/1e-4|512/1e-4|
| Transformer层/头 |9/4|8/4|
| Hidden/FFN |512/1024|512/1024|
| latent/noise |128|128|
| 最大自回归历史替换概率 |1.0|0.8|

两阶段共同29DoF/69D、50Hz、history2/future8、4primitive；VAE Adam，LDM AdamW，EMAfalse、gradclip1。LDM冻结CLIP与新VAE、5步cosine/x0、CFG5、textmask.1。发布命令use_full_sample=True在stage>0生效，快版false差异保留。

VAE检查点50k间隔，4常规+1紧急，各528MiB上限；2GiB磁盘保留/512MiB日志，每100步及保存复验。原发布20k保存间隔因容量改为50k，数学/课程不变。派生大数据缓存RAM可重建，统计/CLIP与恢复信息已D和server私有备份，原NPZ持久备份保持；checkpoint仅持久磁盘。D优先最终阶段恢复文件，中间版本仍server持久保留，不删除。不能承诺全部中间D备份/后续WBC媒体容量。

原快版工程链路通过，但语义质量未通过；固定文本/无条件、多seed和TMR指标评估仍需推进。后续筛选数据对照、完整生成器、最终WBC/增强、实际在线语义及物理质量/视频达到目标后通知。

论文与配置：[论文TableVI](https://arxiv.org/html/2602.07439v1#A1.T6)、[固定官方训练命令](https://github.com/TeleHuman/TextOp/blob/ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22/USAGE.md)。
"""
(BASE/'reports/generator8278_current_20261027.md').write_text(text,encoding='utf-8')
parts=[];table=False
for line in text.splitlines():
 if line.startswith('|'):
  cells=[x.strip() for x in line.strip('|').split('|')]
  if all(set(x)<=set('-: ') for x in cells):continue
  if not table:parts.append('<table>');table=True
  parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>');continue
 if table:parts.append('</table>');table=False
 if line.startswith('# '):parts.append('<h1>'+html.escape(line[2:])+'</h1>')
 elif line:parts.append('<p>'+html.escape(line)+'</p>')
if table:parts.append('</table>')
body='<!doctype html><meta charset="utf-8"><title>TextOp全来源生成器训练</title><style>body{font:17px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:20px;background:#101927;color:#edf2fa}table{width:100%;border-collapse:collapse}td{padding:9px;border:1px solid #52647b}a{color:#7fc4ff}</style>'+''.join(parts)
name='TextOp全来源生成器训练_20261027.html';(D/name).write_text(body,encoding='utf-8');(D/'TextOp全来源生成器训练_20261027.md').write_text(text,encoding='utf-8')
main=D/'TextOp下载与训练报告.html'
if main.exists():
 old=main.read_text(encoding='utf-8');old=re.sub(r'<!--generator8278-current-start-->.*?<!--generator8278-current-end-->','',old,flags=re.S)
 section='<!--generator8278-current-start--><section style="padding:22px;background:#e7f0ff;color:#16304d;font:17px/1.7 system-ui"><h2>最新：全来源生成器优先训练</h2><p>'+html.escape(state)+f'：VAE {steps}/200000，实际更新{updates}。当前为物理筛选前实验，完整物理与语义质量尚未通过。</p><a href="'+name+'">查看实际训练进度、参数和存储范围</a></section><!--generator8278-current-end-->'
 match=re.search(r'<body[^>]*>',old,re.I)
 if match:old=old[:match.end()]+section+old[match.end():]
 else:old=section+old
 main.write_text(old,encoding='utf-8')
print(json.dumps({'status':state,'attempts':steps,'updates':updates,'report':str(D/name)},ensure_ascii=False))
