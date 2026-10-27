"""User-corrected filtering-first route takes precedence over historical receipts."""
import html,json,pathlib,re
BASE=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
p=BASE/'results/generator_pause_20261027/private_backup_verified.json';r=json.loads(p.read_text(encoding='utf-8'))
assert r['status']=='PREFILTER_GENERATOR_PAUSED_CHECKPOINT_PRIVATE_BACKUP_VERIFIED'
loose=BASE/'results/teacher10000_full_filter_20261027/relaxed_scope_v1/summary.json'
l=json.loads(loose.read_text(encoding='utf-8')) if loose.exists() else None
extra='全库筛选尚未完成；下一步读取10k教师的全8278来源宽松阶段统计，再对通过第一阶段的动作做严格完整回放。'
if l:
 c=l['counts'];extra=f"已读取全部{l['source_count']}来源真实宽松训练统计：{c['eligible_for_strict']}条达到严格回放候选门槛，{c['rejected_relaxed_rate']}条宽松历史失败率超过5%，{c['unattempted']}条无有效尝试；严格回放与最终训练语料验收尚未完成。"
 if c['eligible_for_strict']==0:extra+=' 当前为空候选，不能建立合格语料，也不能据此宣称全部动作不可行。EMA最低18.66%、中位24.65%；原始累计失败率最低13.33%。发布版本统计有初始化与学习历史影响，不据此断言唯一根因。保持5%阈值并改进教师，在新15k检查点复查，不强制等40k才检查。'
recovery=BASE/'results/teacher10000_full_filter_20261027/progress.json'
if recovery.exists():
 j=json.loads(recovery.read_text(encoding='utf-8'));tr=j.get('receipt') or {};extra+=f" 当前教师接续状态{tr.get('status')}，精确状态恢复核验{tr.get('resume_exact_state_verified')}，收据轮数{tr.get('completed_learning_iterations')}，原生日志最新完成轮数{j.get('observed_completed_from_native_log')}；每100轮刷新收据，初始0不代表从0重训。"
text=f'''# TextOp：恢复先筛选、后训练的论文顺序

用户2026-10-27明确调整：安全保存并暂停未筛选数据生成器，先完成全库物理筛选。先前推荐的“未筛选全源先训练”属于额外对照，并非论文主线；现在已取消其自动接续，不能把这条路线说成论文训练数据已合格。

当前生成器已在完整批次边界保存并退出：{r['completed_attempts']}次primitive尝试、{r['optimizer_updates']}次实际Adam更新、{r['skips']}次跳过。完整模型、优化器、训练时钟和随机状态保留，CPU模型/Adam重载一致，CPU随机状态恢复通过；CUDA随机状态完整保存并检查结构，本轮未执行GPU恢复训练。检查点{r['bytes']}字节，SHA `{r['sha256']}`，服务器与D盘私有副本已核验。原始训练输出的非零退出和旧RUNNING收据均保留，独立暂停收据解释为用户授权的完整批次保存退出，不是200k训练完成。

暂停没有向真实训练进程发信号，也没有修改训练源码/参数。独立验证后的临时记录路径标记触发原程序已有紧急保存逻辑；真实检查点落盘、原进程退出、记录解压完整和共同GPU/SDK锁释放均有证据。先前信号方案的独立审查阻断与全部失败记录保留，未用于真实训练。

{extra}

接续顺序：10k教师全源宽松统计与严格回放 → 保留可跟踪的原始参考动作并配对BABEL文本 → 源隔离、TRAIN-only统计、真实CLIP和加载/恢复/容量验收 → 筛选语料上的VAE200k与LDM300k。10k是用户允许的提前评估预算，与论文40k筛选教师保持区别；严格通过率和质量若不足，保留失败并据实际证据改进教师。未尝试/失败原资产不删除，早期教师失败不证明动作绝对不可行。

原8278来源预处理及私有备份已经完成。当前未筛选VAE只保留为独立对照，不自动恢复，也不自动启动其LDM。完整语义质量、最终WBC/生成增强、在线执行与论文完整效果仍未通过。

论文依据：[TextOp §III-B及附录A-A](https://arxiv.org/html/2602.07439v1)：先用特权跟踪器排除难以可靠跟踪的运动，再将保留下来的AMASS运动与BABEL文本组合训练生成器；宽松阶段失败率超过5%的运动剔除，再按标准终止条件检查其余运动。
'''
(BASE/'reports/filter_first_restored_20261027.md').write_text(text,encoding='utf-8')
body='<!doctype html><meta charset="utf-8"><title>TextOp恢复论文筛选顺序</title><style>body{font:17px/1.8 system-ui;max-width:1020px;margin:40px auto;padding:24px;background:#101927;color:#edf2fa}a{color:#7fc4ff}</style>'+''.join('<h1>'+html.escape(x[2:])+'</h1>' if x.startswith('# ') else '<p>'+html.escape(x)+'</p>' for x in text.splitlines() if x)
name='TextOp恢复论文筛选顺序_20261027.html';(D/name).write_text(body,encoding='utf-8');(D/'TextOp恢复论文筛选顺序_20261027.md').write_text(text,encoding='utf-8')
main=D/'TextOp下载与训练报告.html'
if main.exists():
 old=main.read_text(encoding='utf-8');old=re.sub(r'<!--generator8278-current-start-->.*?<!--generator8278-current-end-->','',old,flags=re.S);old=re.sub(r'<!--filter-first-current-start-->.*?<!--filter-first-current-end-->','',old,flags=re.S)
 section='<!--filter-first-current-start--><section style="padding:24px;background:#edf6ee;color:#153a24;font:17px/1.8 system-ui"><h2>最新：生成器已保存暂停，恢复先筛选后训练</h2><p>'+str(r['completed_attempts'])+'步完整状态已在服务器与D盘核验。全库物理筛选尚未完成；未筛选实验不自动接续。</p><a href="'+name+'">查看暂停、备份及论文顺序说明</a></section><!--filter-first-current-end-->'
 match=re.search(r'<body[^>]*>',old,re.I);old=old[:match.end()]+section+old[match.end():] if match else section+old;main.write_text(old,encoding='utf-8')
print(json.dumps({'status':r['status'],'steps':r['completed_attempts'],'report':str(D/name)},ensure_ascii=False))
