"""Refresh aggregate contact diagnosis without publishing licensed arrays."""
import datetime, html, json, pathlib, shutil
B=pathlib.Path(__file__).resolve().parents[1]
R=B/'results/full_contact_20261026'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
formal_path=R/'formal_launch/progress.json'
formal=json.loads(formal_path.read_text(encoding='utf-8')) if formal_path.exists() else {}
formal_running=formal.get('status')=='RUNNING_FORMAL_FILTER_TRAINING' and formal.get('optimizer_updates',0)>0
comparison=json.loads((R/'contact_counterfactual_comparison.json').read_text(encoding='utf-8'))
labels={'baseline':'原 Plane 地面','no_self_collision':'仅关闭自碰撞（诊断）','no_floor':'仅移除地面（诊断）','solid_floor':'静态箱体平地，保留自碰撞（候选）'}
md=['# TextOp：地面接触对照与候选修复验证','',
'更新：'+datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'',
'当前结论：异常可以在不进行任何 PPO 更新时复现。移除地面会消除首步极端速度；仅关闭自碰撞仍会触发。保留地面支撑与自碰撞、将本地构造的无限 Plane 改为静态箱体平地后，24 个控制步的短回放没有出现原先的数值爆炸。具体底层触发机制仍未确定；候选方案尚不等于论文训练完成或数据通过筛选。','',
'## 已执行的对照','',
'所有完整场景对照使用同一 8,278 条参考库、8,192 个环境及相同初始根/关节状态，三个首发异常机器人的第一步动作差异均为 0。每组实际执行 96 个物理子步、24 个控制步，优化更新 0 次；运行实际退出码均为 0。','',
'|设置|首个物理步最大关节速度 rad/s|整个短回放最大关节速度 rad/s|采样最大接触力分量 N|',
'|---|---:|---:|---:|']
for x in comparison['rows']:
 md.append('|{}|{:.6g}|{:.6g}|{:.6g}|'.format(labels[x['case']],x['first_step_joint_velocity_max'],x['peak_joint_velocity'],x['peak_contact_force_component']))
md.extend(['',
'这些数值是异常定位指标，并非论文动作接纳阈值。没有据此删除动作、裁剪奖励或放宽物理筛选条件。无地面和关闭自碰撞只用于对照，未用于正式训练。', '',
'候选地面是 4,000 × 4,000 × 0.2 m 静态箱体，顶面 z=0，摩擦系数 1、恢复系数 0、multiply 合并模式；保留机器人自碰撞。改变的是地面碰撞体表示，不能宣称与作者未公开的精确碰撞资产一致。', '',
'## 排查范围与负面结果','',
'- 将三个初始异常状态单独放入三个环境时，没有复现完整场景的首步爆炸。因此小场景结果不能用作修复通过的证据。完整场景对照才复现原故障。',
'- 原生物理关节位置/速度与程序缓存一致；原生根状态仅有约 7.63e-6 的浮点差。所有副本质量与惯量一致且为正，尚未发现克隆质量或惯量差异。',
'- 原三状态回放与完整场景所选机器人的初始身体姿态/速度完全一致。这限制了“初始缓存读错”等解释，但仍不能单凭这些检查确定底层根因。',
'- 独立 USD 读取先后因 pxr 模块路径和 libtf.so 路径失败；随后仅为只读子进程补齐现有运行库路径成功，未改共享安装。碰撞节点检查只覆盖父层结构，不冒充完整网格几何核验。', '',
'## 实际 PPO 验证','', '首次独立前置核验发现 LF 字符串/CRLF 文件校验值不一致，以及本地与服务器的说明性配置字段差异，在 GPU 启动前停止。已保留失败证据、按实际文件字节修正校验值，并同步说明字段；执行代码、数据范围和数值参数未变。后续使用新的 document_v2 实际复核。',''])
ppo=[]
for run in ['solid_floor_ppo_author_v1','solid_floor_ppo_document_v2']:
 p=R/run/'receipt.json';e=R/run/'exit.json'
 if not p.exists() or not e.exists():
  md.append('- {}：未取得完整实际退出及回收证据，不能判为通过。'.format(run));continue
 x=json.loads(p.read_text(encoding='utf-8'));ex=json.loads(e.read_text(encoding='utf-8'))
 tracep=R/run/'numerical_trace.jsonl'
 trace=[json.loads(s) for s in tracep.read_text(encoding='utf-8').splitlines()] if tracep.exists() else []
 nonfinite=sum(v.get('nonfinite',0) for t in trace for v in list(t.get('fields',{}).values())+[t.get('reward',{}),t.get('contact_force_components',{})]+list(t.get('weighted_reward_rates',{}).values()))
 row={'run':run,'exit_code':ex['child_exit_code'],'status':x['status'],'optimizer_updates':x['optimizer_updates'],'iterations':x.get('completed_learning_iterations'),'transitions':x.get('environment_transitions'),'loss_history':x.get('loss_history',[]),'numerical_alert_free':x.get('numerical_alert_free'),'sampled_nonfinite_count':nonfinite,'peak_torch_bytes':x.get('peak_torch_allocated_bytes'),'native_close_returned':x.get('native_close_returned'),'teardown_fallback_used':x.get('teardown_fallback_used')}
 ppo.append(row)
 md.append('- {}：退出 {}，实际 {} 轮 / {} 次优化更新 / {} transitions，采样非有限值计数 {}；诊断警报为空={}。'.format(run,row['exit_code'],row['iterations'],row['optimizer_updates'],row['transitions'],nonfinite,row['numerical_alert_free']))
 if row['loss_history']:
  vals=[z['losses'].get('value_function') for z in row['loss_history']];md.append('  价值损失逐轮记录：{}。'.format(vals))
md.extend(['',
'短 PPO 验证只用于判断候选接触配置能否执行实际优化，不能证明 40,000 轮长期收敛、跟踪成功率或最终物理筛选合格。SDK 原生关闭仍可能卡住，既有有界退出保护与相关证据保留，不宣称修复了原生退出。', '',
'## 当前论文流程状态','',
'|阶段|状态|','|---|---|',
'|可得数据 GMR、文本对齐、镜像/69D 编码、私有备份|8,278 源 + 8,278 镜像已完成，之前已通知|',
'|特权跟踪策略正式训练|'+('已启动：{}/40,000 轮，{} 次实际优化更新；仍未完成'.format(formal.get('completed_learning_iterations'),formal.get('optimizer_updates')) if formal_running else '未取得正在运行的正式优化凭据；以真实退出/失败收据为准')+'|',
'|宽松 + 严格两轮物理筛选|未执行|',
'|合格语料与 VAE 200,000 / LDM 300,000|未启动完整训练|',
'|最终 WBC、生成增强、生成—执行视频|待后续训练和真实评价|','',
'范围限制保留：13 条选定 CMU 导出缺失、403 条作者私有动作不可得；8,278 源不等于作者全部 12,296 条公共跟踪动作。所有数据集、检查点和唯一实验记录均保留，旧 57D / PhyGile / ReactiveBFM 继续暂停。','',
('候选接触方案的实际 PPO 和独立复核、启动容量检查已通过，正式策略训练已接续。后续仍需真实两轮物理筛选及最终语料验收，再按既有授权直接接续完整 VAE→LDM 与闭环。' if formal_running else '下一步：完成独立实际复核、资源和检查点预算检查，再接续正式策略及真实两轮物理筛选；验收通过后按既有授权直接训练完整 VAE→LDM 与闭环，不重复征求确认。'),'',
'## 证据及元数据更正','',
'公开聚合证据在 results/contact_replay_20261026 和 results/full_contact_20261026；私有初始状态、来源映射及 SDK 日志位于本地 datasets/contact_diagnostics_private_20261026 和服务器独立 run 目录，已经逐文件 SHA 核验，未上传公开仓库。','',
'原始无自碰撞/无地面收据继承的 simulation_or_reward_changes=false 与实际不符；真实设置以 case/spec 为准，更正已保存在 contact_counterfactual_comparison.json，原始文件保留。零更新对照中的“one real PPO iteration”和继承的 expected_optimizer_updates=20 也不代表训练：实际为一次 rollout、0 优化更新。候选收据 paper_settings 中原 Plane 描述是基线设置，实际箱体在 diagnostic_floor/intervention 中明确记录。后续独立 PPO 规格明确为 10 轮/200 次更新。',''])
(R/'solid_floor_ppo_aggregate.json').write_text(json.dumps({'runs':ppo,'formal_profile_changed':formal_running,'physical_filter_executed':False,'full_generator_training_started':False},indent=2),encoding='utf-8')
if formal_running:
 md.extend(['## 正式策略训练已启动','',
 '已经按既有授权接续服务器上的 40,000 轮特权跟踪策略训练。采样时已完成 {} 轮、{} 次优化更新；这不是生成器训练，也不等于物理筛选完成。'.format(formal['completed_learning_iterations'],formal['optimizer_updates']),'',
 '采用已实际验证的静态箱体地面，仍保留自碰撞和论文 PPO 参数。CPU 绑定 0–3，CUDA/Torch 优化指向 GPU1、显存比例上限 0.65；独立复核观察到 SDK 在 GPU0 还建立约 456 MiB 辅助上下文，不能宣称整个 SDK 仅占用 GPU1。当前资源采样保存在启动证据中。','',
 '检查点计划为 1,000 轮及每 5,000 轮，启动预算包含 9 份检查点、2 GiB 保留空间和 512 MiB 日志余量。每次保存及每 100 轮复查容量；既有数据集和检查点不删除。','',
 '下一阶段是持续检查真实训练、备份新检查点、完成宽松与严格两轮物理筛选；最终语料验收后自动接续 VAE 200,000 → LDM 300,000 与实际闭环视频。'])
if (B/'results/filter_contact_protocol_20261026/server_cpu_v1.json').exists():
 md.extend(['', '## 后续筛选准备（尚未实际执行）', '', '新筛选入口要求评估沿用训练的同一接触设置，并在实际退出后核对全部回放证据。本地及服务器各23项合成逻辑检查通过，活动训练16份文件哈希未变。新回放器仍保持禁用，尚需真实SDK验证；不代表物理筛选或语料验收完成。详细记录见物理筛选接触一致性与退出后验收准备_20261026报告。'])
if (B/'results/contact_supervision_20261026/integration_cpu_v1.json').exists():
 md.extend(['', '## 后续筛选资源监督准备', '', '新增7项实际CPU子进程检查，以及本地/服务器各27项更新后的验收逻辑检查通过。新监督器保留超时、采样失败和实际退出证据，资源监测已经接入。没有新SDK回放，两个正式筛选标志仍禁用；下一步补齐独立有界实际验证入口。详情见后续物理筛选资源监测与失败保留_20261026报告。'])
backup_path=B/'results/formal_filter_checkpoints_20261026/round_1000/backup.json'
if backup_path.exists():
 backup=json.loads(backup_path.read_text(encoding='utf-8'))
 if backup.get('status')=='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED':
  md.extend(['', '## 首个正式策略检查点已备份', '', '第1,000轮/20,000次实际优化更新的完整检查点已保存，服务器原文件与D盘私有备份SHA一致。单独的推理权重已在服务器、D盘及GitHub Release三端核验；公开版本不含逐动作统计和优化器，完整恢复须使用私有检查点。没有删除文件或中断训练。这不是40,000轮完成、物理筛选通过或生成器训练完成。', '', '[推理权重及范围说明]('+backup['github_release']+')。详细记录见首个正式策略检查点与备份_20261026报告。'])
if (B/'results/contact_replay_v4_20261026/server_cpu_author_v1.json').exists():
 md.extend(['', '## 有界回放验证入口已准备，尚未运行SDK', '', '新执行器版本4增加独立验证模式：使用真实完成的40,000轮策略，回放8条完整短源动作，保留控制调用与超时预算。当前三个启动/验证标志仍禁用，本地与服务器各13项CPU检查通过，16份活动训练文件及9份旧候选文件不变。实际SDK验证、验证输出验收和正式全量筛选仍未完成；详细记录见有界物理回放验证入口准备_20261026报告。'])
text='\n'.join(md)
(B/'reports/contact_isolation_20261026.md').write_text(text,encoding='utf-8')
D.mkdir(exist_ok=True);(D/'地面接触对照与候选修复_20261026.md').write_text(text,encoding='utf-8')
try:
 import markdown
 body=markdown.markdown(text,extensions=['tables','fenced_code'])
except ImportError:body='<pre>'+html.escape(text)+'</pre>'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp 地面接触对照与候选修复</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #ccd6df}th{background:#edf3f8}pre{white-space:pre-wrap}a{color:#135fa3}</style>'+body+'</html>'
(D/'地面接触对照与候选修复_20261026.html').write_text(page,encoding='utf-8')
main=D/'TextOp下载与训练报告.html'
old=D/'TextOp主报告_地面接触诊断前快照_20261026.html'
if main.exists() and not old.exists():shutil.copy2(main,old)
main.write_text(page,encoding='utf-8')
print(json.dumps({'report':str(D/'地面接触对照与候选修复_20261026.html'),'collected_ppo_runs':len(ppo)},ensure_ascii=False))
