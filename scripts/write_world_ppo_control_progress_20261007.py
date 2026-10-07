"""Build a separate innovation progress page; leave teacher baselines intact."""
import json,html
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'results/world_residual_ppo_20261007/control_progress.json'
old=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
value={**old,'updated_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Residual world-model representation with real Isaac Lab PPO; Kick only',
 'local_view_user_confirmed':True,'local_view_url':'http://127.0.0.1:6013/live',
 'transport':'SSH TCP MJPEG; not native WebRTC',
 'prediction_gate_seeds':[42,43,44],'prediction_gate_passed':True,
 'diagnostic_five_arms_complete':True,'diagnostic_transitions':61440,
 'capacity_complete':True,'capacity_saved_tensor_checks_passed':43,
 'capacity_transitions':786432,'eval_interface_status':'PASS_WITH_EVIDENCE_LIMITS',
 'eval_interface_transitions':15360,'eval_independent_trace_checks_per_arm':19,
 'formal_training_status':old.get('formal_training_status','READY_AWAITING_DOCUMENTED_LAUNCH'),
 'formal_runs_planned':15,'formal_arms':['P0','P-cap','P-history','W-correct','W-shuffled-original'],
 'formal_seeds':[42,43,44],'envs':8192,'rollout_steps':24,'additional_updates':500,
 'training_transitions_planned':1474560000,'evaluation_transitions_planned':15360000,
 'innovation_validated':False,'extra_IK':False,'teacher_baseline_reports_modified':False,
 'physical_trace_is_not_video':True}
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
arms=[('P0','原PPO','原626维观测，保留参考动作与原策略起点'),
 ('P-cap','容量对照','加入同容量编码器，当前本体输入重复，无有序历史监督'),
 ('P-history','历史对照','有序本体历史编码器，只接受PPO梯度，不训练未来预测'),
 ('W-correct','正确世界模型','残差未来状态/接触监督，冻结特征参与每轮PPO'),
 ('W-shuffled-original','错配监督对照','保持辅助预算，分层错配未来残差目标')]
rows=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in arms)
status=html.escape(value['formal_training_status'])
partial_path=ROOT/'results/world_residual_ppo_20261007/partial_control_results.json'
partial_section=''
if partial_path.exists():
 partial=json.loads(partial_path.read_text(encoding='utf-8'))
 partial_rows=''.join(f"<tr><td>{html.escape(x['arm'])}</td><td>{x['seed']}</td><td>{100*x['first_episode_success']:.1f}%</td><td>{100*x['first_episode_failure']:.2f}%</td><td>{x['recovery_RMSE_AUC_rad_s']:.4f}</td><td>{x['clean_recovery_RMSE_rad']:.4f}</td></tr>" for x in partial['rows'])
 partial_section=f'''<section><h2>当前已完成结果 · 尚未作最终判断</h2><p>已完成训练{partial['completed_train_runs']}/15组、最终评估{partial['completed_eval_runs']}/15组。北京时间快照：{html.escape(partial['observed_beijing'])}。</p>
<table><thead><tr><th>组别</th><th>种子</th><th>首回合成功</th><th>首回合失效</th><th>受扰恢复积分<br>rad·s，越低越好</th><th>无扰恢复误差<br>rad，越低越好</th></tr></thead><tbody>{partial_rows}</tbody></table>
<p>全部为500追加更新后的固定新11参考评估；各参考等权，首回合失效后的帧使用预定2rad插补。0%表示本协议下任务未达成，不能把正常退出记成执行故障；误差较低也不能单独证明踢击任务完成。</p>
<p>本次包含新参考、固定初始phase与受扰分组，与旧官方66参考85.3%不是同配置。三种子和错配对照尚未齐全，未进行最终控制Gate判定，不据中途成绩调整预算、方法或阈值。当前产物包含场景轨迹，尚非已核验的最终评估视频。</p></section>'''
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>世界模型＋PPO · 控制消融进度</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f4f8;color:#16263e;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:1150px;margin:auto;padding:36px 26px}}header{{background:#142d4e;color:white;border-radius:18px;padding:30px}}
h1{{margin:0;font-size:30px}}h2{{font-size:23px;margin:26px 0 12px}}p{{margin:10px 0}}section{{background:white;border-radius:14px;padding:24px;margin-top:20px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:22px}}.card{{background:#eaf5ee;border-radius:12px;padding:18px}}.pending{{background:#fff3dd}}strong{{font-weight:650}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:11px;border-bottom:1px solid #dce3eb}}th{{background:#f0f4f8}}
a{{color:#12619b}}header a{{color:#bde5ff}}small,.muted{{color:#587086}}.flow{{border-left:4px solid #236fa5;padding-left:20px}}code{{font-family:ui-monospace,monospace;background:#f0f3f7;padding:2px 5px}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr}}table{{font-size:14px}}main{{padding:16px}}}}
</style><main><header><h1>世界模型＋PPO · 实际控制消融</h1><p>保留扩散生成的Kick动作参考，以真实Isaac Lab运动跟踪与受扰恢复验证模型作用。</p>
<p><a href="http://127.0.0.1:6013/live">实际仿真画面</a> · <a href="World_model_PPO_新动作预测确认.html">新动作预测确认</a> · <a href="../reports/27_World_model_PPO_runtime_control_ablation_20261007.md">完整运行说明</a></p></header>
<div class="cards"><div class="card"><strong>预测确认 · 3 / 3种子通过</strong><p>新11参考上预测误差改善；不是控制收益。</p></div>
<div class="card"><strong>8192环境容量 · 通过</strong><p>保存张量43项独立核验；本地画面已确认。</p></div>
<div class="card pending"><strong>正式控制比较</strong><p>{status}</p><small>创新点尚未验证；没有提前宣布效果。</small></div></div>
<section><h2>方法与优化顺序</h2><div class="flow"><p><strong>OmniControl人体生成 → 原PyRoki重定向 → G1参考轨迹 → Isaac Lab真实执行</strong></p>
<p>策略观察当前状态、原参考信息和可用历史特征，输出原29维动作。PPO使用真实rollout的奖励、价值与优势，按原裁剪目标更新actor/critic。</p>
<p>W臂先固定世界模型特征完成整轮PPO，再利用该轮真实执行数据做一次辅助预测更新。辅助模型不能读取未来标签来决定当前动作；没有模型想象奖励。</p></div>
<p>不追加IK，不改变原奖励、PD或终止条件。GPU0负责串行训练，GPU1负责独立渲染；网页故障不会阻塞计算。</p></section>
<section><h2>五臂同预算比较</h2><table><thead><tr><th>臂</th><th>作用</th><th>接口与监督</th></tr></thead><tbody>{rows}</tbody></table>
<p>seed42/43/44，各8192环境 × 24步 × 500追加PPO更新；15项共14.746亿真实训练转移。所有臂恢复同一原Kick2000轮模型、Adam状态和学习率，使用原66参考。属于匹配微调实验，不是从零完整论文复现。</p>
<p>只评最后update500；固定新11参考，每项2048环境 × 500帧，共1536万评价转移。无测试反馈调参或模型挑选。此11参考已做预测Gate，不称端到端完全未见测试。</p></section>
{partial_section}
<section><h2>如何判定控制收益</h2><p>受扰组在第50–59步受到躯干世界Y方向±80N推力。主指标为第60–109步27关节RMSE积分；先按参考汇总，再11参考等权。首episode结束后的帧插补固定2rad，所有初始环境计入分母，自动重置不能掩盖失败。</p>
<p>三个种子各自要求：W相对原PPO、容量、历史对照恢复改善至少10%且95%置信区间下界大于0；优于错配监督；无扰动误差非劣、任务成功保持、跌倒比例不增加。配对bootstrap以参考为单位，不以并行环境伪造样本量。</p>
<p>主门通过后还需两项解释消融，然后冻结到MuJoCo。主门失败则停止当前候选。Kick虚拟目标成功不能替代实体打击接触或跨动作语义结论。</p></section>
<section><h2>产物与运行记录</h2><table><tr><th>产物</th><th>用途</th></tr><tr><td>model_update0500.pt</td><td>冻结最终策略、世界模型、normalizer与优化器，服务器保全</td></tr>
<tr><td>TensorBoard / updates.jsonl</td><td>reward、value/surrogate loss、辅助loss、交接KL、真实步数与耗时</td></tr><tr><td>actual_first_episode_trace.npz</td><td>真实执行、终止、误差、力与抽样场景；不是渲染视频</td></tr>
<tr><td>summary.json / control_gate.json</td><td>固定最终统计及三种子控制门，保留失败结果</td></tr></table>
<p>系统预检与失败日志保存在results/world_residual_ppo_20261007/raw。首次评估异常被清理遮蔽并卡住；v2打印出固定清单与目录加载顺序不一致。v3只按文件名映射索引，未重排动作或修改科学设置，两臂各64×120接口已自然完成，19项独立轨迹检查和逐参考统计重算通过。</p>
<p>接口核验保留证据范围：完整力张量、逐帧参数哈希、成功事件增量，以及未发生的timeout/root失败路径未获全量独立实证。失败预检不记0分，接口通过不等于控制效果通过。</p></section>
<p class="muted">状态为检查时快照 · 单独的创新探索页，DreamControl与CLoSD教师基线报告保持独立。</p></main></html>'''
path=ROOT/'media/World_model_PPO_控制消融进度.html';path.write_text(body,encoding='utf-8')
print(json.dumps({'page':str(path),'status':value['formal_training_status']},ensure_ascii=True))
