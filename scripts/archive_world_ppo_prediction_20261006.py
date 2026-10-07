"""Archive verified stage outcomes without distributing data/learned weights."""
from pathlib import Path
import hashlib
import html
import json

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/world_ppo_20261006'
RAW=RESULT/'raw'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    fits={s:json.loads((RAW/f'fit_seed{s}_summary.json').read_text()) for s in [42,43,44]}
    collect={s:json.loads((RAW/f'collect_seed{s}_summary.json').read_text()) for s in [42,43,44]}
    smoke=json.loads((RAW/'smoke_docwitness_v3_summary.json').read_text())
    queue=json.loads((RAW/'r3_world_ppo_prediction_queue.state').read_text())
    assert queue['status']=='COMPLETE' and queue['prediction_gate']=='FAIL'
    for row in queue['completed']:
        assert row['summary_sha256']==sha(RAW/f"fit_seed{row['seed']}_summary.json")
    assert smoke['critic_uses_actor_norm_like_original_nonprivileged_runner']
    total=sum(x['actual_transitions'] for x in collect.values())
    assert total==11796480
    rows=[]
    for seed,f in fits.items():
        rows.append({'seed':seed,'world_H10_body_fk_rmse_mm':f['metrics']['correct']['10']['body_fk_rmse_m']*1000,
                     'ridge_H10_body_fk_rmse_mm':f['metrics']['ridge']['10']['body_fk_rmse_m']*1000,
                     'state_only_H10_body_fk_rmse_mm':f['metrics']['state_only']['10']['body_fk_rmse_m']*1000,
                     'shuffled_H10_body_fk_rmse_mm':f['metrics']['shuffled']['10']['body_fk_rmse_m']*1000,
                     'paired_reference_relative_gain_and_CI':f['paired_H10_gate'],
                     'prediction_gate':f['prediction_gate'],'actual_resets':collect[seed]['reset_count']})
    result={'status':'COMPLETE_STAGE_STOPPED_AT_PRE_REGISTERED_GATE',
            'prediction_gate':'FAIL','innovation_validated':False,
            'formal_PPO_updates':0,'five_arm_control_ablation_executed':False,
            'new_MuJoCo_transfer_executed':False,
            'diffusion_references_still_tracking_target':True,'extra_IK':False,
            'actual_unique_warmup_transitions':total,'actual_warmup_resets':sum(x['reset_count'] for x in collect.values()),
            'warmup_seed_count':3,'prediction_model_count':9,'aux_updates_per_predictor':1000,
            'diagnostic_only_correct_v3':{'envs':16,'steps':48,'PPO_updates':2,'aux_updates':1,
                                         'actual_reset':0,'reset_and_timeout_fixtures':True,
                                         'five_arm_initial_function_error':smoke['five_arm_initial_function_errors']},
            'v1_failed_and_v2_critic_mismatch_preserved':True,
            'parameters':{'base_actor_critic':1044027,'added_actor_critic_latent_columns':131072,
                          'encoder':85248,'three_prediction_heads':256406,
                          'P_cap_history_deploy_and_PPO':1260347,
                          'W_deploy':1260347,'W_PPO':1175099,'W_aux':341654,'W_all':1516753},
            'rows':rows,'conclusion':'Action-conditioned predictions improve over the future-action-zero model but not the best ridge. Current candidate fails its gate; this does not test or disprove eventual PPO control benefit.',
            'limitations':['Old actor already saw all 66 reference motions; grouped predictor testing is not unseen policy generalization.',
                           'H10 FK is root-frame URDF kinematics, not future world pose or entity collision success.',
                           'Contact labels are ground force>5N. AP should be read against per-foot prevalence.',
                           'Legacy raw root_velocity_rmse mixes linear/angular units; do not interpret as m/s. New code separates channels.',
                           'raw eligible_reset_free_windows count metadata corrected separately, metrics unchanged.',
                           'Per-update optimizer/gradient traces not stored in full predictor artifacts; audit used source, progress, finite weights, hashes and completion.',
                           'Only bounded diagnostic PPO updates; no 500-update control outcomes or new sim-to-sim claims.'],
            'artifact_distribution':'Only code, report, metadata and logs archived. Raw motion/windows/weights remain server-side.',
            'provenance':{f.name:sha(f) for f in RAW.glob('*') if f.is_file()}}
    (RESULT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    table=[]
    for r in rows:
        gain=r['paired_reference_relative_gain_and_CI']['ridge']
        ci=gain['95pct_ci']
        table.append(f"| {r['seed']} | {r['world_H10_body_fk_rmse_mm']:.2f} | {r['ridge_H10_body_fk_rmse_mm']:.2f} | {r['state_only_H10_body_fk_rmse_mm']:.2f} | {100*gain['paired_reference_relative_improvement']:.2f}% [{100*ci[0]:.2f}%, {100*ci[1]:.2f}%] | FAIL |")
    md='''# 世界模型＋PPO：Isaac 域内阶段验证（2026-10-06）

用户授权按既定协议实施。**本轮真实采集、模型拟合和接口诊断已完成，当前候选未通过预测门；五臂正式PPO控制消融与新的MuJoCo迁移没有启动，创新点尚未验证。**这不是对全部世界模型＋PPO方案的否定，也不是控制成功率结果。

## 方法与实际完成的工作

扩散生成的官方66条Kick参考仍是跟踪目标，经已有原始G1重定向使用。世界模型没有替代扩散生成器，也没有再做机器人IK。本轮主要实现“可部署本体历史→128维GRU表征→动作条件多步物理预测→PPO附加特征”的接口。actor的626维原观测仍包含参考运动；latent只读当前及过去92维本体／上一动作历史。未来实际动作只进入监督预测头，真实未来状态只作标签，不进入当前actor。

正式设计的五臂为原PPO、容量控制、历史控制、正确预测、错配预测；各8192环境×24步×500 PPO更新、seed42/43/44。该预算还未执行。所有臂原actor/critic与normalizer同一checkpoint，新增latent列零初始化；P-cap使用当前本体重复10次、GRU每决策从零、过去动作新增通道置零，P-history使用有序真实10帧窗口。两者只由PPO训练GRU。W由独立监督更新encoder；PPO epochs缓存同版本latent、停止PPO梯度进入世界模型。部署参数相同，总训练参数因预测头而不同；W-correct/shuffled匹配全部辅助预算。

先决预测门现已完成：三个seed各8192环境480步，由同一冻结原策略及原std采集3,932,160真实Isaac转移，合计11,796,480。无warm-up PPO学习，奖励／PD／termination／参考不变，实际reset为9/12/4。44训练、11validation、11锁定test的完整reference划分跨seed一致。所有有序历史与未来窗口在reset处断开；独立只读见证与原数组逐值核验。旧actor先前已见过66参考，因此不能说策略未见动作泛化。

每seed拟合正确、未来动作置零、错配三个预测器（共9个），相同初始化、batch1024、各1000次辅助更新。错配在0.2秒phase和当前接触模式内跨reference/episode精确置换，标签边际不变；不可匹配行对所有模型共同排除。归一化仅使用训练reference。Ridge相同历史／动作／标签，alpha仅validation选择；模型checkpoint也是validation选择。锁定test只做本轮固定模型验收，不据test改结构或门槛。

## 预测结果与停止原因

H10为未来0.2秒，FK指标是同一27DOF URDF根坐标下全身位置重建误差，不含世界坐标根运动，也不等于实体接触成功。

| seed | 世界模型FK误差/mm | Ridge误差/mm | 未来动作置零误差/mm | 相对Ridge改善及参考配对95%CI | 门 |
|---|---:|---:|---:|---|---|
'''+ '\n'.join(table)+'''

相对未来动作置零模型，正确模型参考配对改善约54.3%，说明动作条件对该预测任务有帮助；这不等于向PPO增加latent已经提高控制。相对Ridge三seed均未显著改善，也未满足预设≥10%且CI下界>0的门。因此按照启动前协议停在预测阶段，不启动约14.7亿转移的主控制训练，也不把两次诊断PPO更新冒充正式消融。

正确模型H10地面contact Brier约0.041–0.043；AP须结合左右脚实际接触比例约88.5%／71.6–72.1%解释。只监督地面接触力>5N，Kick没有实体打击目标。接触、足滑、根速度等细项原始结果全部保留；根速度旧字段混合m/s与rad/s，不能据它报告具有单一物理单位的效果。没有证明实体打击接触或语义改善。

## 接口检查与运行故障

v1诊断无实际step/summary，原日志保全。新增754宽网络零列相对原626宽网络仍有约6.4e-7浮点累积差异；严格等于0不合理。v2修改容差后真实PPO／aux更新成功，随后原runner审计发现它误用了未启用的privileged normalizer，不能算正确原价值函数续训。warm-up只用原actor，无PPO，因此数据与预测门不受此问题影响。

v3纠正critic：此Kick没有privileged观测，严格使用actor-normalized observation作为原critic输入。新鲜文档见证真实Isaac16环境48步、两次PPO更新、一次世界模型辅助更新，参数实际改变且loss有限。五臂初始action mean/value误差≤6.41e-7／3.58e-7，log-prob≤3.82e-6。世界模型在PPOepochs内逐张量冻结；独立history reset与timeout单次bootstrap fixture通过。v3真实reset0与fixture分开；真实reset断窗由warm-up数组审计覆盖。

独立Isaac Sim renderer经非阻塞UDP展示真实Isaac采样状态，计算与显示进程分离。本地6013隧道已修复，采集阶段实际检查到步300、480等更新；采集结束后最后姿态不是持续训练。新的正式控制训练若启动，仍须当前实际Kick本地连续画面Gate，不沿用旧Open-Door确认。

## 产物与证据范围

- `results/world_ppo_20261006/summary.json`：阶段结论、预算、参数账和原始元数据哈希。
- `raw/collect_seed*_summary.json`：3组真实数据文件大小/hash、有限性、reset、actor来源和配置。
- `raw/fit_seed*_summary.json`：9个预测器、3个Ridge、全部H1/5/10指标、每reference FK及CI、validation选择和loss记录。
- `raw/smoke_docwitness_v3_summary.json`：真实PPO接口更新／函数等价／reset及timeout检查；旧v2亦保留。
- `metadata_audit.json`：原eligible字段误记采样数的独立校正、真实完整窗口数和contact比例。原结果不覆盖、门与指标不变。
- 服务器 `artifacts/world_ppo_20261006/`：真实observable/labels/actions/done/reference/episode/phase/reward数组，窗口、预测权重、Ridge权重和诊断checkpoint。权重及原始衍生数组不上传GitHub。

独立见证核对24个数据数组、9个模型、窗口来源、normalizer和hash。日志记录每模型200/400/600/800/1000进度及正常退出。原拟合产物没有完整optimizer或每步gradient trace，独立验收不是对每个反传kernel的逐步旁观。没有正式PPO训练reward曲线、恢复AUC或新迁移视频，不能用预测误差代替这些证据。

## 后续修改边界

当前证据只说明动作条件GRU预测没有优于同输入的线性预测基线。当前原任务轨迹与PD局部响应可能较接近线性，这是待检验的解释。若继续，应保留Ridge强基线，先研究其残差／非线性接触与扰动转移；结构选择仅使用开发validation并设置新的独立确认参考。不得反复用本轮已曝光test调到通过，也不得修改预设门后把本轮报告为PASS。只有新的预测门通过，再实施五臂匹配真实rollout PPO、恢复与clean指标、最终冻结MuJoCo迁移。

协议见 `plans/isaac_feedback_world_ppo_20261006.md`；运行文档见 `docs/world_ppo_invocation_20261006.md` 和 `docs/world_ppo_prediction_witness_20261006.md`。原F反馈与旧learned-MPC否定结果单独保留，论文基线汇报不混入此创新阶段。
'''
    report=ROOT/'reports/24_World_model_PPO_prediction_gate_20261006.md'
    report.write_text(md,encoding='utf-8')
    htmlrows=''.join(f'<tr><td>{r["seed"]}</td><td>{r["world_H10_body_fk_rmse_mm"]:.2f}</td><td>{r["ridge_H10_body_fk_rmse_mm"]:.2f}</td><td>{r["state_only_H10_body_fk_rmse_mm"]:.2f}</td><td>{r["shuffled_H10_body_fk_rmse_mm"]:.2f}</td><td>未通过</td></tr>' for r in rows)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>世界模型＋PPO阶段验证</title><style>
body{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f3f5f8;color:#182d43;margin:0}main{max-width:1080px;margin:auto;padding:44px 28px}h1{font-size:32px}h2{margin-top:34px}.card{background:white;border-radius:14px;padding:24px;margin:16px 0;box-shadow:0 3px 18px #142a4410}.bad{color:#ac4c20;background:#fff3e9;padding:8px 14px;display:inline-block;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:13px;border-bottom:1px solid #dce4ec}th{background:#eff4f8}p,li{line-height:1.8}a{color:#146ba6}small{color:#526579}.stats{display:flex;gap:25px;flex-wrap:wrap}.stats b{font-size:28px;display:block}.flow{background:#eaf2f8;padding:18px;border-radius:9px}code{font-size:13px}details{margin:12px 0}summary{cursor:pointer}
</style><main><small>独立创新阶段记录 · 2026-10-06 · Isaac Lab / PhysX</small><h1>世界模型＋PPO：阶段验证</h1><div class="bad">候选预测门未通过 · 尚未验证控制收益</div><div class="card"><p>已经完成真实数据采集、九个预测器拟合与PPO接口诊断。按照预设预测门，当前版本没有进入五臂正式训练，也没有产生新的MuJoCo迁移成绩。</p><div class="stats"><div><b>1,179.6万</b>真实共享转移</div><div><b>3</b>数据种子</div><div><b>9</b>拟合预测器</div><div><b>0</b>正式PPO更新</div></div></div>
<h2>方法链路</h2><div class="card"><div class="flow">扩散生成的66条Kick参考 → 原始G1重定向 → 真实Isaac状态与参考跟踪 → 因果历史世界模型 → PPO附加特征</div><p>参考动作仍由扩散模型提供；世界模型学习机器人运动的物理响应，没有替代参考生成。未来实际状态只作监督，未来实际动作只进预测头。训练与独立实际场景监控分离，没有增加额外IK。</p></div>
<h2>预测效果</h2><div class="card"><p>未来0.2秒的根坐标全身FK误差，单位mm，越低越好。</p><table><tr><th>种子</th><th>正确世界模型</th><th>线性Ridge</th><th>未来动作置零</th><th>错配监督</th><th>门</th></tr>'''+htmlrows+'''</table><p>动作条件带来约54%的误差下降；但正确模型与Ridge基本相当，三个种子均未满足“比最佳简单基线改善≥10%且95%CI下界&gt;0”。不能把预测精度当作PPO控制收益，也不能由此否定整个世界模型路线。</p></div>
<h2>已运行与尚未运行</h2><div class="card"><ul><li>已运行：原冻结策略三seed真实采集，每seed8192环境×480步；实际reset合计25次，窗口均已审计不跨reset。</li><li>已运行：44训练／11validation／11test参考分组；每模型1000辅助更新，归一化只使用训练分组。</li><li>已运行：修正后的16环境PPO接口48步诊断，两个PPO更新与一个辅助更新；五臂初始函数、版本冻结及reset／timeout fixture通过。</li><li>未运行：五臂×三seed×500更新的正式策略消融、恢复AUC、正式控制视频与新MuJoCo迁移。</li></ul><p>Kick只有虚拟目标，接触监督是地面力&gt;5N。旧actor已见过66参考，本次分组不是策略未见动作泛化。</p></div>
<h2>后续依据</h2><div class="card"><p>保留Ridge作为强基线，先检查非线性接触／扰动下的残差预测，并锁定新确认参考。新的预测门通过后，再进行等预算真实rollout PPO消融和冻结策略迁移。当前门槛与结果保持不变。</p><details><summary>运行问题及产物说明</summary><p>首轮严格零误差检查不适合浮点矩阵；第二轮发现未使用critic normalizer的接线问题，第三轮已按原runner修复并验收。所有旧记录保留。原完整窗口数元数据已通过独立校正补充，没有修改预测指标。</p><p>原始summary、每reference误差／CI、训练loss与哈希在仓库results/world_ppo_20261006。真实数组和权重保留服务器，不公开上传。完整运行与局限见Markdown报告。</p></details><a href="../reports/24_World_model_PPO_prediction_gate_20261006.md">完整实验记录</a> · <a href="../results/world_ppo_20261006/summary.json">结构化结果</a></div></main></html>'''
    (ROOT/'media/World_model_PPO_阶段验证.html').write_text(page,encoding='utf-8')
    print(json.dumps({'report':str(report),'summary':str(RESULT/'summary.json'),
                      'html':str(ROOT/'media/World_model_PPO_阶段验证.html')},ensure_ascii=False))

if __name__=='__main__':main()
