# 世界模型＋PPO：Isaac 域内阶段验证（2026-10-06）

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
| 42 | 5.76 | 5.75 | 12.62 | -0.27% [-4.08%, 3.36%] | FAIL |
| 43 | 5.85 | 5.72 | 12.78 | -2.49% [-8.93%, 2.68%] | FAIL |
| 44 | 5.92 | 5.73 | 13.00 | -3.12% [-7.93%, 1.27%] | FAIL |

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
