"""Evidence-bounded Chinese report; previous milestone reports stay intact."""
from pathlib import Path
import hashlib,html,json
import markdown
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/completion_v1';RUN=BASE/'document_run'
pipeline=json.loads((RUN/'pipeline.json').read_text())
assert pipeline['status']=='COMPLETED'
assert all(s['status']=='COMPLETED' and s['exit_code']==0 for s in pipeline['stages'])
codec=json.loads((RUN/'codec/output/verification.json').read_text())
reg=json.loads((RUN/'regression/verification.json').read_text())
service=json.loads((RUN/'curriculum/output/evaluation.json').read_text())
mask=json.loads((RUN/'expert_mask/output/evaluation.json').read_text())
training=json.loads((RUN/'training_hook/output/verification.json').read_text())
loop_path=BASE/'closure_recovery/output/verification.json'
if not loop_path.exists():loop_path=RUN/'closed_loop/output/verification.json'
loop=json.loads(loop_path.read_text())
assert codec['status']=='COMPLETED' and reg['status']=='PASS' and service['valid'] and mask['valid']
assert mask['active_experts']==2 and not training['decision']['promote']
assert loop['status']=='COMPLETED_ENGINEERING_VALIDATION'
audit_path=BASE/'audit/EXPERIMENT_AUDIT.json'
audit_note='独立实验审查尚未完成；此版本仅供核查，不是最终交付。'
if audit_path.exists():
 audit=json.loads(audit_path.read_text(encoding='utf-8'))
 assert str(audit.get('verdict',audit.get('overall_verdict'))).lower() in ('pass','warn')
 assert not audit.get('blocking_defects')
 audit_note='独立审查：'+str(audit.get('verdict',audit.get('overall_verdict')))+'；同模型家族、独立上下文，临时接受。见 `results/completion_v1/audit/`。'
rows='\n'.join(f"|{Path(r['motion_file']).parent.name}/{Path(r['motion_file']).name}|{r['successes']}/{r['attempts']}|{r['mpjpe_m_until_termination']:.6f}|{r['mpjae_rad_until_termination']:.6f}|" for r in service['per_file'])
violations=sum(r['projection']['joint_limit_violations'] for r in codec['rows'])
feedback=loop['actual_feedback_fixture'];rollback=loop['rollback_witness']
suffix_source=loop.get('terminal_target_source','当前初始化姿态；不是作者确认的站立后缀')
text=f'''# PhyGile 课程评估、262D 编解码与真实生成—执行接口报告

日期：2026-10-17。**本轮补齐独立实现的工程路径，并实际验证；完整 PhyGile 和训练后的文本动作生成能力仍未复现。** 原始作者快照、此前 500/2000 轮 PPO 结果和上一轮课程/前缀报告均保留。

## 本轮完成情况

|组件|实际完成与验证|不能据此声称|
|---|---|---|
|课程评估器|独立进程逐文件评估，物理步后/自动重置前采集；请求文件、回合数、模型哈希、激活专家数一致性检查；统一升级判据|作者完整难度标签、完整 Stage I/II 训练已复现|
|训练调用|32 环境、4 轮 PPO 的接口测试实际触发独立评估，因成功率不足保持等级 1；两专家掩码另测|4 轮接口测试提高了主模型能力，或第二专家已训练|
|262D 编解码|{codec['sequence_count']} 个实际片段往返；按名称和 URDF 固定变换映射掌心；训练集归一化|与作者未公开的坐标/旋转/差分约定逐项一致，或全部 71D 速度可无损恢复|
|真实生成—执行|{loop['generator']['parameters']:,} 参数扩散模型实际采样，经解码和关节投影后交给 Isaac/G1；真实失败检测、拒绝、重试和重放恢复|随机权重具有语言语义生成能力，或 1 秒候选已成功执行|
|真实接受与反馈|两个各 1 帧的保持初始化姿态测试，实际接受后反馈到下一次提案，并重放已接受历史|已完成多秒已训练生成、完整片段成功率或论文生成指标|

## 课程评估与训练入口

固定一个训练片段和一个测试片段，各 8 个首回合，完整保留失败检测。下表误差以米/弧度报告，均在失败处截断；短暂存活的低误差不能等同于完整成功。文件等权汇总，所有请求文件都进入分母。

|片段|完整片段成功/尝试|终止前 MPJPE（m）|终止前 MPJAE（rad）|
|---|---:|---:|---:|
{rows}

训练调用测试于第 {training['decision']['iteration']} 轮（零起始）实际触发评估，决策为 `{training['decision']['reason']}`，等级 {training['decision']['old_level']} → {training['decision']['new_level']}。它使用两个任意测试 ID 驱动接口，**不是研究难度标注**；测试训练从随机初始化开始，4 轮 PPO / 实测 80 次 Adam（`checkpoint_accounting.json`），仅检验调度和评估调用，不并入原 2000 轮主策略。

加载模型权重不会自动恢复 `_max_active_experts` 这个 Python 属性。本实现把当前激活专家数显式传给评估进程并核对；另一个 2 回合测试确实激活两名专家。第二专家未训练，这组结果只验证掩码兼容性，不用作策略分数。

升级统一要求有效评估、每文件最低成功率、最低训练轮数，再检查双误差阈值/非恶化收敛/配置超时。它是保守的自实现选择，解决公开 Runner 与 Manager 判据不一致的问题，不冒充作者确认算法。关闭收敛检测时不会执行该升级分支。旧的 EMA 与不可靠评估入口不再决定这条新路径的升级。

## 262D 约定与验证

版本：`phygile-g1-262-semantic-urdf-discrete-v1`。维度按论文式 6 划分为 3+3+1+36+174+39+4+2。刚体与关节均按名称核对，不把论文索引直接套到 Isaac 顺序。掌心通过 URDF 的左右固定掌心变换求得，没有把手腕位置直接冒充掌心。

- 根位置最大往返绝对误差：{codec['max_position_error_m']:.9g} m。
- 关节角最大往返绝对误差：{codec['max_joint_error_rad']:.9g} rad。
- URDF FK 与已有仿真参考的最大刚体位置差：{codec['max_fk_error_m']:.9g} m。
- 归一化仅拟合 {codec['normalization_training_sequences']} 个训练片段；测试片段不参与拟合。
- 原参考中检出 {violations} 个帧—关节越界值，记录并保留原值；它们不是编码误差，也没有为了往返指标而剪裁原参考。

采用明确的 canonical heading、父子局部旋转、列式 6D 旋转和前向差分。初始根位置与四元数是必要附加信息；保持首帧 pitch/roll。速度按离散轨迹重建，不宣称与物理求解器瞬时速度完全相同。12 点位置和 13 点速度属于冗余描述；解码以根运动和 29 个关节旋转重建 FK，另外报告冗余残差。未训练输出会进行显式关节与边界投影，原始输出及投影前后状态全部保留。

## 真实闭环结果与回滚

本轮随机权重、零文本条件生成器实际提出 {len(loop['proposals'])} 个候选。计划每段 50 帧（1 秒），最多尝试 2 次，总计划延长 {loop['requested_new_frames']} 帧。实际接受 {loop['accepted_attempts']} 次、拒绝 {loop['rejected_attempts']} 次，提交 {loop['committed_new_frames']} 个新帧；结果为 `{loop['outcome']}`。没有为了通过而调整容差，也没有把拒绝动作写入前缀。

末端目标来源：{suffix_source}。接受判据使用原始米制 MPJPE、完整窗口与失败标记，并单独检查末端目标。8 步线性噪声日程、0.1 m 阈值、工程边界投影均是显式诊断设置，不是已确认的作者参数。

开发中发现：直接恢复暴露的位姿/速度/缓存后，10 帧重复执行仍产生最大状态差 1.1155769825、最大刚体位置差约 0.00315432 m；这条回滚已禁用。随后改为独立进程重建相同种子的场景，重放所有已接受窗口，恢复接触与控制器演化；每次续写前核对重放状态。

首次站立补测 `closure_final` 中，即使新进程重放也出现状态差 4.238388896、位置差 0.00181043148 m，超过 5e-4；该次验证失败，保留原始记录。两次初始状态完全相同，首帧动作已有微小差异，随后接触放大。新增 cuBLAS workspace、PyTorch 确定性算法、禁用 TF32 和 PhysX enhanced determinism，并记录质量、惯量、质心、材质及关节刚度/阻尼；这些是工程确定性配置，不能归为作者原参数。修复后的单独重复测试所有状态、动作及物理属性差异为零。

本次独立修复复跑的重放检查：{rollback['repeated_rollout_frames']} 帧，状态最大差 {rollback['state_max_abs_delta']:.9g}，位置最大差 {rollback['body_max_abs_delta_m']:.9g} m。另有 {feedback['accepted_windows']} 个各 1 帧的保持姿态窗口完成真实接受与反馈，第二段实际重放第一段。它是明确标注的 self_supervised_proxy 初始化测试，不能当作文本生成或多秒动作成功率。长序列重放会增加计算开销；没有声称私有 PhysX 求解器状态已被序列化。

GMT 与生成器参数在该闭环中均冻结，闭环 PPO 更新次数为 0。PyTorch 分配器峰值 {loop['torch_peak_allocated_bytes']/2**20:.2f} MiB 仅属于此闭环主进程，不包含子进程 PhysX/渲染及其他任务，**不能作为 RL 总显存占用**。

## 可继续研究的接口与仍缺的工作

已有训练入口、真实 DistilBERT 接口、版本化编解码、独立课程服务和真实仿真事务接口。训练后的生成器入口要求研究训练用途、相同 codec schema/骨架、相同归一化、固定文本编码器版本，并拒绝 unit_test 或人工张量检查点。

仍需真实配对动作—文本数据与对应机器人数据，才能训练和评价语义生成器。还缺作者一致约定核验、可信课程标签与完整 Stage I/II 预算、真实生成分布上的 GMT PPO 适配、多种子主实验/消融、正式生成评估器及 sim-to-sim 验证。真机按用户要求不做。此前固定主策略测试成功 26/128（20.31%）是历史小规模结果，本轮没有新增正式跟踪分数。

## 复跑与原始证据

- 命令与解释边界：`docs/completion_invocation.md`。
- 独立文档运行：`results/completion_v1/document_witness/`。
- 站立边界修复复跑：`docs/terminal_recovery_invocation.md`、`results/completion_v1/recovery_document_witness/`、`results/completion_v1/closure_recovery/`；失败的 `closure_final` 和其见证保留。
- 逐阶段命令、退出码和源代码哈希：`results/completion_v1/document_run/`。
- 原始采样、目标、实际状态、动作、重放请求：闭环输出目录；大文件以发布附件提供，附 SHA-256。
- 开发失败、手动停止和旧源码版本保留在 `results/completion_v1/development/` 及 `development_versions/`；不能把两次手动停止时外层返回 0 的收据当作成功。
- 开发状态勘误：`results/completion_v1/development_status_erratum.json`。独立七阶段运行与后续确定性修复使用不同源码，分别归档为 `phygile_completion_execution_source.tar.gz` 与 `phygile_completion_recovery_source.tar.gz`。

{audit_note}

方法依据：[PhyGile 原文 v3](https://arxiv.org/html/2603.19305v3)、[固定作者跟踪源码](https://github.com/Baojch/phygile_tracking/tree/d48621620f522f5c901ef378d57db8df92d2888d)。本文报告全部新增组件为独立实现。
'''
reports=ROOT/'reports';reports.mkdir(exist_ok=True)
md=reports/'PhyGile课程评估与262D闭环补全报告.md';web=md.with_suffix('.html')
md.write_text(text,encoding='utf-8',newline='\n')
web.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PhyGile 工程补全报告</title><style>body{max-width:1040px;margin:42px auto;padding:0 24px;font:17px/1.8 system-ui;color:#17202a}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #d7dde3;padding:10px;text-align:left}th{background:#edf3f8}code{background:#f2f4f6;padding:2px 4px}a{color:#1261a0}</style>'+markdown.markdown(text,extensions=['tables','fenced_code'])+'</html>',encoding='utf-8',newline='\n')
build={'reports':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (md,web)},
       'pipeline_sha256':hashlib.sha256((RUN/'pipeline.json').read_bytes()).hexdigest(),
       'closed_loop_primary':str(loop_path.relative_to(ROOT)),
       'closed_loop_sha256':hashlib.sha256(loop_path.read_bytes()).hexdigest()}
(BASE/'report_build.json').write_text(json.dumps(build,indent=2)+'\n')
print(json.dumps(build,ensure_ascii=False))
