"""Add evidence-grounded issue summaries without changing experiments or media."""
from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

REPO = Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT = Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
DATE = '2026-10-06'
FOLDER = '复现问题汇总_20261006'
OUTPUT = REPO / 'results/reproduction_issues_20261006'


def issue(id, title, group, status, certainty, symptom, cause, remedy, result, evidence):
    return dict(id=id, title=title, group=group, status=status,
                cause_certainty=certainty, symptom=symptom, cause=cause,
                remedy=remedy, result_and_limits=result, evidence=evidence)


DREAM = [
    issue('DC01', '启动依赖、参数冲突与完成判定', '运行与配置', '已修复',
          '已定位配置或实现差异',
          '独立环境曾缺少 joblib、rich、isaac_utils 路径和 jaxlie；Kit 追加的参数与 Hydra 冲突。旧包装器在 traceback 或未完成训练时也可能返回退出码 0。',
          '依赖与入口参数问题有具体报错依据；退出码 0 单独不足以证明 PPO 已运行或训练已结束。',
          '补齐项目环境依赖；派生入口仅在 Kit 启动后清除其追加参数。联合检查实际迭代、最终模型、日志和视频，不只检查进程或退出码。',
          '启动链路恢复。1 次更新诊断只证明能启动；正式控制任务还需第 1999 轮模型和后续官方评估证据。',
          ['reports/02_dreamcontrol_reproduction.md']),
    issue('DC02', '早期缩容基线与论文训练预算不同', '运行与配置', '已按论文规模完成',
          '已定位配置或实现差异',
          '早期原 66 条 Kick 参考采用 1024 环境、2000 迭代、seed123，最终官方评估为 0%；不能作为论文默认配置的结果。',
          '论文配置为 8192 环境、2000 迭代。同迭代数下，缩容组采样量更少；重新训练还改变了 seed，并非单变量对照。',
          '保留旧结果，使用官方 66 条参考、8192 环境、2000 迭代、seed42 重新完成论文规模控制基线，按官方 1000 环境、500 帧评估。',
          '新 Kick 基线为 85.3%。不能断言旧 0% 只由环境数造成，也不能将旧结果与不同参考集的成绩直接比较。',
          ['reports/02_dreamcontrol_reproduction.md', 'reports/03_dreamcontrol_multitask_baselines.md']),
    issue('DC03', 'WebRTC/GLFW 与训练同进程时出现停滞', '运行与配置', '已分离计算与显示',
          '观测关联，底层原因未证实',
          '早期 Kick 停在第 1319 轮；Punch 首次停在约第 1618 轮。日志长时间不更新，但进程仍占 GPU；仅重连客户端未证明训练恢复。',
          '串流超时、GLFW 警告与部分停滞同时出现，支持显示与计算耦合的判断；没有完整原生栈，不把本地关机或断网认定为唯一原因。',
          '保全并验 SHA 的最新检查点，经授权断点续训；改为 headless PPO 与独立渲染进程，使用非阻塞 UDP 传递实际仿真状态抽样。',
          'Punch 从 model_1600.pt 追加 400 轮完成并评估为 87.2%。画面故障不再位于 PPO 计算路径；实际状态抽样、检查点预览和全部环境逐帧直播须分别标注。',
          ['reports/02_dreamcontrol_reproduction.md', 'reports/03_dreamcontrol_multitask_baselines.md']),
    issue('DC04', '柜体资产仍为 Git LFS 占位文件', '运行与配置', '精确官方资产已恢复',
          '已定位配置或实现差异',
          'Open-Drawer 旧尝试在场景初始化失败：cabinet.usd 仅 130 字节，未产生 PPO 迭代或模型。',
          '文件是 Git LFS 指针，不是可加载的 USD 资产；这是输入资产缺失，不是控制策略得分 0%。',
          '保留原指针、状态和失败日志，水合固定官方提交对应的 17599 字节对象并核验 SHA。16 环境 1 次更新诊断后恢复原唯一队列。',
          '不改资产物理、关节配置或奖励。随后 Open-Drawer 与 Open-Door 正常完成训练和评估，分别为 100% 与 99.0%；诊断不单独计作正式成绩。',
          ['docs/cabinet_asset_recovery_20261003.md', 'results/r3_dreamcontrol_cabinet_recovery_20261003.json', 'reports/03_dreamcontrol_multitask_baselines.md']),
    issue('DC05', '已完成训练后的默认视频评估 native malloc 异常', '运行与配置', '评估已恢复，根因待查',
          '具体原因未证实',
          'Sit、Bimanual-Pick、Ground-Pick 已保存最终模型，但默认视频评估在初始化出现 malloc(): invalid size (unsorted)。Sit 同参数重试仍失败。',
          '错误发生在原生初始化路径，尚无证据定位到具体库调用或显存不足；未产生正式成功率的尝试不能记为 0%。',
          '备份模型、失败日志和哈希；使用官方 play_eval.py 的 --disable_fabric，以 USD 场景 I/O 替代 Fabric，进行一次明确记录的评估适配。',
          '同一模型、参考、PPO/奖励/物理及 1000 环境、500 帧预算保留，正常退出并得到 100%、99.9%、38.1%及视频。运行路径与默认不同，不能称完全相同运行配置。',
          ['docs/sit_usd_video_eval_20261001.md', 'docs/bimanual_eval_recovery_20261002.md', 'docs/ground_pick_eval_recovery_20261002.md', 'reports/03_dreamcontrol_multitask_baselines.md']),
    issue('DC06', '人体生成质量筛查与完整复现范围', '评价与数据边界', '筛查边界已明确',
          '指标或协议边界已核实',
          '六任务展示来自作者发布的人体轨迹：342 条原始轨迹，18 段固定样本视频；下游控制使用官方已处理参考，并非全部重新生成。',
          '本次穿地、近地滑动和骨段变化是探索性代理，可重叠；未测接触力、关节力矩或动力学稳定性，不能据此计算完整物理失败率。',
          '六任务统一导出逐样本指标与阈值；说明论文在 G1 重定向后、RL 前还有方向、骨盆高度和场景碰撞等启发式筛选。',
          '342 条结构有效不等于动力学通过。原 66 条 Kick 是生成与处理后的 RL 跟踪参考，不能当成扩散模型的 66 条输入示例；下游控制完成不等于从生成到实机的完整复现。',
          ['reports/15_human_generation_quality_and_paper_filtering_20261002.md', 'results/human_generation_quality_20261002/summary.json']),
    issue('DC07', '官方高成功率与自然动作、实体接触并非同一指标', '评价与数据边界', '按原判据解释',
          '指标或协议边界已核实',
          'Sit 为 100%、Bimanual-Pick 为 99.9%，但代表画面有机器人位于座椅或桌面上方/上面；仅凭评分不能验收自然坐姿或双手抓握搬运。',
          'Kick 按右脚踝在短时窗接近参考目标小于 0.1m 计分；Sit 按交互转换时骨盆距目标小于 0.05m 计分。它们不直接统计持续实体接触。',
          '将官方任务分数与动作语义、持续座椅接触、相对抬物及打击接触分开列示。纠正早期把默认物体高度误当 Kick 目标高度的解释。',
          'Sit、Jump、Bimanual 等单参考任务也不能支持未见动作泛化结论；官方分数保留，未测或未通过的行为判据另列。',
          ['reports/03_dreamcontrol_multitask_baselines.md', 'reports/02_dreamcontrol_reproduction.md', 'reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md']),
    issue('DC08', '地面取物：训练曲线改善但任务效果不足', '效果与迁移', '有效负面结果已保留',
          '负面结果已核实，成因未定位',
          'Ground-Pick 为 38.1%；Ground-Pick-Top 正常训练和评估后为 0%。后者不是初始化失败，也不是未完成任务。',
          '两项 reward 前/末 100 轮均值分别由 −10.108 到 4.805、−9.759 到 8.051；value loss 下降，末期回合约 499 步，仍未说明抓取目标完成。',
          '保留模型、原始 TensorBoard、正式评分和失败视频，联合解释奖励、回合长度与任务指标，不用提高 reward 替代抓取验收。',
          '两项是 11/11 控制任务归档中的低分/失败结果；未定位唯一行为原因，不能称全部任务效果通过，也不为追求高分重复正常完成的试验。',
          ['reports/03_dreamcontrol_multitask_baselines.md']),
    issue('DC09', 'MuJoCo 接口核验通过，但部分任务迁移失败', '效果与迁移', '迁移局限已保留',
          '负面结果已核实，成因未定位',
          'Jump 约 2.9 秒摔倒；Sit 达参考目标但无持续座椅接触；Pick 67 条与 Bimanual 1 条均完整 10 秒，却未满足相对抬升 5cm。',
          '各任务的 30 个观测—动作快照及共 70 条参考插值核验通过。Sit 训练座椅与参考目标不对齐；隐式/显式 PD、惯量/碰撞网格和质量随机化仍有未统一或未完全核实的差异。',
          '冻结原 actor 与 normalizer，保留真实碰撞和正负接触夹具结果；将原判据代理、不摔倒、持续接触及相对抬物分别统计，不脚本带动物体。',
          '接口正确不等于可靠迁移。单参考无自动重置口径不能与 Isaac 的 1000 环境成功率直接比较；仅六任务已有 MuJoCo 测试，其余五任务尚无迁移成绩。',
          ['reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md', 'results/sim2sim_completed_20261002/summary.json']),
    issue('DC10', '扰动测试的设定摩擦未实际生效', '运行与配置', 'V2 实际接触参数已核验',
          '已定位配置或实现差异',
          'Kick/Punch 扰动测试 V1 只降低地面摩擦，实际接触摩擦仍为 1.0；条件间质量缓存恢复也需修正。',
          '该接触组合采用地面与脚部摩擦参数中的较大值，仅改地面数值没有施加预定扰动。',
          '保全 V1，修正双方摩擦与条件间质量恢复后执行 V2；逐接触检查实际摩擦，并保留原冻结策略和评分阈值。',
          '汇报仅使用 V2。摩擦 1.5 时 Kick/Punch 稳定完成为 12.1%/10.0%，不能宣布普遍鲁棒性；固定打击靶也不等于可移动沙袋、对手或实机。',
          ['reports/11_Kick_Punch_robustness_contact_20261001.md', 'results/mujoco_robust_contact/summary.json']),
]

CLOSD = [
    issue('CL01', '任务配置和 Hydra 搜索路径导致启动失败', '运行与配置', '启动适配已修复',
          '已定位配置或实现差异',
          '默认配置缺少 env_im；新外部包装器首次又在任何 PPO 更新前因 Hydra 配置目录搜索错误退出。',
          '未显式选用 README 的 closd_multitask 任务，或导入入口后配置目录定位不同；属于启动配置错误，不是策略任务失败。',
          '显式使用 env=closd_multitask，按官方绝对配置目录组合配置；固定官方源码 d8950bdb…，保留第一次失败日志与运行环境证据。',
          '配置检查与后续诊断恢复。模型、奖励、PPO 和 DiP 架构未更换；配置检查成功本身不计作论文训练或效果复现。',
          ['reports/01_closd_reproduction.md', 'docs/closd_reproduction_20261001.md']),
    issue('CL02', '动作库并行加载超时', '运行与配置', '串行加载适配可运行',
          '已定位加载阶段，底层超时原因未证实',
          '早期 9 环境并行推理在动作加载阶段等待 20 分钟后 exit124；debug=True 串行诊断只完成 299 步、均值回报约 108.40。',
          '超时发生在动作库加载阶段；未证明是 GPU 显存不足或扩散模型本身算不动。299 步没有完整逐任务统计。',
          '仅在 MotionLib 加载范围临时启用串行路径，加载后恢复 debug 设置；记录配置与退出信息，不将整个 PPO 训练改成调试模式。',
          '随后容量诊断、两组官方权重评估与补录推理可完成。早期 299 步只作诊断，不能替代正式训练、逐任务分数或生成质量验收。',
          ['reports/01_closd_reproduction.md', 'docs/closd_reproduction_20261001.md']),
    issue('CL03', '续训 4000 轮被误写成总 epoch 上限', '运行与配置', '轮数语义已修正',
          '已定位配置或实现差异',
          '首次 512 环境微调将 max_epochs=4000 作为总上限，而源检查点已是 epoch62000，执行一次更新就触发 MAX EPOCHS NUM，随后退出139。',
          '该 runner 使用总 epoch 上限；从 epoch62000 追加约4000轮必须设为目标总 epoch66000。上限错误与随后139的具体原生原因是不同问题。',
          '单独保全 failed_v1，修正总轮数后写入独立 v2 输出；不覆盖旧日志，不把一次更新计成完整4000轮。',
          'v2 保存了 epoch66000 检查点，logger 还记录到66001；但退出仍异常，不能仅凭到达目标轮次称完整验收通过。',
          ['reports/01_closd_reproduction.md']),
    issue('CL04', '最终保存后退出139，检查点存在但正常退出未通过', '运行与配置', '检查点保全，根因待查',
          '具体原因未证实',
          '旧 512 环境 v2 在保存 Humanoid_00066000.pth 后出现 MAX EPOCHS NUM，再退出139；原训练状态保留 FAILED。',
          '原日志没有完整原生栈，只能怀疑退出清理路径，未定位具体库调用；不能断定显存溢出、论文架构错误或与后续相机崩溃同源。',
          '核验216577317字节检查点与SHA，在独立进程完成固定模型评估；新隔离入口记录官方 runner 返回及显式 Gym 清理。',
          '新3072环境三次更新诊断保存62003并正常退出0，证明短程生命周期可运行；不证明旧139唯一成因，也不证明正式4000轮训练已完成。',
          ['reports/01_closd_reproduction.md', 'docs/closd_reproduction_20261001.md', 'results/r3_closd_capacity3072_20261001.json']),
    issue('CL05', '512 环境缩容微调后，多数任务严重退化', '效果与评价', '退化已确认，成因待查',
          '已定位预算差异，唯一退化原因未证实',
          '匹配512环境评估中，未微调→本地微调：REACH 100%→0%、共用 STRIKE 87%→5%、SIT 77%→7%、GET_UP 56%→69%。',
          '官方3072×32=98304帧/更新，缩容512×32=16384，交互预算约少六倍且任务覆盖/小批次结构不同。学习率2e-5核对与官方相同，不能再称已发现学习率错误。',
          '保留退化模型与匹配评估，不重跑同一缩容配置；分别完成官方已发布两组权重的4096环境评估，规划按3072环境复现剩余正式微调。',
          '样本预算、任务分布或遗忘是待验证解释，不是已证实唯一原因。官方已发布微调权重表现好，不等于本地新3072微调成功；该训练尚未启动。',
          ['reports/01_closd_reproduction.md', 'results/r3_closd_pretrained_eval_512.json', 'results/r3_closd_finetuned_eval_512.json', 'reports/13_CLoSD_official_checkpoint_eval_20261001.md']),
    issue('CL06', '实时监控停帧与本地可视验收未通过', '可视化与证据边界', '显示独立，当前验收待完成',
          '发送错误已记录，本地显示原因未完全定位',
          '旧训练约epoch62110出现ZMQ EAGAIN，停止发送后PPO仍继续。网页可连接也可能只有旧帧；2026-10-06用户反馈“画面不动或打不开”。',
          '训练进程、网页HTTP200或LIVE标签都不能证明本地收到当前运动帧。限时推理预检结束后的STALE只是没有新帧，不是正式训练停滞。',
          '计算与显示分进程，独立非阻塞UDP接收真实Isaac Gym刚体及物体状态；检查capture时间、step和姿态变化。MeshCat不称Isaac Sim WebRTC或原生Viewer。',
          '当前新3072环境、4000轮正式微调尚未启动，本地实时Gate未通过。离线视频、旧确认或检查点回放不能代替本轮真实画面确认。',
          ['reports/01_closd_reproduction.md', 'docs/closd_reproduction_20261001.md', 'results/r3_closd_local_view_attempt_20261006.json']),
    issue('CL07', '原生相机补录三次检查均在退出时139', '运行与配置', '独立状态渲染补录完成',
          '具体原因未证实',
          '三次限定30帧的原生相机检查能产帧，但都在解释器关闭时退出139；显式相机销毁及限制相机数量仍未解决。',
          '原生相机退出路径不稳定；现有证据不能证明与旧512训练退出139同源，也不能把部分帧当成功完整录像。',
          '停止完整原生相机路线并保全失败日志；关闭物理进程图形上下文，导出实际Isaac Gym状态，在独立MeshCat进程渲染。',
          '两组冻结官方权重各900帧推理正常退出0、权重SHA不变，得到8段30秒视频和1段8.27秒起身截取。它们不是原生相机录像、原4096评估录像或sim-to-sim。',
          ['docs/closd_independent_recording_20261006.md', 'reports/17_CLoSD_supplementary_videos_and_curves_20261006.md', 'results/closd_supplementary_20261006/summary.json']),
    issue('CL08', '训练 reward 上升、critic loss 下降，但正式成绩退化', '效果与评价', '曲线与效果分开解释',
          '指标边界与退化已核实',
          '旧512日志六项主要曲线各3998点（epoch62004—66001）：前/末100点reward均值32.469→103.851，critic loss 0.03043→0.00723。',
          '训练总回报、回合长度和critic拟合误差是代理，不能替代固定模型的分任务成功率；actor/PPO loss也不是越小越好。',
          '导出原TensorBoard 51项标量及六项主曲线，将正式分任务评估与训练代理并列，并明确被评估的是epoch66000检查点。',
          '曲线支持训练目标被优化，但不支持多数任务改善。新3072正式训练没有开始，因此没有该阶段reward或loss曲线；不借用旧512或DreamControl曲线代替。',
          ['results/closd_old512_training_curves_20261006/scalar_summary.json', 'reports/17_CLoSD_supplementary_videos_and_curves_20261006.md']),
    issue('CL09', '官方分数、共享计数与补录视频的口径不同', '可视化与证据边界', '分组与出处已标注',
          '指标或协议边界已核实',
          '原两组4096环境正式评估未录像；新增12环境、seed42、900帧展示来自独立冻结策略推理。未微调固定SIT环境没有产生GET_UP片段。',
          '官方STRIKE_KICK和STRIKE_PUNCH共享成功计数，不能视作两个独立成绩；日志只给两位小数。固定单环境片段不能估计群体成功率。',
          '保留原evaluation_video_verified=false；新增演示另记video_verified=true。用明确来源、条件和缺失占位呈现视频，不挑选补造起身片段。',
          '官方已发布微调/未微调结果分别为100/94/89/97%与100/85/76/56%（REACH/共用STRIKE/SIT/GET_UP）。这是权重评估复现，非本地新训练完成，不能简单平均或据单片段改写成绩。',
          ['reports/13_CLoSD_official_checkpoint_eval_20261001.md', 'results/r3_closd_official_eval4096_20261001_summary.json', 'results/closd_supplementary_20261006/summary.json']),
]

EVIDENCE_TO_COPY = {
    'DreamControl': [
        'reports/03_dreamcontrol_multitask_baselines.md',
        'reports/11_Kick_Punch_robustness_contact_20261001.md',
        'reports/14_Completed_tasks_MuJoCo_sim2sim_20261002.md',
        'reports/15_human_generation_quality_and_paper_filtering_20261002.md',
        'docs/sit_usd_video_eval_20261001.md',
        'docs/bimanual_eval_recovery_20261002.md',
        'docs/ground_pick_eval_recovery_20261002.md',
        'docs/cabinet_asset_recovery_20261003.md',
        'results/r3_dreamcontrol_cabinet_recovery_20261003.json',
    ],
    'CLoSD': [
        'docs/closd_reproduction_20261001.md',
        'docs/closd_independent_recording_20261006.md',
        'reports/13_CLoSD_official_checkpoint_eval_20261001.md',
        'reports/17_CLoSD_supplementary_videos_and_curves_20261006.md',
        'results/r3_closd_capacity3072_20261001.json',
        'results/r3_closd_pretrained_eval_512.json',
        'results/r3_closd_finetuned_eval_512.json',
        'results/r3_closd_local_view_attempt_20261006.json',
        'results/r3_closd_official_eval4096_20261001_summary.json',
        'results/closd_old512_training_curves_20261006/scalar_summary.json',
        'results/closd_supplementary_20261006/summary.json',
    ],
}

STYLE = '''
.issue-overview{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}
.issue-overview article{background:#f2f7fa;border:1px solid #d7e3eb;border-radius:12px;padding:18px}
.issue-overview h3{font-size:17px;margin:0 0 8px}.issue-overview p{font-size:14px;margin:0}
.reproduction-issue summary{display:flex;justify-content:space-between;align-items:center;gap:14px}
.reproduction-issue summary span:first-child{font-size:17px}.reproduction-issue dl{margin:18px 0 8px}
.reproduction-issue dt{font-size:13px;font-weight:750;color:#46657a;margin-top:12px}
.reproduction-issue dd{margin:4px 0;font-size:15px}.issue-certainty{font-size:13px;color:#647b8d}
.issue-evidence{font-size:12px;color:#748995;overflow-wrap:anywhere}
@media(max-width:1050px){.issue-overview{grid-template-columns:1fr}.reproduction-issue summary{align-items:flex-start}}
@media print{.reproduction-issue{break-inside:avoid}.issue-overview{grid-template-columns:repeat(3,1fr)}.reproduction-issue dd{font-size:12px}}
'''


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_markdown(paper, issues, headline):
    parts = [f'# {paper}：复现实验问题总结', f'文档更新：{DATE}。{headline}',
             '本文件只整理已有实验记录；没有新增训练或评估。已定位的配置问题、未经证实的原因及有效负面结果分别记录。']
    for x in issues:
        parts += [f"## {x['id']} · {x['title']}",
                  f"类别：{x['group']}；状态：{x['status']}；原因证据：{x['cause_certainty']}。",
                  f"**现象：** {x['symptom']}", f"**原因与证据：** {x['cause']}",
                  f"**处理：** {x['remedy']}", f"**结果与限制：** {x['result_and_limits']}",
                  '记录来源：' + '、'.join(f'`{p}`' for p in x['evidence']) + '。']
    return '\n\n'.join(parts) + '\n'


def build_section(paper, issues, headline):
    esc = html.escape
    cards = []
    for group in dict.fromkeys(x['group'] for x in issues):
        count = sum(x['group'] == group for x in issues)
        descriptions = {
            '运行与配置': '区分启动失败、显示故障、资产缺失和原生库异常；恢复措施与运行适配明确记录。',
            '评价与数据边界': '区分生成骨架、处理后参考和控制指标；代理筛查与实体行为验收分别解释。',
            '效果与迁移': '保留低分与迁移失败；reward或接口检查通过，不能替代任务行为完成。',
            '效果与评价': '训练代理指标与正式固定模型评估并列；缩容训练的退化和原因假设分开陈述。',
            '可视化与证据边界': '区分当前实时画面、独立渲染演示与原评估录像；未通过的本地验收保持未通过。',
        }
        cards.append(f'<article><h3>{esc(group)} · {count} 项</h3><p>{descriptions[group]}</p></article>')
    pieces = [f'<section id="failures" data-issue-summary="{paper}" data-updated="{DATE}">',
              '<div class="eyebrow">复现实验问题总结 · 2026-10-06</div>',
              '<h2>遇到的问题、修复证据与尚未解决的限制</h2>',
              f'<div class="note">{esc(headline)}</div>',
              '<p>问题按“现象 → 原因证据 → 处理措施 → 结果与限制”记录。初始化失败和未完成评估不记为0%；正常完成后得到0%的结果仍保留为有效负面结果。具体根因没有证据时，明确标为未证实。</p>',
              '<div class="issue-overview">' + ''.join(cards) + '</div>']
    copy_paths = set(EVIDENCE_TO_COPY[paper])
    for x in issues:
        badge = ('fail' if x['id'] in {'DC08', 'DC09', 'CL05'} else
                 'pending' if x['id'] in {'DC05', 'CL04', 'CL06'} else 'done')
        # All entries are open initially so the summary is directly available in a presentation.
        pieces += [f'<details open class="reproduction-issue" id="issue-{x["id"]}">',
                   f'<summary><span>{x["id"]} · {esc(x["title"])}</span><span class="badge {badge}">{esc(x["status"])}</span></summary>',
                   f'<p class="issue-certainty">原因证据：{esc(x["cause_certainty"])}</p><dl>']
        for label, key in [('现象', 'symptom'), ('原因与证据', 'cause'), ('处理措施', 'remedy'), ('结果与限制', 'result_and_limits')]:
            pieces.append(f'<dt>{label}</dt><dd>{esc(x[key])}</dd>')
        pieces.append('</dl><p class="issue-evidence">记录来源：')
        refs = []
        for source in x['evidence']:
            if source in copy_paths:
                refs.append(f'<a href="{FOLDER}/{paper}/证据/{source}">{esc(source)}</a>')
            else:
                # The complete source path remains traceable without bringing innovation content
                # or large private artifact directories into the presentation bundle.
                refs.append(f'<code>{esc(source)}</code>')
        pieces += ['；'.join(refs), '</p></details>']
    if paper == 'DreamControl':
        pieces.append('<div class="warning">当前11/11控制任务已归档，但Ground-Pick 38.1%与Ground-Pick-Top 0%仍是效果不足的结果。仅六任务已有MuJoCo测试；控制任务归档不等于整篇论文、完整生成流程或所有迁移效果通过。</div>')
    else:
        pieces.append('<div class="warning">2026-10-06本轮用户反馈“画面不动或打不开”：当前实时可视验收未通过，新3072环境、4000轮正式微调尚未启动。两组已发布权重的官方评估及离线演示已完成，不能替代本地新训练或实时验收。</div>')
    pieces += [f'<p><a href="{FOLDER}/{paper}/问题总结.md">下载文字版问题总结</a> · <a href="{FOLDER}/{paper}/问题清单.json">查看结构化问题与证据清单</a> · <a href="{FOLDER}/update_receipt.json">查看报告更新回执</a></p>', '</section>']
    return ''.join(pieces)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    backup = REPO / 'artifacts/report_issue_update_20261006/before'
    backup.mkdir(parents=True, exist_ok=True)
    updates = []
    specs = [
        ('DreamControl', DREAM, '18_DreamControl_reproduction_issues_20261006.md',
         '已完成11/11官方控制任务的训练与评估归档；生成、动作质量、实体接触与跨仿真验证各自保留适用范围。',
         {'archived_control_tasks': 11, 'tested_mujoco_tasks': 6}),
        ('CLoSD', CLOSD, '19_CLoSD_reproduction_issues_20261006.md',
         '两组官方已发布权重评估与独立执行演示已完成；旧512微调退化且退出异常，新3072正式微调仍未启动。',
         {'official_checkpoint_evaluations': 2, 'formal3072_started': False,
          'current_live_gate_passed': False, 'supplementary_videos': 9}),
    ]
    for paper, issues, md_name, headline, scope in specs:
        for x in issues:
            for source in x['evidence']:
                assert (REPO / source).is_file(), source
        local = ROOT / FOLDER / paper
        local.mkdir(parents=True, exist_ok=True)
        for source in EVIDENCE_TO_COPY[paper]:
            target = local / '证据' / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / source, target)
        data = dict(paper=paper, updated=DATE, kind='documentation_of_existing_experiments',
                    scope=scope, new_training_or_evaluation=False,
                    issue_count=len(issues), issues=issues,
                    source_files=[dict(path=source, sha256=sha(REPO / source))
                                  for source in sorted({p for x in issues for p in x['evidence']})])
        text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        (OUTPUT / f'{paper}.json').write_text(text, encoding='utf-8', newline='\n')
        (local / '问题清单.json').write_text(text, encoding='utf-8', newline='\n')
        markdown = make_markdown(paper, issues, headline)
        (REPO / 'reports' / md_name).write_text(markdown, encoding='utf-8', newline='\n')
        (local / '问题总结.md').write_text(markdown, encoding='utf-8', newline='\n')

        path = ROOT / f'{paper}_实验结果汇报.html'
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        original_sha = sha(backup / path.name)
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
        before = {s['id']: sha_text(str(s)) for s in soup.find_all('section') if s.get('id') != 'failures'}
        media = [(str(t), t.get('src')) for t in soup.find_all('video')]
        old = soup.find(id='failures')
        assert old is not None
        replacement = BeautifulSoup(build_section(paper, issues, headline), 'html.parser').section
        old.replace_with(replacement)
        style = soup.find(id='reproduction-issue-style')
        if style is None:
            style = soup.new_tag('style', id='reproduction-issue-style')
            soup.head.append(style)
        style.string = STYLE
        nav = soup.find('nav')
        for a in nav.find_all('a'):
            if a.get('href') == '#failures':
                a.string = ('09' if paper == 'DreamControl' else '07') + ' 复现问题总结'
        nav.find('small').clear()
        nav.find('small').string = '适合离线打开 · 文档更新：2026-10-06'
        after = {s['id']: sha_text(str(s)) for s in soup.find_all('section') if s.get('id') != 'failures'}
        assert before == after, 'An experiment/results section changed'
        assert media == [(str(t), t.get('src')) for t in soup.find_all('video')], 'Media changed'
        assert len(soup.select('#failures details.reproduction-issue')) == len(issues)
        path.write_text(str(soup), encoding='utf-8', newline='\n')
        updates.append(dict(paper=paper, report=str(path), before_sha256=original_sha,
                            after_sha256=sha(path), issues=len(issues),
                            unchanged_other_sections=True, unchanged_video_elements=True))

    split_path = ROOT / '报告拆分清单.json'
    if not (backup / split_path.name).exists():
        shutil.copy2(split_path, backup / split_path.name)
    split = json.loads(split_path.read_text(encoding='utf-8'))
    for row in split['reports']:
        row['sha256'] = sha(ROOT / row['file'])
    split['updated_at'] = datetime.now(timezone.utc).isoformat()
    split['reproduction_issue_summaries'] = {'DreamControl': len(DREAM), 'CLoSD': len(CLOSD)}
    split['subsequent_updates'] = 'Separate evidence-grounded reproduction issue summaries; scores, videos and training status unchanged.'
    split['results_changed'] = False
    split['media_duplicated'] = False
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    receipt = dict(updated_at=split['updated_at'], changes=updates,
                   no_training_or_evaluation_started=True,
                   no_scores_or_media_changed=True, private_backup=str(backup))
    (OUTPUT / 'update_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    shutil.copy2(OUTPUT / 'update_receipt.json', ROOT / FOLDER / 'update_receipt.json')
    print(json.dumps(receipt, ensure_ascii=False))


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    main()
