# 条件驱动人体运动生成：Baseline 复现、创新验证与最终实验完整执行报告

**版本：R2**  
**日期：2026-09-24**  
**用途：** 直接交给 Codex 在服务器执行。  
**研究主线：**

\[
\boxed{
\text{Condition}
\rightarrow
\text{Conditional Human Motion}
\rightarrow
\text{Physics / Contact Refinement}
\rightarrow
\text{Robot-space Reference}
\rightarrow
\text{Standard PPO Tracking}
\rightarrow
\text{Execution Feedback}
\circlearrowleft
}
\]

论文核心目标仍然是**条件驱动人体运动生成**。  
PPO motion tracking 是标准执行器和反馈采集器，不作为创新点。

---

# 1. 当前三个创新点

## I1 — Embodiment-Consistent Constraint-Efficient Generation

目标：

- 保持强约束生成能力；
- 降低高约束 test-time optimization 成本；
- 保持语义与局部运动质量；
- 减少 human-space 条件在 G1 retarget 后发生漂移。

核心问题：

\[
\text{Human-space valid}
\not\Rightarrow
\text{Robot-space valid}.
\]

---

## I2 — Task- and Contact-Preserving Physics Refinement

目标：

- 改善 penetration、floating、foot skating、contact error；
- 扩展到 sit、step/stair、support 等 rigid contact-rich motion；
- physics refinement 不能为了稳定性删除关键动作事件或改写原任务。

核心问题：

\[
\text{Physics improvement}
\not\Rightarrow
\text{Task preserved}.
\]

---

## I3 — Failure-Attributed Execution-Feedback Motion Data Iteration

目标：

- 用 Isaac Lab rollout 得到真实执行反馈；
- 区分：
  - Condition / Embodiment Error
  - Physics / Contact Error
  - Valid-Hard
  - Mastered
- 把不同失败定向路由到 I1、I2 或继续 PPO 训练；
- 提升：

\[
\boxed{\text{Task-Preserving Robot Executability}}
\]

核心原则：

\[
\boxed{\text{Tracking Failure}\neq\text{Bad Motion}}
\]

---

# 2. 整体执行顺序

Codex 必须严格按以下顺序执行：

\[
\boxed{
\text{Phase A: Server Audit}
\rightarrow
\text{Phase B: Baseline Reproduction}
\rightarrow
\text{Phase C: Ours-Base}
\rightarrow
\text{Phase D: I1 Feasibility}
\rightarrow
\text{Phase E: I2 Feasibility}
\rightarrow
\text{Phase F: Standard PPO}
\rightarrow
\text{Phase G: I3 Feasibility}
\rightarrow
\text{Phase H: Full Integration}
\rightarrow
\text{Phase I: Final Experiments}
}
\]

任何阶段未通过 GO/NO-GO gate，都不要继续后续阶段。

---

# 3. Phase A — 服务器审计

第一步不改任何算法。

生成：

```text
reports/00_server_audit.md
```

必须记录：

```text
OS
CPU
RAM
GPU
GPU memory
nvidia-smi
CUDA runtime
CUDA toolkit
Python
conda / mamba
PyTorch
Isaac Sim
Isaac Lab
Isaac Gym
MuJoCo
磁盘空间
已有 dataset
已有 checkpoint
已有 repo
Git status
```

执行：

```bash
nvidia-smi
nvcc --version || true
python --version
conda env list || true
df -h
```


# 3.1 本地可视化与远程监控是强制验收项

服务器训练不能只留下日志。Codex 必须保证用户在**本地电脑**可以观察训练过程，并且能够查看服务器生成的视频。

整个远程可视化分为三层：

\[
\boxed{
\text{Live Simulation View}
+
\text{Training Curves}
+
\text{Recorded Videos}
}
\]

其中：

- Live Simulation View：看 humanoid 当前在 Isaac Lab / Isaac Sim 中如何执行动作；
- Training Curves：看 reward、success、loss、episode length、fall rate 等训练指标；
- Recorded Videos：定期保存 MP4，并自动同步到本地。

---

# 3.2 本地实时查看 Isaac Lab / Isaac Sim 仿真画面

## 推荐方案：WebRTC Livestream

对于 DreamControl 和本文 Ours 的 Isaac Lab / Isaac Sim 训练，优先配置 NVIDIA 官方 WebRTC livestream。

服务器保持仿真和 RL 计算，本地电脑只接收视频流，不在本地执行 physics。

架构：

```text
Server
  Isaac Sim / Isaac Lab
  GPU Simulation + PPO
        │
        │ WebRTC
        ▼
Local PC
  Isaac Sim WebRTC Streaming Client
```

Codex 必须先检查当前 Isaac Sim / Isaac Lab 版本是否支持以下任一方式：

```text
LIVESTREAM=2
--livestream 2
isaac-sim.streaming.sh
isaacsim isaacsim.exp.full.streaming --no-window
```

**实际命令必须以当前安装版本 `--help` 和官方 API 为准，不允许盲目复制不同 Isaac Sim 版本的参数。**

对于基于 `AppLauncher` 的 Isaac Lab 脚本，优先尝试：

```bash
LIVESTREAM=2 <原训练命令>
```

或：

```bash
<原训练命令> --livestream 2
```

若项目使用 standalone Isaac Sim streaming app，则根据安装方式启动 streaming experience。

---

# 3.3 本地客户端要求

本地电脑安装：

```text
Isaac Sim WebRTC Streaming Client
```

然后连接：

```text
<SERVER_PRIVATE_IP>
```

优先连接方式：

```text
同一局域网
VPN
Tailscale
WireGuard
```

不要把 Isaac Sim WebRTC streaming 端口直接暴露到公网。

官方文档明确指出 livestream endpoint 本身不提供认证和加密，因此如果服务器在公网：

```text
必须使用 VPN / firewall / trusted network
```

Codex 需要在：

```text
reports/00_server_audit.md
```

增加：

```text
Remote Visualization
- WebRTC available: YES/NO
- NVENC available: YES/NO
- Server private/VPN IP
- Livestream mode
- Required ports
- Local client test result
```

注意：

> 没有 NVENC 的 GPU 不适合使用 Isaac Sim WebRTC livestream。

如果当前训练 GPU 不支持或 livestream 与训练冲突，则使用后面的“周期性录制 + 本地同步”方案。

---

# 3.4 Livestream 性能策略

不要为了实时画面显著拖慢正式训练。

采用两种模式：

## Debug / Visual Mode

用于：

- 环境检查；
- reward 检查；
- reference 检查；
- failure attribution 人工核查。

特点：

```text
少量 environments
开启 rendering
开启 livestream
```

例如：

```text
num_envs = 1–16
```

---

## Full Training Mode

用于正式大规模 PPO：

```text
hundreds / thousands of parallel environments
headless
```

此时优先：

```text
关闭持续 livestream
开启周期性 video recording
TensorBoard 持续监控
```

推荐不要让正式大规模训练因为持续 GUI rendering 明显减速。

因此：

\[
\boxed{
\text{Debug: Live View}
\qquad
\text{Full Training: Headless + Periodic Video}
}
\]

---

# 3.5 本地实时查看训练曲线

无论是否开启 livestream，都必须能在本地查看 PPO 训练指标。

优先使用：

```text
TensorBoard
```

服务器端：

```bash
tensorboard \
  --logdir <RUN_LOG_ROOT> \
  --host 127.0.0.1 \
  --port 6006
```

本地电脑建立 SSH tunnel：

```bash
ssh -N \
  -L 6006:127.0.0.1:6006 \
  <USER>@<SERVER>
```

然后本地浏览器：

```text
http://127.0.0.1:6006
```

本地至少要实时看到：

```text
episode reward
tracking reward
task success
execution success
fall rate
episode length
policy loss
value loss
entropy
learning rate
```

I3 外循环还需要额外记录：

```text
motion validity
Valid-Hard count
Mastered count
Condition/Embodiment Error count
Physics/Contact Error count
repair success
regeneration success
held-out execution success
outer iteration index
```

若原 baseline 使用 WandB，可以保留官方 WandB；但不得因为接入监控修改 baseline 核心算法。

---

# 3.6 服务器训练必须定期自动录像

对于 Isaac Lab / Isaac Sim 训练，必须保存可回看的 rollout 视频。

Isaac Lab 原生支持训练时录制视频。

Codex 必须先检查训练脚本：

```bash
<train-command> --help
```

若支持标准 Isaac Lab 参数，正式训练建议使用类似：

```bash
--video
--video_length <N>
--video_interval <N>
```

headless 模式录制时，如当前版本要求，还需：

```bash
--enable_cameras
```

例如概念形式：

```bash
./isaaclab.sh -p \
scripts/reinforcement_learning/rsl_rl/train.py \
--task=<TASK> \
--headless \
--video \
--video_length 300 \
--video_interval 10000
```

**具体 flag 以当前 DreamControl / Isaac Lab 版本实际支持参数为准。**

Isaac Lab 官方训练视频通常会保存在对应 run/checkpoint 日志目录的：

```text
videos/train/
```

或项目定义的视频目录。

Codex 必须在实验报告中记录实际路径。

---

# 3.7 Evaluation 视频必须单独保存

训练视频用于观察学习过程。

最终评价还必须单独运行 evaluation rollout。

例如 DreamControl 官方：

```bash
./isaaclab.sh -p \
scripts/reinforcement_learning/rsl_rl/play_eval.py \
--task=Isaac-Motion-Tracking-Kick-v0 \
--headless \
--video \
--num_envs 1000
```

对于 Ours 也必须建立统一：

```text
scripts/eval_and_record.py
```

或在现有 evaluation script 上增加非侵入式 video output。

每个最终 checkpoint 至少保存：

```text
best checkpoint evaluation video
final checkpoint evaluation video
representative success cases
representative failure cases
```

I3 还必须保存：

```text
Before Iteration
After Iteration
```

成对视频，方便判断数据迭代是否只是把动作变简单。

---

# 3.8 本地电脑自动同步服务器视频

服务器上的 MP4 不能只留在服务器。

推荐让**本地电脑主动拉取**，避免服务器保存本地私钥。

本地 Windows / WSL / Linux 可以使用：

```bash
rsync -avz \
  <USER>@<SERVER>:<REMOTE_VIDEO_ROOT>/ \
  <LOCAL_VIDEO_ROOT>/
```

只同步新增视频：

```bash
rsync -avz --ignore-existing \
  <USER>@<SERVER>:<REMOTE_VIDEO_ROOT>/ \
  <LOCAL_VIDEO_ROOT>/
```

训练期间可运行：

```bash
while true; do
  rsync -avz --ignore-existing \
    <USER>@<SERVER>:<REMOTE_VIDEO_ROOT>/ \
    <LOCAL_VIDEO_ROOT>/
  sleep 60
done
```

如果本地 Windows 没有 rsync，可以使用：

```bash
scp
```

或：

```text
WinSCP
VS Code Remote SSH
SFTP
```

但优先提供可自动化的同步脚本。

Codex 需要生成本地模板：

```text
scripts/local_sync_videos.sh
```

内容中保留：

```text
SERVER
USER
REMOTE_VIDEO_ROOT
LOCAL_VIDEO_ROOT
```

四个可配置变量。

---

# 3.9 视频目录必须统一

所有实验的视频统一放到：

```text
artifacts/videos/
├── baseline/
│   ├── closd/
│   └── dreamcontrol/
├── ours/
│   ├── base/
│   ├── i1/
│   ├── i2/
│   ├── i3/
│   └── full/
└── ablations/
```

每个视频名称必须包含：

```text
method
task
seed
iteration
checkpoint
status
```

例如：

```text
ours_kick_seed0_iter2_best_success.mp4
ours_kick_seed0_iter2_failure_003.mp4
dreamcontrol_kick_seed0_final.mp4
```

---

# 3.10 本地可视化验收 Gate

进入大规模正式训练前必须通过：

## Gate V1 — Live View

至少完成一次：

```text
Server Isaac Lab
→ WebRTC livestream
→ Local PC successfully displays simulation
```

如果硬件/网络不支持 livestream，必须在报告中说明原因，并用周期录像替代。

## Gate V2 — TensorBoard

本地浏览器能实时看到服务器训练曲线。

## Gate V3 — Video

服务器生成 MP4 成功。

## Gate V4 — Local Sync

本地电脑能自动同步并播放服务器生成的视频。

输出：

```text
reports/00_remote_visualization_validation.md
```

只有这四项完成后，才进入长时间训练。

---

# 3.11 对 CLoSD 的特殊说明

CLoSD 原生使用 Isaac Gym，不是 Isaac Lab。

因此 CLoSD 不要求强行实现 Isaac Lab WebRTC。

CLoSD baseline 保持原论文环境：

```text
Isaac Gym
```

本地观察方式可以使用：

```text
Isaac Gym GUI via remote desktop/X forwarding
```

或：

```text
官方 recording / Blender visualization
```

更重要的是保存官方 evaluation output 和 recorded motion。

不要为了统一可视化方式把 CLoSD 移植到 Isaac Lab。

---

# 3.12 对 DreamControl 与 Ours 的强制要求

DreamControl 和 Ours 都基于 Isaac Lab / Isaac Sim，因此必须优先做到：

```text
Local Live View
+
TensorBoard
+
Periodic MP4
+
Local Video Sync
```

这四项在服务器实验开始前配置完成。

---

---

# 4. Phase B — 系统级 Baseline 复现

最终论文外部主表只保留完整系统：

\[
\boxed{
\text{DreamControl}
\quad
\text{CLoSD}
\quad
\text{Ours}
}
\]

RoboGhost 暂不作为第一阶段必跑 baseline，可作为后续可选扩展。

---

# 5. Baseline 1 — CLoSD

## 5.1 平台和算法

CLoSD 使用：

\[
\boxed{\text{NVIDIA Isaac Gym}}
\]

不是 Isaac Lab。

控制器：

\[
\boxed{
\text{PHC-based PPO Physics Controller + AMP/Imitation}
}
\]

底层 RL 使用：

```text
rl-games
PPO
```

配置中明确存在：

```yaml
ppo: true
e_clip: 0.2
```

---

## 5.2 为什么先复现 CLoSD

CLoSD 公开程度高：

- 代码公开；
- checkpoint 公开；
- Hugging Face 自动下载数据和权重；
- evaluation script 完整；
- training script 完整；
- 官方提供 Table 1 / Table 3 的复现命令。

因此最适合先验证服务器 baseline 复现能力。

---

## 5.3 Codex 执行

官方仓库：

```text
GuyTevet/CLoSD
```

执行：

```bash
git clone https://github.com/GuyTevet/CLoSD.git
cd CLoSD
git rev-parse HEAD
```

环境按官方 README。

先跑：

### Multi-task inference

```bash
python closd/run.py \
  learning=im_big robot=smpl_humanoid \
  epoch=-1 test=True no_virtual_display=True \
  headless=False env.num_envs=9 \
  env=closd_multitask \
  exp_name=CLoSD_multitask_finetune
```

### Multi-task evaluation

```bash
python closd/run.py \
 learning=im_big \
 env=closd_multitask \
 robot=smpl_humanoid \
 exp_name=CLoSD_multitask_finetune \
 epoch=-1 \
 env.episode_length=500 \
 env.dip.cfg_param=7.5 \
 env.num_envs=4096 \
 test=True \
 no_virtual_display=True \
 headless=True \
 closd_eval=True
```

### Text-to-motion evaluation

```bash
python -m closd.diffusion_planner.eval.eval_humanml \
  --external_results_file \
  closd/diffusion_planner/saved_motions/closd/CloSD.pkl \
  --do_unique
```

---

## 5.4 CLoSD Gate

CLoSD 复现成功至少满足：

- checkpoint 自动获取成功；
- Isaac Gym 启动成功；
- planner 有有效输出；
- physics controller 可执行；
- evaluation script 能产生结果；
- 至少一项官方表格指标与论文数量级一致。

输出：

```text
reports/01_closd_reproduction.md
```

---

# 6. Baseline 2 — DreamControl

## 6.1 平台和算法

DreamControl 使用：

\[
\boxed{\text{Isaac Sim 4.5 + Modified Isaac Lab}}
\]

机器人：

\[
\boxed{\text{Unitree G1}}
\]

流程：

\[
\text{Guided Human Motion}
\rightarrow
\text{G1 Retargeting}
\rightarrow
\text{Trajectory Refinement}
\rightarrow
\text{Task-specific RL}
\]

---

## 6.2 官方公开内容

已经公开：

- trajectory generation code；
- `model_humanml` 预训练生成模型；
- generated reference trajectories；
- Isaac Lab RL training code；
- Sim2Real code。

当前 README 未明确提供所有 task-specific RL 最终 policy 的统一下载入口。

因此 baseline 复现策略：

\[
\boxed{
\text{Official Reference}
\rightarrow
\text{Official RL Code}
\rightarrow
\text{Train Policy on Server}
}
\]

---

## 6.3 第一轮只选简单代表任务

不要一开始训练全部任务。

优先：

```text
Kick
Jump
```

先选其中一个。

原因：

- 不依赖复杂 object manipulation asset；
- 属于 whole-body motion；
- 与我们的生成/执行主线直接相关。

---

## 6.4 Codex 执行

官方仓库：

```text
GenRobo/DreamControl
```

安装：

```bash
conda create -n dreamcontrol python=3.10 -y
conda activate dreamcontrol
```

按官方要求安装 PyTorch、Isaac Sim、modified Isaac Lab。

---

## 6.5 先验证 trajectory generation

使用官方：

```text
model_humanml
```

并运行：

```bash
bash collect_Kick.sh
```

或：

```bash
bash collect_Jump.sh
```

确认：

```text
human reference
→ retargeted reference
→ refined robot reference
```

均成功生成。

---

## 6.6 再训练 RL

例如 Kick：

```bash
cd Training

./isaaclab.sh -p \
scripts/reinforcement_learning/rsl_rl/train.py \
--task=Isaac-Motion-Tracking-Kick-v0 \
--headless \
--device cuda:0
```

训练后：

```bash
./isaaclab.sh -p \
scripts/reinforcement_learning/rsl_rl/play_eval.py \
--task=Isaac-Motion-Tracking-Kick-v0 \
--headless \
--video \
--num_envs 1000
```

---

## 6.7 DreamControl Gate

至少满足：

- reference generation 成功；
- G1 retarget/refinement 成功；
- RL training 可稳定运行；
- checkpoint 可保存；
- play_eval 可加载；
- task success / execution metric 可输出。

输出：

```text
reports/02_dreamcontrol_reproduction.md
```

---

# 7. Baseline 总结 Gate

生成：

```text
reports/03_baseline_summary.md
```

表格：

| Baseline | Platform | Robot | Code | Checkpoint | Training | Eval | Status |
|---|---|---|---|---|---|---|---|
| CLoSD | Isaac Gym | SMPL Humanoid | ✓ | ✓ | ✓ | ✓ | |
| DreamControl | Isaac Lab | G1 | ✓ | Generator ✓ / RL需训练 | ✓ | ✓ | |

只有 baseline 都跑通，才能进入创新验证。

---

# 8. Phase C — 建立 Ours-Base

建立一个不含创新的基础 pipeline：

\[
\text{Base Conditional Generator}
\rightarrow
\text{Fixed Human-to-G1 Retargeting}
\rightarrow
\text{Standard PPO}
\]

不启用：

```text
I1
I2
I3
```

这个 Base 只用于后续消融。

---

# 9. 固定数据协议

## 9.1 HumanML3D

使用官方：

```text
train / val / test
```

不允许改 split。

用途：

- conditional generation；
- common text-conditioned evaluation；
- human motion quality。

---

## 9.2 I1 development constraints

使用固定开发集：

```text
narrow gap
low barrier
specified steps
trajectory
end-effector target
keyframe
```

这些主要用于验证 I1 是否可行，不是最终外部 baseline。

---

## 9.3 I2 contact-rich set

优先使用正式数据来源中的 rigid interaction：

```text
HUMANISE
SAMP / TeSMo-compatible scenes
```

任务：

```text
Sit
Stand-up
Step / Stair
Support
Jump / Airborne
```

禁止加入：

```text
Swimming
Fluid interaction
Complex deformable object
```

---

## 9.4 Robot execution split

固定：

```text
execution_train.json
execution_val.json
execution_test.json
```

必须包含：

```text
walk / turn
jump
kick
punch
crouch
sit / stand
upper-body + whole-body coordination
```

测试集冻结后不可修改。

---

# 10. Phase D — I1 可行性验证

## 10.1 目标

验证：

\[
\boxed{
\text{Embodiment-Consistent Constraint-Efficient Generation}
}
\]

是不是有效。

---

## 10.2 内部实验

比较：

```text
Base
Base + Constraint-Efficient Module
Base + Robot-Space Consistency
Full I1
```

不要接 I2/I3。

---

## 10.3 最小规模

```text
100–300 fixed prompts / constraints
3 seeds
```

---

## 10.4 指标

Human-space：

```text
Constraint Success ↑
Constraint Error ↓
R-Precision / Text Alignment ↑
FID ↓
Diversity
Optimization Steps ↓
Wall-clock Time ↓
```

Robot-space：

```text
Robot-space Condition Success ↑
Retargeting Drift ↓
Post-retarget Rejection ↓
End-effector Error ↓
```

---

## 10.5 I1 Gate

I1 GO 至少要求：

1. human-space constraint success 不明显下降；
2. robot-space condition success 提升；
3. retargeting drift 下降；
4. semantics 不明显退化；
5. 如果 claim 包含 efficiency，则 inference cost 确实下降。

输出：

```text
reports/04_I1_feasibility.md
```

---

# 11. Phase E — I2 可行性验证

## 11.1 目标

验证：

\[
\boxed{
\text{Task- and Contact-Preserving Physics Refinement}
}
\]

---

## 11.2 输入

冻结一批 I1 输出。

不允许为 I2 人工重新筛容易样本。

---

## 11.3 对比

```text
Before I2
I2 w/o Task Preservation
I2 w/o Contact Loss
Full I2
```

---

## 11.4 Physics 指标

```text
Penetration ↓
Foot Skating ↓
Floating ↓
Contact Error ↓
Physics Imitation Failure ↓
```

---

## 11.5 Task Preservation 指标

```text
Condition Retention ↑
Contact Event Success ↑
Event Timing Error ↓
Motion Amplitude Change ↓
Duration Change ↓
```

---

## 11.6 I2 Gate

必须满足：

\[
\text{Physics Quality}\uparrow
\]

且：

\[
\text{Task Preservation 不下降}
\]

如果 physics 更好但任务被改写：

```text
I2 = NO-GO
```

输出：

```text
reports/05_I2_feasibility.md
```

---

# 12. Phase F — 固定 Standard PPO

PPO 不是创新。

从这里开始固定：

```text
Policy Architecture
Observation
Action
Reward
PPO Hyperparameters
Domain Randomization
Training Steps
Termination
```

后续 I3 所有配置完全相同。

---

# 13. PPO 最低验证要求

固定 PPO 必须能够输出：

```text
joint tracking error
root tracking error
end-effector tracking error
contact error
fall
collision
joint limit
torque violation
task success
```

输出：

```text
reports/06_standard_ppo_validation.md
```

---

# 14. Phase G — I3 可行性验证

## 14.1 目标

不是训练更强 PPO，而是验证：

\[
\boxed{
\text{Execution Feedback}
\rightarrow
\text{Failure Attribution}
\rightarrow
\text{Targeted Motion Data Update}
}
\]

是否有效。

---

# 15. Motion Reference Experience Bank

每条 motion 保存：

```json
{
  "sample_id": "...",
  "condition": {},
  "human_motion": "...",
  "robot_reference": "...",
  "condition_metrics": {},
  "physics_metrics": {},
  "execution_metrics": {},
  "failure_label": "...",
  "iteration": 0,
  "parent_sample_id": null
}
```

---

# 16. 四类标签

```text
Condition / Embodiment Error
Physics / Contact Error
Valid-Hard
Mastered
```

处理：

## Condition / Embodiment Error

```text
→ I1 regenerate / correction
```

## Physics / Contact Error

```text
→ I2 refinement
```

## Valid-Hard

```text
→ reference 不修改
→ 继续进入下一轮 PPO 训练
```

## Mastered

```text
→ 保留一定比例用于 coverage / anti-forgetting
```

---

# 17. I3 第一项实验：Attribution Audit

先不要跑多轮。

抽取：

```text
>=100 failure samples
```

对照：

```text
Automatic Attribution
Human Inspection
Rollout Video
Condition Metrics
Physics Metrics
Execution Metrics
```

生成 confusion matrix。

如果归因效果不可接受：

```text
I3 = NO-GO
```

输出：

```text
reports/07_I3_attribution_audit.md
```

---

# 18. I3 第二项实验：小规模外循环

先跑：

```text
2–3 outer iterations
```

比较：

```text
No Iteration
I3 w/o Failure Attribution
I3 w/o Motion Bank
Full I3
```

所有配置共享：

```text
same generator
same I1
same I2
same PPO
same env steps
same seed protocol
```

---

# 19. I3 指标

每轮：

```text
Intrinsic Motion Validity ↑
Robot-space Condition Validity ↑
Physics / Contact Validity ↑
Held-out Execution Success ↑
Valid-Hard Coverage ↑
Mastered Coverage ↑
Repair Success ↑
Regeneration Success ↑
Fall / Unsafe Rate ↓
Environment Steps
GPU-hours
```

同时监控：

```text
Diversity
Motion Amplitude
Velocity Distribution
Duration
Key-event Completion
```

防止：

> 系统只是把动作变慢、变小、变简单。

---

# 20. I3 Gate

I3 GO 需要：

1. held-out execution success 提高；
2. condition validity 不下降；
3. physics validity 不下降；
4. Diversity / amplitude 不明显塌缩；
5. Valid-Hard 不被误改简单；
6. failure attribution 有可接受可靠性。

输出：

```text
reports/08_I3_feasibility.md
```

---

# 21. 可选：Generator 参数级更新

只有 routing-based I3 已经有效后才做。

比较：

```text
Routing-only I3
Routing + Lightweight Generator Fine-tuning
```

训练数据：

Positive：

```text
condition valid
physics valid
最终 robot execution success
```

Negative / constraint：

```text
真实 condition invalid
真实 physics invalid
```

禁止：

```text
Valid-Hard 作为负样本
```

只有 fine-tuning 进一步提高 executability 且不降低 semantics/diversity，才保留。

---

# 22. Phase H — 完整系统集成

三个创新均 GO 后，集成：

\[
\boxed{
I1
\rightarrow
I2
\rightarrow
\text{Standard PPO}
\rightarrow
I3
}
\]

形成：

```text
Full Ours
```

---

# 23. Phase I — 最终论文实验

最终只保留两类：

\[
\boxed{\text{Full-System External Comparison}}
\]

+

\[
\boxed{\text{Internal Ablation}}
\]

---

# 24. 最终系统级外部对比

主表：

```text
CLoSD
DreamControl
Ours
```

如果后续资源足够，再考虑 RoboGhost。

不允许把：

```text
RG-DNO + Morph + SONIC
```

拼成 baseline。

---

# 25. End-to-End Evaluation Suite

## Suite A — Common Text-conditioned Whole-body

固定：

```text
Walk / Turn
Jump
Kick
Punch
Crouch
Sit / Stand
Upper-body + Whole-body
```

---

## Suite B — Shared Task / Scene

优先：

```text
Jump
Kick
Punch
Sit
```

若方法原生支持并能对齐资产：

```text
Pick
Open Drawer
```

原生不支持则记：

```text
N/A
```

---

# 26. 完整系统主指标

共同指标：

```text
Task / Condition Success ↑
Execution Success ↑
Contact / Event Success ↑
Fall / Unsafe Rate ↓
End-to-end Latency ↓
Training / Data Cost ↓
```

对于显式 reference 系统：

```text
Robot-space Condition Error ↓
MPJPE-L / Tracking Error ↓
Post-retarget Rejection ↓
Motion Validity ↑
```

CLoSD 与 Ours embodiment 不同，不强行比较不等价 joint error。

---

# 27. 最终主消融

比较：

```text
Full w/o I1
Full w/o I2
Full w/o I3
Full Ours
```

共享：

```text
same tasks
same splits
same seeds
same PPO
same G1 assets
same training budget
```

---

# 28. I1 消融诊断

重点：

```text
Robot-space Condition Success
Retargeting Drift
Post-retarget Rejection
Text / Motion Quality
```

---

# 29. I2 消融诊断

重点：

```text
Penetration
Foot Skating
Contact Success
Task Retention
Event Timing
```

---

# 30. I3 消融诊断

重点：

```text
Held-out Execution Success
Valid-Hard Coverage
Repair / Regeneration Success
Motion Validity
Data Efficiency
```

---

# 31. I3 机制消融

额外比较：

```text
No Iteration
w/o Failure Attribution
w/o Motion Bank
Full I3
```

用来证明：

> 提升来自 failure attribution 和 motion-level experience，而不是单纯多训练几轮 PPO。

---

# 32. 外循环收敛

固定 validation benchmark。

定义：

```text
P_k = held-out execution performance
Q_k = condition / physics validity
H_k = Valid-Hard fraction beyond current capability
```

连续 M 轮满足：

\[
|P_k-P_{k-1}|<\epsilon_P
\]

\[
|Q_k-Q_{k-1}|<\epsilon_Q
\]

\[
H_k<\epsilon_H
\]

或：

```text
iteration >= K_max
```

停止。

注意：

> 这是 practical convergence，不是理论全局收敛证明。

---

# 33. 统计要求

主要训练：

```text
>= 3 seeds
prefer 5 seeds
```

报告：

```text
mean ± std
95% CI（适用时）
```

统计单位：

```text
task / prompt / trajectory
```

禁止：

> 把同一 trajectory 的每帧当独立样本。

---

# 34. 最终论文主表建议

## Table 1 — Full-System Comparison

| Method | Platform | Embodiment | Task Success ↑ | Exec Success ↑ | Contact ↑ | Fall ↓ | Latency ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| CLoSD | Isaac Gym | SMPL Humanoid | | | | | |
| DreamControl | Isaac Lab | G1 | | | | | |
| Ours | Isaac Lab | G1 | | | | | |

---

## Table 2 — Main Ablation

| Method | I1 | I2 | I3 | Condition Success ↑ | Motion Validity ↑ | Exec Success ↑ |
|---|---|---|---|---:|---:|---:|
| Full w/o I1 | × | ✓ | ✓ | | | |
| Full w/o I2 | ✓ | × | ✓ | | | |
| Full w/o I3 | ✓ | ✓ | × | | | |
| Full Ours | ✓ | ✓ | ✓ | | | |

---

## Table 3 — I3 Mechanism Ablation

| Variant | Attribution | Motion Bank | Exec Success ↑ | Valid-Hard ↑ | Validity ↑ |
|---|---|---|---:|---:|---:|
| No Iteration | × | × | | | |
| w/o Attribution | × | ✓ | | | |
| w/o Motion Bank | ✓ | × | | | |
| Full I3 | ✓ | ✓ | | | |

---

# 35. 必须输出的 Codex 报告

```text
reports/
├── 00_server_audit.md
├── 00_remote_visualization_validation.md
├── 01_closd_reproduction.md
├── 02_dreamcontrol_reproduction.md
├── 03_baseline_summary.md
├── 04_I1_feasibility.md
├── 05_I2_feasibility.md
├── 06_standard_ppo_validation.md
├── 07_I3_attribution_audit.md
├── 08_I3_feasibility.md
├── 09_full_system_integration.md
├── 10_full_system_comparison.md
├── 11_main_ablation.md
├── 12_I3_mechanism_ablation.md
└── 13_final_summary.md
```

---

# 36. 每个报告的固定格式

每份报告必须包含：

```text
Goal
Repo / Commit
Environment
Dataset
Split
Command
Config
Seed
Checkpoint
GPU
Runtime
Raw Metrics
Aggregated Metrics
Paper Metric（baseline适用时）
Difference
Error / Failure
Interpretation
GO / NO-GO
Next Step
```

---

# 37. Codex 第一阶段提示词

```text
先不要修改我们的算法。

第一阶段只执行：
1. 服务器审计；
2. 本地远程可视化链路验证；
3. CLoSD 官方 baseline 复现；
4. DreamControl 官方 baseline 复现。

本地远程可视化是强制项：
- DreamControl/Ours 的 Isaac Lab 优先配置 WebRTC livestream，让本地电脑能实时看到仿真；
- 本地通过 SSH tunnel 查看 TensorBoard；
- 服务器训练定期生成 MP4；
- 生成 scripts/local_sync_videos.sh，让本地自动同步服务器视频；
- 大规模正式训练优先 headless + periodic video，不允许为了持续 GUI 明显拖慢训练；
- 所有 WebRTC 连接必须走可信网络/VPN/防火墙限制，不直接裸露到公网。

CLoSD：
- 保持 Isaac Gym + PHC/PPO 原生 pipeline；
- 使用官方 checkpoint；
- 优先复现官方 multi-task / text-to-motion evaluation；
- 不移植到 Isaac Lab。

DreamControl：
- 保持 Isaac Sim 4.5 + modified Isaac Lab 原生 pipeline；
- 先使用官方 model_humanml 和 sample references；
- 第一轮只跑 Kick 或 Jump；
- 跑通 trajectory generation -> retarget/refine -> RL training -> play_eval；
- 如果没有公开最终 RL checkpoint，则使用官方代码自己训练。

所有兼容性修改独立 commit，并注明不改变算法。

baseline 未完成之前，不进入 I1/I2/I3。
```

---

# 38. Codex 第二阶段提示词

```text
Baseline 复现完成后，开始我们的创新 feasibility。

顺序必须是：
I1 -> I2 -> Standard PPO -> I3。

I1：
- 100–300 固定 constraints；
- 3 seeds；
- 测 human-space 和 robot-space；
- 重点看 retargeting drift。

I2：
- 冻结 I1 输出；
- 用 Sit / Stand / Step / Support / Jump；
- 必须同时提升 physics 并保持 task。

PPO：
- 固定 architecture/reward/domain randomization/training budget；
- PPO 不是创新；
- 只提供执行与 feedback。

I3：
- 建立 Motion Reference Experience Bank；
- 先人工审计 >=100 failure samples；
- 归因：
  Condition/Embodiment
  Physics/Contact
  Valid-Hard
  Mastered
- 归因可靠后再跑 2–3 个 outer iterations；
- 比较 No Iteration / w/o Attribution / w/o Motion Bank / Full I3；
- Valid-Hard 绝对不能作为 generator negative。

每个阶段输出 GO/NO-GO 报告。
```

---

# 39. Codex 最终实验提示词

```text
只有 I1/I2/I3 都通过 GO 后，才运行最终论文实验。

外部完整系统：
- CLoSD
- DreamControl
- Ours

不要构造拼接 baseline。

最终实验：
1. Full-System Comparison
2. Full w/o I1
3. Full w/o I2
4. Full w/o I3
5. Full Ours
6. I3 mechanism ablation

所有内部实验：
- same data splits
- same seeds
- same G1 assets
- same PPO
- same training budget

所有生成失败、retarget failure、physics failure、execution failure 都保留在最终 denominator。

最终生成：
reports/10_full_system_comparison.md
reports/11_main_ablation.md
reports/12_I3_mechanism_ablation.md
reports/13_final_summary.md
```

---

# 40. 最终验收问题

整个实验完成后必须回答：

### Q1 Baseline
CLoSD 和 DreamControl 是否成功按官方 pipeline 复现？

### Q2 I1
是否减少 robot-space condition drift，而不牺牲生成质量？

### Q3 I2
是否改善 physics/contact，同时保持原任务？

### Q4 I3
是否在 condition/physics validity 不下降的前提下提高 robot executability？

### Q5 Full System
完整 Ours 是否在共同任务上相对于 DreamControl / CLoSD 展示有意义的端到端收益？

如果这五个问题都有真实实验结果，才进入最终论文结果整理。
