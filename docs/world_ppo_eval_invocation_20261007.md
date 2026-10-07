# 冻结策略控制评估接口独立见证

只读run-experiment skill、docs/dreamcontrol_ssh_environment_ledger.md与本文。依文执行一次；不编辑/安装/调参/重试/启动formal。原两个8192容量诊断均childrc0后才运行。本次只验证新的固定初始分配、first-episode统计与力/姿态记录，不对2轮诊断策略做科学收益比较。

本地D:\具身智能\conditional-human-motion-generation执行一次：

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/launch_world_ppo_eval_interface_20261007.py
```

唯一tmux r3_world_ppo_eval_interface_20261007，GPU0串行，GPU1viewer不改；spec configs/env-world-ppo-eval-interface-20261007.json，source configs/world_ppo_eval_sources_20261007.json，warm依赖不变。先P-history后W-correct固定capacity最终model_update0002.pt，各64环境×120步、seed42、0PPO/aux/norm。各1800秒时限+30秒终止；state r3_world_ppo_eval_interface_20261007.state及r3_world_ppo_eval-diagnostic_<arm>_42_20261007.state/.log。立即捕获rc；输出artifacts/world_residual_ppo_20261007/eval_interface/eval-diagnostic_<arm>_seed42_v1。失败保全，不由见证者修复重试。

只读核验每stage实际rc0/sentinel，64×120=7680评价转移、所有模型norm哈希前后不变。原新11 manifest及按文件次序identity，初始phase0。allocation=arange(N): ref=i%11，within=i//11，clean/push=within%2，sign=((within//2)%2)；不是随机挑幸存样本。原reset事件仅首次参考采样换固定表，完整原手部0/27跟踪关节、root/velocity和0.15m起始高度，后续auto-reset原规则。第一次done inclusive成功，后续跟踪失效每步2rad；模拟step50..59 worldY80N buffer逐步核验，60..109仅push primary。新11已经用于预测确认，非端到端untouchedtest。

独立载入actual_first_episode_trace.npz重算first_alive切断、RMSE失效填充、reference/domain/sign分配、firstfailure/timeouts、11行macro积分/clean统计。root-height/angle原termination各自保存，不把timeout当失败。每trace原生body/root/joints选4实际环境120帧、finite且姿态变化；trace不是渲染视频，evaluation_video_verified=false。policy/norm/world前后hash确切相同且0updates。仅metadata回传results/world_residual_ppo_20261007/raw/eval_interface；权重/trace不上传。完成后独立报告真实文档差异；不擅自启动formal。
