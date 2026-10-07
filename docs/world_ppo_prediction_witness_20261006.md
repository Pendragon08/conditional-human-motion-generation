# 世界模型预测门：新鲜文档见证

仅按此文档执行一次。环境及资源账见 `docs/world_ppo_invocation_20261006.md`，基础JSON规范哈希47f5eb26。已由新鲜agent执行v2真实Isaac诊断：16环境48步、两PPO更新、一次辅助CUDA更新正常；reset_count0未覆盖reset。新predictor是同环境Torch＋已有pytorch_kinematics，不装库。服务器用户ouyanghao、10.147.17.32、源10.147.17.250、私钥C:/Users/Pende/.ssh/id_ed25519。

## 唯一确切调用

```
/usr/bin/python3 -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_world_ppo_prediction_queue.py
```

调用最多等待warm-up队列1800秒，然后顺序做seed42/43/44三次fit，每次自身上限7200秒，只执行一次，不修复、不改代码、不安装、不重试。formal PPO始终false。GPU0文件锁防和数据采集同时跑；GPU1独立预览不受影响。

采集三seed各8192环境×480步，完整reference固定划分（同np随机种子20261006）、44训练/11validation/11锁定test；旧policy见过所有参考，不称未见参考策略泛化。每seed最多采样65,536训练窗口，16,384 validation和test，10历史＋10未来步，episode/同reference确保窗口不跨任何reset。若phase0.2秒/contact pattern组无法跨reference精确错配，所有预测器共同删除该组。错配保存完整置换索引，跨reference及episode，标签边际完全一致。禁止普通随机时间窗作为holdout。

正确、state-only、错配预测器相同初始化、1024batch、1000辅助更新、每200更新以validation关节MSE选择checkpoint。Norm只训练reference拟合。Ridge相同有序历史＋未来动作，alpha0.1/1/10仅validation选择。未来实际动作只进入预测头，不能进actor；真实未来只监督。H10全身URDF FK与locked test真实未来q比较，paired bootstrap参考为统计单位；相对最佳ridge和state-only都改善至少10%且95%CI下界>0才通过。并报告各horizon关节/根速度/重力/足滑/contact AP/Brier；通过此预测门不能称控制创新成功。

结果位置：`artifacts/world_ppo_20261006/fit_seed<seed>_v1/summary.json`、correct/state_only/shuffled_world.pt、ridge.npz、train/validation/test_windows.npz，均在服务器，不上传权重。queue state在`artifacts/monitoring/r3_world_ppo_prediction_queue.state`。日志每seed`r3_world_ppo_fit_<seed>_v1.log`。预期三个fit完成sentinel及`WORLD_PPO_ALL_SEED_PREDICTION_COMPLETE PASS/FAIL`，队列退出0。预测门FAIL是合法负结果，不等于程序FAILED。

核查：warmup实际三×3,932,160，PPO更新0；原termination保留并记录真实reset数量；window保存episode和t/env索引核实断开；划分相同、normalizer训练分组、错配跨reference/episode；CUDA模型梯度确实完成、1000次每模型；日志／summary／hash一致；只一次locked-test调用。若输出或状态预存在即拒绝，不覆盖旧v1／v2失败记录。文档执行证据和收益结论要分开。
