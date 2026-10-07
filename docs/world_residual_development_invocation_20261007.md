# 残差开发队列：新鲜文档见证

只读此文档、run-experiment/compute契约及provider ledger。SSH ouyanghao@10.147.17.32，source10.147.17.250，私钥C:/Users/Pende/.ssh/id_ed25519。ROOT=/data/code/ouyanghao/oyh/conditional-human-motion-generation。基础spec哈希47f5eb26不变，GPU0 A6000计算，GPU1旧独立viewer，约61GiB磁盘可用，无依赖安装。上一份16环境扰动接口已真实exit0、7680trans、10steps×8env×80N、0PPO；该probe不重跑。完整开发详见plans/world_ppo_residual_development_20261007.md，旧FAIL保留，新formal_PPO=false。

只执行以下确切调用一次，不修复、不改代码、不安装、不重试。未知启动结果先只读查session/state，不能重复调用。

```
tmux new-session -d -s r3_world_residual_development_20261007 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; /usr/bin/python3 -u scripts/r3_world_ppo_residual_queue_20261007.py > artifacts/monitoring/r3_world_residual_development_20261007.log 2>&1'
```

先真正CUDA残差witness，核对零残差起始线性预测、future-action-zero两路独立、有限非零CUDA更新；随后seed42/43/44依次collect一次、fit一次。所有stage防重复输出，GPU0共享锁，每stage7200秒TERM+30秒KILL；GPU磁盘低于15GiB阻塞，不无限重试。队列已启动并不等于整个实验完成，需核对实际step/gradient推进。

预期每collect8192×480=3,932,160trans、0PPO；rank按reference/phase平衡4096clean/4096push，step50..59力buffer10条且4096个80N、dir均分，历史及H10窗口断reset。fit只train44/validation11，retired_test_evaluations=0；六预测器含两个ridge与四GRU，各GRU1000updatesbatch1024、实际gradientCUDAfinite、optimizerstep1000、初末hash和optimizer保存。Ridge四reference折OOF不含留折标签，shuffled保存完整donor且跨reference/episode/phase/contact/sign/recovery域精确排列，置换的是连续残差和contact标签。

每stage state COMPLETE exit0且对应sentinel；最后queue state COMPLETE development_triage=PROMISING/FAIL均是合法结果，formal_prediction_gate_passed及formal_PPO始终false。预期stdout/log末尾WORLD_RESIDUAL_DEVELOPMENT_COMPLETE，不能将开发validation变成确认集。hash与数组/window/target/OOF/source、每模型gradtrace实际核验，不另跑test或训练。

产物artifacts/world_residual_20261007/{collect,fit}_seed{42,43,44}_v1，state/log前缀r3_world_residual_<mode>_<seed>_v1。总state/log为r3_world_residual_development_20261007.*。所有原失败记录和v1旧test完整保留；数据及训练权重留服务器，不上传GitHub。
