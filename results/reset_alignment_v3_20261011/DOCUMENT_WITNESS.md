# 初始化坐标测量 V3：独立文档执行见证

结论：**见证失败，保留现场，无重试。** 文档测量命令执行一次。首批 planned_s11 子进程退出0，完成12/12案例；外层在下一批启动前触发GPU资源保护断言，退出1。7项导出运动学张量的初始化对照中4项不一致，初始化一致性未通过。该结论限于执行和测量接口，不提供论文机制有效性结论。

## 阅读与预检

执行前只阅读指定的 docs/reset_alignment_measurement_v3_20261011.md 与 .aris/compute/closd-baseline-20261010.md。环境JSON仅用于规范哈希，结果与总账仅用于事后覆盖核验，版本清单中的文件仅按原始字节校验SHA256；未阅读源代码内容或修改代码。

预检UTC 2026-10-11T11:23:55Z：旧队列 COMPLETE_GATE_NOT_MET；新结果目录不存在。环境规范SHA256为81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780，匹配文档记录。规范为json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False)的UTF-8字节。环境JSON原始文件哈希e5f9e6b711b7c6221cfb01e3deab404c5849e521ec08a3d25dca1820a279fcf6与规范哈希用途不同。

GPU0为699MiB、0%利用率，已有诊断play_eval.py与长期live viewer进程，无训练进程，满足<2048MiB/<=5%前提。未操作其他GPU进程。使用冻结warm runtime，见证人未发出安装或构建命令；首批日志中的cached gymtorch loader显示ninja: no work to do，没有实际编译工作。

## 唯一测量命令及原始输出

```sh
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reset_alignment_checks_v3.py
```

启动UTC 2026-10-11T11:24:42.3527247Z，结束UTC 2026-10-11T11:25:34.3809353Z，外层退出码1。执行的SSH argv与文档命令一致。measurement.command.txt保存精确命令，measurement.stdout.txt与measurement.stderr.txt保存分离的原始字节流。首批子进程的合并日志原样保存在result_snapshot/planned_s11.log；子进程没有独立保存两个流，不能事后重建其分离stdout/stderr。

外层stdout全文：

```text
RESET_ALIGNMENT_BATCH planned_s11 0
```

外层stderr全文：

```text
Traceback (most recent call last):
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reset_alignment_checks_v3.py", line 118, in <module>
    main()
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reset_alignment_checks_v3.py", line 89, in main
    assert int(gpu[0])<2048 and int(gpu[1])<=5, 'No displacement of another workload'
AssertionError: No displacement of another workload
```

失败当时的memory/utilization采样值未输出，因此不能确定是哪一个资源界限失败。未重试或启动后续测量。V3 state.json仍写RUNNING/planned_s11，是失败后保留的旧状态；必须以外层退出码1和异常为准。

## 每批记录与覆盖

| 批次 | 启动 | 案例 | 子进程退出码 | 计费秒 |
|---|---:|---:|---:|---:|
| planned_s11 | 是 | 12/12完成、每例120帧、0终止 | 0 | 31.998405760154128 |
| 其余7批 | 否 | 0，未覆盖204次尝试 | 无 | 无 |

文档预期8批/216次尝试，实际1批/12次。首批来源4项，jump和dance各2项，每源-0.15/0/+0.15m。全部24个验证来源的两模式批次未运行。首批脚本成功不构成文档见证通过。

初始实际/参考/引导重置根XY已记录。独立计算实际与参考的最大欧氏误差0.0007189986346922763m（0.719mm），最大单分量0.0007125437259674072m，12例均<1mm；引导重置与参考XY为0误差。该项首批通过。

| 引导重置与首次动作前的张量导出对照 | 最大差 | SHA256相同 |
|---|---:|---:|
| _humanoid_root_states | 0 | 是 |
| _dof_pos | 0 | 是 |
| _dof_vel | 0 | 是 |
| _rigid_body_pos | 0.035006970167160034 | 否 |
| _rigid_body_rot | 1.4754348993301392 | 否 |
| _rigid_body_vel | 3.7104039192199707 | 否 |
| _rigid_body_ang_vel | 6.5190958976745605 | 否 |

**7项一致性失败。** 这些是metadata导出的运动学张量最大差及成对SHA256诊断；完整初始张量数组没有写入trajectories.npz，见证人不能从原始数组重算7项差值。此对照不涵盖完整PhysX隐藏状态。

只有生成种子11、planned模式运行。原目标哈希61f125340149451df2f94ec5408ad493fe01a782a83f03f89dafab88edf6726c已按float32字节独立重算匹配；prefix哈希553e3c4fccda00f1a6583756e17ea7f858b476fa48dc6999985a4c974405d84c，noise哈希1a13fe432251f25e4aa92decf9dc07e208ed82a979948e32b82868b6c21b0ec1。种子12/13未运行，不能核验跨种子目标/前缀一致及噪声不同。原始prefix/noise数组没有保存在NPZ。

4个来源的三偏移案例已独立对照：绝对原目标相对零偏移的差与预设[±0.15,0,0]最大分量偏差3.063678741449527e-6m；同源引导根位置差最高2.86102294921875e-6m；扣除各自引导根位置后的目标差最大偏差3.073364496231079e-7m。记录全部数值，不临时引入验收容差，也不能仅从这些输出独立证明没有不同随机平移。详表保存在witness.json/audit.stdout.json。

NPZ时间索引独立检查恰为0..119，1440/1440有效帧，无终止，所有数值数组finite。两模式原目标/前缀/时间对照仅覆盖planned一侧，不能进行跨模式相等核验。

## 请求与输出lineage

planned_s11_execution_request.json与lineage.json中的request完全相等。版本文件哈希、manifest原始文件哈希、冻结源JSON配置哈希和阈值哈希均对应，全部13项版本清单文件的当前原始字节哈希仍匹配。解析后的resolved_config.yaml与请求绑定的源JSON为不同artifact，其原始字节哈希不相等不构成lineage失配。

| 输出 | SHA256 | 请求绑定校验 |
|---|---|---|
| metadata.json | 14dd190dd8d7c4f1aa4309ca20705c1390abc07dbe590ac28a9b9840268fef5d | 匹配 |
| metrics.json | 3075c125ef53ec6357896c5c23e45566914d4ed61e287bde31794915cd1be0aa | 匹配 |
| trajectories.npz | 453d0d2c4f344e42c71e493a9dad13a4f8cbd5576d701723108898ec9c0b90cc | 匹配 |

lineage certification为bound_request_at_execution，confirmatory=false，study_status为amended diagnostic run; thresholds retained。首批lineage通过。12个本地结果快照文件均与远端只读审计SHA256相等。

## 预算与停止范围

首批31.998秒<600秒。旧轮、V1失败27.184939824044704秒、V2失败28.387516954913735秒和V3首批合并登记7139.618935374543秒（约1.98323GPU小时），其中measurement1987.6664440799505秒（约0.55213GPU小时）。记录保留旧轮1800秒开发与见证预留扣款。另用完整V3外层wall time替代V3批次计费做保守核对，数值见accounting_and_snapshot_verification.json；仍低于总72/测量4GPU小时。

旧physics-guided与V1/V2/V3结果目录共1999984856字节（1.862631GiB）<15GiB，也低于20GiB；数据盘可用43626024960字节（40.629902GiB），>20GiB。V3目录2090634字节。此储存核对限上述结果目录，不独立认证整个项目累积输出总量。

首批receipt记录joint_pass=0/12，保留为测量输出。初始化7项一致性失败，见证结论FAIL_PARTIAL_EXECUTION。按文档停止后续G1与下游训练；不放宽阈值、不与旧成绩合并。V3结果目录现已存在，因此不得重跑本命令。旧目录、代码和远端输出均由见证人保留原状。没有论文机制有效性结论。

完整机器记录witness.json包含精确命令、完整stdout/stderr、逐项覆盖、paired tensor哈希、偏移详表、总账、结果快照哈希与只读审计。其他文件包括preflight、measurement、audit和completion_audit的命令/输出/执行信息以及result_snapshot原样副本。