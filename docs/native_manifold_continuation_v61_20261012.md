# V61：保留失败、先做CPU检查，再续行唯一缺失批次

V56反馈被动测量实际通过，未修改物理/控制/参考，33项数组与原指标完全一致。后续三个入口在启动随机臂前失败：V56见证复制尚未完成；V58错误字符串检查与保存repr不符；V60把str传给需要Path的hash函数。全部源码/独立失败见证/首次失败快照保留，台账仍只有一次81.41秒measurement，不能称这三个入口成功。

V61使用明确函数而非文本变换续行。新增CPU-only --preflight，检验Path哈希、原见证/失败事实/缺失批次；root先运行一次，仅保存带时间戳CPU检查，不运行模拟。--run调用同样的前置检查，只跑缺失的随机臂，使用原V56冻结的batch执行器，保存V61启动绑定和收据。

独立文档跟随者读真实两端文件，确认root CPU preflight存在且input/source/hash对应当前文件，不据其推定仿真成功。只执行以下命令一次，不调用此前失败入口、不自动重试、不改见证status。

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v61_20261012.py --run
```

预期exit0及NATIVE_MANIFOLD_V61_CONTINUATION_COMPLETE。确认随机臂72输入的33数组与原评分完全一致、非零接触/零解析错误、generator采样0、台账两条measurement，反馈轨迹及所有旧失败哈希不变。仍仅24验证来源×三偏移×单种子、两条固定参考回放；不是效力试验，不替换评分、不认证创新或真实接触真值。总600GPU墙秒被动测量、原4小时measurement/72小时全轮/15GiB输出/至少20GiB可用盘门不变。

保存CONTINUATION_DOCUMENT_WITNESS_V61.json/.md（两端）记录actual命令/exit/source/doc/manifest/receipt/summary与CPU preflight哈希，保留V56/V58/V60见证；不做后续诊断、不写Overleaf。若V61启动manifest或随机臂已存在，核对真实进程/终态，不再启动。
