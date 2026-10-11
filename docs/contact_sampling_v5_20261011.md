# 接触采样时点复核 V5

仅检查GPU状态读取时点。冻结生成器、控制器、原始目标、输入来源及评价阈值。不改变PhysX配置或控制输入。V5在reward后、各类张量复制前加入`torch.cuda.synchronize()`，确保生产者流完成。旧V4与V2记录不覆盖。

## 文档见证：执行一次

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sampling_v5_20261011.py --check
```

预期exit0、`CONTACT_SAMPLING_BATCH_PASS repeat_a`和`CONTACT_SAMPLING_DOCUMENT_CHECK_PASS`。首批目录`results/contact_sampling_v5_20261011/raw/repeat_a`应保留12首次尝试；receipt complete，七张量重置缓存差值0，输入/输出、冻结模型、源数据、原始目标与数值重算绑定通过。见证者只执行一次，不重试、不改代码、不执行 --run；可只读查看证据，不下载许可数组，不操作Overleaf。

后续执行同一命令的--run，由执行者收集同臂、同种子、同噪声和同输入的repeat_b，比较完整120帧运动学、接触力及支撑滑移代理。1mN仅是采样可重复性诊断容差，不修改任何动作合格阈值，也不直接认证真实接触。

若同步后接触力仍不一致，不能宣称已经修复完整仿真重置/接触测量，也不能靠放宽原阈值继续主要效能测试。记录失败边界，暂停G1和下游训练。此诊断仅4个自定义验证来源与1个生成种子，不作泛化或创新结论。

数据盘≥20GiB；所有旧失败与新输出累计≤15GiB，含此前对比的总预算≤72GPUh、本诊断计入4GPUh测量上限。只用空闲GPU，不安装依赖、不删除其他项目。本轮实验结果不写Overleaf。
