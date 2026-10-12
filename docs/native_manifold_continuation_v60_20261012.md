# V60：按实际保存的失败记录进行一次明确续行

V56反馈臂的被动测量已完成，一次独立--check通过；V56原--run在见证尚未远端同步时失败，未启动随机臂。V58的续行又在启动前失败，因为其错误字符串检查要求文件名，而原状态只保存了FileNotFoundError的简短repr。V58独立调用仅一次，exit1；无随机批次、无新增GPU计费，所有代码与失败见证保留。

V60不编辑V58，仅先核验V58源码和失败见证、固定首次失败文件哈希，再将这一条错误记录检查改为实际完整repr匹配。其余冻结接口、严格见证哈希、缺失批次、资源门和状态相等检查不变。V60自己保存新的启动绑定和收据，不称V56/V58成功。

独立跟随者检查本文件与两端新源码相同，读取V56原文及V58失败见证。执行以下命令一次，预期exit0及NATIVE_MANIFOLD_V60_CONTINUATION_COMPLETE。如出现任何失败，保留完整标准输出，不重试。若V60 manifest或随机臂已经存在，核查实际进程/结果，不再次运行。

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v60_20261012.py
```

只续行随机臂72个既定验证输入、24来源、三偏移、种子11。核对random_equality.json的33项数组和原评分完全相同，非零模拟接触、零解析错误、generator采样0，预算台账最终两条measurement。反馈轨迹与两个旧失败记录不变。总被动测量GPU墙秒上限600，原4小时测量阶段、72小时总预算、15GiB输出、20GiB可用磁盘门继续有效。

保存CONTINUATION_DOCUMENT_WITNESS_V60.json/.md到V56目录，两端同步。记录actual命令/exit/source/doc/manifest/receipt/summary哈希；不覆盖V56/V58见证，不做离线分析、不作效力结论、不写Overleaf。
