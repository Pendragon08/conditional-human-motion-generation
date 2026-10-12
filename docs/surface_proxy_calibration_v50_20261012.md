# V50：冻结滑移代理的解析接触校准

本命令只在CPU上计算24个明确标为合成数值对照的刚体状态，不启动仿真、生成器或GPU。复用既有NumPy1.23.5环境，不安装或修改包。

使用V46保存的四个足部碰撞盒及逐刚体COM，调用未修改的V21测量函数和原阈值。每个碰撞盒包含静止、0.2m/s平移、正向边缘滚动、反向边缘滚动、滚动叠加平移以及腾空六种控制。无滑动控制按刚体速度公式独立规定最低接触点速度为零；滚动角速度3rad/s。支撑力是假设输入，不是仿真测得的真值。姿态角是独立参数样本，不是积分轨迹。

## 独立执行命令（只执行一次）

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/calibrate_surface_proxy_v50_20261012.py
```

预期命令退出0并输出`SURFACE_PROXY_ANALYTICAL_CONTROLS PASS_ANALYTICAL_CONTROLS 24`。已有`first_attempt`则拒绝执行，不覆盖或重试。若失败，保留文件及错误。

见证核对源码与三个输入哈希、24项逐例判据、NumPy版本以及数组哈希。输出路径为`results/surface_proxy_calibration_v50_20261012/first_attempt/summary.json`和`analytical_controls.npz`。

## 判断边界

该结果只能检查测量公式对已知瞬时刚体状态的响应。通过不证明真实人体执行发生滑动，也不证明仿真接触点正确；失败不允许更改旧阈值或重算历史成功率。不能把高角速度本身当作误判证据。当前创新数量保持三项候选、一项首轮效力对比、两项未完成、确认零项。实验结果不写入Overleaf。
