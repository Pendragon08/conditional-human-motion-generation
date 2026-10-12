# V52：解析接触校准的远程文件路径修复

V50首个CPU命令退出1：正文引用了仅在本地导出目录存在的`public/raw/.../metadata.json`。保留V50源代码、空的首次输出目录和独立见证失败，不重试该命令。本版仅改为服务器执行绑定的`raw/.../metadata.json`，输出另存V52，并记录V50父源码哈希。24项解析刚体控制、V21测量函数、原阈值和判断边界完全不变。

复用既有NumPy1.23.5，CPU执行，不安装环境，不调用生成器／物理仿真／GPU。四个碰撞盒各包含静止、0.2m/s平移、正向滚动、反向滚动、滚动叠加平移及腾空。滚动角速度3rad/s，接触边缘由独立盒边界选择，接触点速度按刚体公式规定。支撑力是假设输入，角度是独立参数样本，不能作为真实接触或动态积分实验。

## 独立执行（仅一次，不修复或重试）

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/calibrate_surface_proxy_v52_20261012.py
```

预期退出0，输出`SURFACE_PROXY_ANALYTICAL_CONTROLS PASS_ANALYTICAL_CONTROLS 24`。核对V52及V50源码哈希、原生元信息和冻结阈值／测量函数哈希、24项逐例判据、数组哈希、NumPy版本。在`results/surface_proxy_calibration_v52_20261012/DOCUMENT_WITNESS.json`记录`status`及具体证据，附Markdown判断边界。

通过只代表公式能够区别这些已知瞬时刚体状态，不证明人体执行没有滑移，也不证明当前高角速度关联是误判。不得因此更改旧成功率或放宽阈值。失败保留，不进行仿真。结果不写入Overleaf。
