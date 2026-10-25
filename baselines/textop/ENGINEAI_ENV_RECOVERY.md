# 闲置 EngineAI Python 环境的完整备份与恢复

本次清理对象是个人目录中的`.venv-engineai-official`及指定旧缓存。
该环境没有运行进程、没有其他环境依赖；当前TextOp、GMR及共享IsaacSim/
IsaacLab使用的环境均不在清单内。框架源码和基础Python保留。
最终是否已删除原环境，以results/storage_cleanup_20261025/cleanup.json为准。

完整恢复包位于：
https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/inactive-engineai-environment-cache-20261025

归档包含原环境的全部文件、链接、依赖许可及旧缓存，未包含AMASS/SMPL/
BABEL原始资产、实验数据集或实验检查点。源文件82,327个逐个通过SHA256
核对，压缩归档完整读取验证；六个分片须全部核对GitHub摘要后才允许清理。

- 压缩归档：5,769,849,788字节。
- 拼接后SHA256：`fe695c51334e5cd3baed3b3f35587c30a984e2cf2122f61ec186beb8e8ebe5a2`。
- 分片顺序：`inactive_engineai_and_caches.tar.gz.part001`至`part006`。
- 每片字节数和SHA256见同一Release中的archive_index.json。
- 依赖锁见requirements-engineai.txt，原环境配置见pyvenv.cfg。
- 基础Python来自`/data/code/ouyanghao/oyh/world-model-risk-planning/.env/bin`，
  其文件保留；可编辑安装所指向的IsaacLab-engineai-official源码也保留。

恢复时，先在空间足够的个人目录下载分片，按001至006顺序拼接，核对上面
的完整归档SHA256。在服务器上可使用个人RAM目录暂存分片，恢复环境本体
需要另有约8GiB的数据盘空间。仅恢复原环境而不恢复无用缓存时，可执行：

```bash
tar -xzf inactive_engineai_and_caches.tar.gz -C / data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-engineai-official
```

这会还原到原绝对路径。若该位置已创建新环境，应先安排两个版本的位置，
不要直接覆盖。恢复后需重新做依赖、实际CUDA计算和对应任务验证，不能仅
凭文件解压成功就称环境就绪。

系统盘资源迁移属于另一项操作：TextOp资源和PhyGile保留资源逐文件核对
SHA256后迁到个人数据盘retained-system-assets-20261025，原/home路径保留为
链接；它们的内容保留。详见relocation.json和本地个人目录空间清理报告。
