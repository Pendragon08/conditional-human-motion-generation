# TextOp 复现

23DoF/57D基线训练现场已暂停；优先准备论文29DoF/69D数据。

优先查看 [数据准备核查与重建报告](reports/dataset_preparation_20261025.md)。另见 [启动报告](reports/启动报告_20261024.md)、[环境说明](ENVIRONMENT.md)、[官方版本](upstream/README.md)。

官方代码以固定提交镜像保存于upstream/，下载数据不入Git。scripts/为本次下载、隔离环境安装、训练启动与指标记录代码。官方算法源文件未修改；论文29DoF重建部分为独立自实现并复用GMR核心，尚待真实动态筛选。
