# AMASS 兼容 SMPL-X 模型安装及验证

2026-10-21。用户已明确同意 SMPL-X 独立模型许可，取得官网注明适配 AMASS 的 `smplx_lockedhead_20230207.zip`，不是 v1.1 的不同形状空间。官方来源：[下载页](https://smpl-x.is.tue.mpg.de/download.php)、[模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html)。

压缩包为 411,448,739 字节，SHA-256 `88d35123fc97151bc258dc6f28f3967ae4c16117e4b452445f20598c43c43099`。female、male、neutral 三个 NPZ 及 md5sums 文件合计 411,447,688 字节。压缩包 CRC、路径检查、服务器归档哈希与解压后逐文件哈希已有独立收据；不能仅凭这些收据宣称模型前向计算成功。

本地文件在 `D:/具身智能/PhyGile复现/后续AMASS下载_20261021/body_models_smplx_lockedhead_20230207/models_lockedhead/smplx/`。

服务器模型根目录：

`/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead`

使用既有 `gmr_amass_20261021` 环境（spec `141547bf`），没有更新父环境或第三方 GMR 文件。以下命令在服务器 CPU 上执行，必须选择尚不存在的输出文件：

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/model_witness_20261021/witness_smplx_models.py --model-root /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead --ssm-file /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021/extracted/SSM/20160930_50032/dribble_shoot_sync_stageii.npz --server-model-receipt /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/server_verification.json --output /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/model_witness_20261021/model_witness_document.json
```

预期退出码为 0、状态为 `SMPLX_MODEL_CPU_FORWARD_PASSED`、seed 为 42、device 为 cpu。三种模型均输出有限的 `1×10475×3` 顶点；female 使用一帧真实 SSM 的 body/root/trans 和全部16个 beta，其余两个模型使用默认姿态。首次执行者的输出另存 `model_witness_author.json`，独立执行者按上文原样命令生成 `model_witness_document.json`；对应收据不存在或失败时，该验证尚未确认。

验证程序显式设置 `num_betas=16`，手、脸保留模型默认值。它不运行原样 GMR 数据加载器，不证明完整手脸动作恢复、G1 重定向、文本时间对齐、过滤、262D 编码或训练已完成。人体模型、原始动作及逐帧产物不公开上传 GitHub；仓库仅保存代码、清单和验证证据。
