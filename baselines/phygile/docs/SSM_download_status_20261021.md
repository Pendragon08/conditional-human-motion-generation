# SSM 下载与核验结果（2026-10-21）

**SSM 已下载成功；完整 AMASS/HumanML3D 动作及人体模型仍未全部取得。当前新增 G1 配对动作和 262D 序列均为 0。**

## 实际文件核验

在用户下载目录发现 `SSM.tar.bz2` 和 `SSM (1).tar.bz2`，每份 38,268,098 字节。两份 SHA-256 完全相同：

`42f91a1a0e74056b918e714192784b3aa9ae6a300031c8c94a5c2c65701741e4`

压缩包完整读取通过，包含 30 条 SMPL-X stage-II 动作、3 份 stage-I 配套参数和 LICENSE.txt。30 条动作的模型种类、性别、采样率、16 维 betas、root_orient、trans、pose_body 和 165 维 poses 均已检查；需要的数值数组全部有限，冗余姿态字段一致。没有反序列化 object 字段。

之前根据下载网页仍显示登录表单判断“未下载”的状态有误；以本次实际文件核验更正。较早的直接传输失败收据保留为历史尝试，不代表当前 SSM 文件状态。

## 文本对应情况

30 个 HumanML3D SSM 录制来源均按目录和文件名精确匹配到下载包的 SMPL-X 导出，可关联 60 个官方文本 ID（包含镜像 ID）。这些只是候选对应：尚未核实 SMPL+H 原始参考时钟与当前 SMPL-X 导出的时间对应，也未实际生成镜像机器人动作。不能把这 60 个 ID 当作 60 条训练就绪的 G1 配对序列。

## 尚未全部取得的部分

- SSM 是本次 HumanML3D 索引涉及的 17 个 AMASS 子集之一；其余 16 个子集没有在本次核查的下载目录或原始资产目录中出现。
- SMPL-X 人体模型文件尚未找到。本 SSM 包全部动作标为 female，先验证该子集至少需兼容的 SMPL-X female 模型；其他子集可能还需要 male 或 neutral 模型。
- `female_stagei.npz` 仅含拟合形状及标记点等参数，不含完整人体网格模板、关节回归器和蒙皮模型，不能替代 `SMPLX_FEMALE.pkl` 或对应的 npz 模型。
- HumanAct12 是独立的骨架来源，不能计入 AMASS SMPL-X 下载完成率。
- 实际 GMR 重定向、文本时间范围核验、动作过滤、262D 编码和生成器训练尚未执行。

## 文件与复跑

可复跑核验脚本为 `scripts/ingest_ssm_archive.py`，输入是用户已授权取得的官方压缩包。输出目录和收据必须不存在；脚本拒绝覆盖既有材料。原始动作留在本地，GitHub 只保存代码、哈希、统计与来源对应记录。

```sh
python scripts/ingest_ssm_archive.py --archive /path/to/SSM.tar.bz2 --public cache/humanml3d_public_20261018 --output cache/new-ssm-verification --receipt results/new-ssm-verification.json
```

本次实际执行结果见 `results/paired_data_20261021/SSM_verified_acquisition.json`。状态是 `SSM_RAW_ARCHIVE_VERIFIED_NOT_G1_TRAINING_DATA`。
