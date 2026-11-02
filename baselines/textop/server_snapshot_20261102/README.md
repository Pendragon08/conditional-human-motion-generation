# TextOp 复现：服务器源码与归档索引

本分支保存 2026-11-02 服务器端 TextOp 脚本、配置及文本报告的快照，共 517 个文件，逐项 SHA-256 见 [SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json)。该快照保留原有 GitHub 仓库历史。

模型检查点、实验轨迹及完整 HTML 复现报告已分批上传至本仓库的 **GitHub Release 草稿**，尚未公开发布。草稿供仓库所有者在 Releases 页面检查；用户已确认持有相关材料公开发布的授权；正在完成剩余资料的上传与核验。以下清单记录草稿当前内容，随着整理可能继续增加。

绝大部分大型服务器文件直接从服务器上传至 GitHub，未通过用户电脑转发；HTML 报告从本地通过服务器上传。压缩包中的 ARCHIVE_MANIFEST.json 逐项记录文件校验值。

SuperMimic 15000、20000 和 latest 检查点已核对上传字节，但尚未完成数值完整性或统一回报评估。10000 统一回报尚未评测。`textop_gen_author_*_after4` 是 4 步诊断检查点。原文档见证版本经独立哈希核对与这些文件字节相同。

| GitHub Release 草稿附件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| supermimic4096_ckpt_0005000_20261101.pt | 216279375 | `737ca118fc9ae8400285c2ce84ec2fa0b61854aca6b02846dc1eedba6c13b85c` |
| supermimic4096_ckpt_0010000.pt | 216279375 | `ad6de467f1bdafc602a089bae5e0868e28ece46513f326d2d503a2c57f090c6d` |
| supermimic4096_ckpt_0015000.pt | 216279375 | `c4854166b035a7da669771de0996eac524c44d14af234328b55585a50f7d05b8` |
| supermimic4096_ckpt_0020000.pt | 216279375 | `8004e0072247e1b2cf8ab4f9d0c3f7820b99cd3026fefa4f3ffa46a5ca180a2b` |
| supermimic4096_ckpt_latest.pt | 216279203 | `90f713c010bea691a27848c728a9706ba5151242facaede7baa6c6861cbf6942` |
| textop_common_return_ppo_super5000_20261102.tar | 205864960 | `aca2faa8ea8e8d5e0af4220b42021e92091c98b95c11c7e4964c846af0e47a49` |
| textop_filter_policy_1000.pt | 90799598 | `76b3cecff93807b90b61202907cc62e5b40c6678d0da2f4258d3a5541ce91b08` |
| textop_filter_policy_10000.pt | 90799682 | `6f949e4edc50d9a763a3634d716162cb5ddecb079bd1142d3720f907f5efab45` |
| textop_filter_policy_15000.pt | 90799682 | `66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88` |
| textop_filter_policy_5000.pt | 90799598 | `cfb443ab9ac80a615fdffdd881ede38b6e1997a0993752ff2a82a9f7c9939fae` |
| textop_gen_author_ldm_after4.pth | 305124157 | `bdf540b976b26a6ca6e47fc3a595314970211a2419386b72c561d8f90e786be3` |
| textop_gen_author_vae_after4.pth | 684683326 | `5785d063f37ad30f639760a79bad5ffe0e69c30507a24243df90c895f8e5f74b` |
| textop_ppo_training_mujoco_and_comparison_results_20261102.tar | 1256058880 | `3d8754ff190197b712d3f4af8fea370f2473b259facb016f97764d22f7181bf1` |
| TextOp_reproduction_report_20261102.html | 31396569 | `15c4a9ad2348bb306034f4d22ec0e1b20eaec78e1c214a01fad254b242f277fa` |
| textop_supermimic_other_runs_and_audits_20261102.tar | 1087191040 | `fbceb1d47a930f0395e21b341dc57ba8c6aa661e0225fcbd37883e16526e3ddd` |
| textop_vae100k_inference.pth | 171154895 | `3f36bcbccdee10a8c1c2eb67b0614be63b32bd8a05cff92de23cc358087117f4` |
| textop_vae_emergency_8100.pth | 513526637 | `2b958a5e4efbd74ddfe6bb99b8a5eb8174cec143bb782689b1f6af8934a6c2fe` |
| textop_vae_public_full_1000.pth | 513352517 | `b1ffba568972a9cca916847f674e8d9d9c4b391084cecde67772550bf807c6e0` |
| textop_vae_quick_1000.pth | 513523773 | `cbe07e8671726d2c355faf9e457976e2f09d49cf62af1cf8da5bc9774309b3a6` |
| textop_vae_quick_2000.pth | 513523901 | `5109f593309349504d52677997bcdd21de3d27e045b9154be8c06278182b1219` |

| textop_earlier_ppo_training_and_witnesses_20261102.tar | 1284618240 | `b2c7aa6f4bc3bb2ad46b45dd0cc36911e05f9eb063ab5d25698258be48d435e3` |

| textop_additional_training_logs_and_results_20261102.tar | 1182894080 | `0c986fbe8a427335bd4439d0d858171f49f66f8f1452a513e38d7730a86a69cd` |

| textop_snapmogen_retarget_and_evidence_20261102.tar | 300933120 | `80bd7739c10dd7348ff89d3ba95cd17007d25c49e9bba0dde8b16e67c74d0e40` |

| textop_generated_fk_library_part01_20261102.tar | 1257318400 | `df863c69825d2bc1674ed0f9f5c240facaec706fad11f95182e2829243fc8a68` |
| textop_generated_fk_library_part02_20261102.tar | 1257338880 | `c8746f3a78fb0b6fde958612f19609cc60d5b9c3d9630936efe41d06f4ebbb9c` |
| textop_generated_fk_library_part03_20261102.tar | 1257359360 | `c1c68f6829d8fa6ab99dad7bd6859081039ba74159619b479ab64078067cb1e8` |
| textop_generated_fk_library_part04_20261102.tar | 1257359360 | `5b012105bdfa38d1c1d3880d70e1f340ecdfe19fc235c5b8646f52d3c427270d` |
| textop_generated_fk_library_part05_20261102.tar | 759111680 | `89feceffc55c1cae0eb133a147e65ae0e88e2ce8441f661e83ca84be11d80cec` |

原始第三方动作数据、重定向数据及逐条文本的公开传播须遵守各自数据集许可。仓库此前的存储清理记录说明了相关限制；不能把源码或模型的公开授权推定为训练数据的再分发授权。

这是归档进度索引，不代表所有历史文件或全部评估均已完成上传、审核或公开。

### 原始训练参考与报告更新

原始训练参考共 5,560 个文件，分成五个归档；每个归档包含逐文件校验清单。更新后的报告增加了同动作视频尚不符合展示条件的说明，旧版报告保留以便追溯。

| Release 草稿附件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| textop_original_fk_library_part01_20261102.tar | 1260410880 | `56a9ac9eb7412e5ade933c38e92071313df8fbd0576bfe238358c52812e532bb` |
| textop_original_fk_library_part02_20261102.tar | 1258106880 | `d26e9abe9371fd008f5071cc2fdfd7cbfcbaab277477fd25ef0c9966550a3c07` |
| textop_original_fk_library_part03_20261102.tar | 1259601920 | `6a257102aa21d2a5a9817cf973b9cc287eaac6e7ce2426bce17b684d3b77776e` |
| textop_original_fk_library_part04_20261102.tar | 1258362880 | `1dce7f7f82502c669b778450a50c65d269f71de1a92806bc5eba1aa94e6bb6ee` |
| textop_original_fk_library_part05_20261102.tar | 1137008640 | `405771bfcd757f36928430697f0e20b85c4b9fe6ea37b9517876270bc251b4f7` |
| TextOp_reproduction_report_video_status_20261102.html | 31397035 | `2a29cf271eae7f8ee484f679f00d6e69d0f4d847bbdb90d38aaa1fcc04cbc914` |


### 数据准备与在线验证记录

| Release 草稿附件 | 文件数 | 字节数 | SHA-256 |
| --- | ---: | ---: | --- |
| textop_preparation_online_checkpoint_video_results_20261102.tar | 1970 | 1163274240 | `3ae81cbe294cf792b3d1441cc44a111dca3b37e12e663af09f5a43bfbc3443c5` |

### 历史跟踪器验证

| Release 草稿附件 | 文件数 | 字节数 | SHA-256 |
| --- | ---: | ---: | --- |
| textop_tracker_teacher15000_validation_20261102.tar | 16569 | 681922560 | `63ee05e5a520e23819eb1b6eaca2b276b2edc8de2531f8e5ceb17e6adf628b0f` |

### Additional historical tracker validations (draft assets)

These four archives contain historical validation outputs. The release remains a draft; inclusion is not a license determination for bundled third-party models or software.

| Asset | Files | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| textop_tracker_other_validations_part01_20261102.tar | 767 | 1346641920 | `f3ac723e7e3be69c2a50c5b5b56245f35d82acc8d3a2e26a4d2b4e8483127f8e` |
| textop_tracker_other_validations_part02_20261102.tar | 72 | 1341317120 | `7e306c9c555fa173bf840a2d7d18e0d5d17adaf7614cb54643f3334fdae990b3` |
| textop_tracker_other_validations_part03_20261102.tar | 279 | 1339228160 | `00fbe0758043130603b23ca2f206cbfc4e002cae58783c84c1127aa0619faa01` |
| textop_tracker_other_validations_part04_20261102.tar | 654 | 871526400 | `aba4735b5c9c53441489cc89647a24d3e813f6e51e4f674e65e7e85e44f8edb8` |

Part 03: the original upload client timed out while awaiting the response. A subsequent read-only reconstruction from the current server files under the shared lock produced the same SHA-256 as the draft asset. The original timeout remains part of the record.

### Generator validation evidence (draft asset)

`textop_generator_validation_evidence_20261102.tar` contains 20 small evidence files (122880 bytes); SHA-256 `caf511c8266d09768250da363dc60936290304b1f8a085e261a44df0dffe501e`. The previously indexed author VAE and LDM model assets cover the generator-state models: document copies were independently checked to match the author copies, and each `ldm/vae.pth` is a hard link to its corresponding `vae/state_after4.pth`. Those six model paths are excluded from this evidence archive to avoid duplication.

### Additional numerical experiment evidence (draft asset)

`textop_numeric_remaining_evidence_20261102.tar` contains 109 files (144271360 bytes); SHA-256 `65ac1340084f72e23680fa99e4cf3b3e6a3753eb11b5b41b1d48e0e4abb84538`. This covers the remaining inventory-listed numerical experiment logs and witnesses. The previously archived PPO groups and original FK library are excluded to avoid duplication.

### TMR evaluator checkpoint (draft asset)

`textop_tmr_epoch100_fullstate.pth` is the user-trained, resumed 100-epoch full-state checkpoint (192952453 bytes; SHA-256 `7bc1955dfcb31d1eeaa966c8f5b8e972e17d6d58da1205c70fd8029b85050fc8`). The earlier interrupted training run remains part of the provenance; this checkpoint does not represent a 500-epoch run. The asset remains in the draft release pending the broader source and redistribution review.

### TMR evaluator checkpoint (draft asset)

`textop_tmr_epoch100_inference.pth` is the 100-epoch inference export from the user-trained resumed checkpoint (64234041 bytes; SHA-256 `6ed04cb040419f714962522ce37590437740896a6fb230533e3c81c599b9b357`). The earlier interrupted training run remains part of the provenance; this checkpoint does not represent a 500-epoch run. The asset remains in the draft release pending the broader source and redistribution review.
