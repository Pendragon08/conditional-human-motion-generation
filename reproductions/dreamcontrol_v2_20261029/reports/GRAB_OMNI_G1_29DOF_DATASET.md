# GRAB + OmniRetarget 文本—G1参考轨迹基线

2026-11-03：按用户指定的数据构建范围完成，尚未启动模型训练。

## 数量

| 来源 | 轨迹 | 保留的文字描述 | train | val | test |
|---|---:|---:|---:|---:|---:|
| DexWeave GRAB + MotionHub | 1,335 | 4,005 | 903 | 162 | 270 |
| OmniRetarget / OMOMO | 1,826 | 1,826 | 1,188 | 322 | 316 |
| 合计 | **3,161** | **5,831** | **2,091** | **484** | **586** |

OmniRetarget包含312条original、1514条修改/增强参考，因此轨迹条数不等于独立原始录制数。GRAB每条保留3句来源文字；当前加载器默认使用主描述，其他文字保留在`candidate_captions`字段。

## 已完成的构建步骤

1. 复用已重定向的G1轨迹；按来源序列和文件哈希关联MotionHub/OMOMO原有文字。
2. 按关节名称和来源顺序接入29DoF G1，保留根位姿与29关节状态，不重新进行人体到机器人重定向。
3. 从声明的源帧率转换至20Hz，将模型输入限制在最多200个状态点，即最长10秒名义窗口。特征描述相邻状态，最多199行；批次加载提供长度与补齐掩码。
4. 提取22个对应关键点，将世界Z-up转换至HumanML的Y-up坐标，规范初始朝向至Z+，编码263维特征。
5. 按来源名义主体分组拆分；仅以训练集274,426帧计算归一化统计。
6. 检查全部文件哈希、数组维度、有限值、20Hz时间网格、29关节保留、跨拆分完全相同特征、统计复算，以及两个来源在三个划分中的加载。

没有对GRAB或OmniRetarget做额外的物理动作筛选或逐条语义筛选。所有3,161条仍在最终清单中；旧的2,609条候选阶段不再作为主数据集。

## 数据位置和使用

服务器数据目录：

`/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/grab_omni_29dof_weak_baseline_v1_20261103`

加载时的`dataset_root`为其父目录：

`/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029`

- `manifest.json`：3,161条轨迹、原文字及其他来源描述、文件相对路径、哈希、拆分、20Hz和裁剪信息。
- `stats.npz`：训练侧特征与空间统计。
- `feature_audit.json`、`package_audit.json`：技术格式、统计和加载验收。
- `paper_scope_completion.json`：此次数据构建范围的完成收据及上述文件哈希。
- 引用的NPZ保存263D特征、22点、采样时间及完整29关节状态；Omni对象状态在其源布局后7列，GRAB对象状态在独立字段中。

该目录采用已校验的相对路径引用既有轨迹文件，不复制大型轨迹数据。迁移时须连同`manifest.json`引用的文件一起迁移。

## 范围和改编说明

[DreamControl-v2 III-A4、III-B1](https://arxiv.org/html/2604.00202)说明对GRAB和OmniRetarget不施加所述预处理筛选，并使用20FPS、最长10秒、朝向规范化及22点263D表示。本项目按用户要求使用29DoF；DexWeave的现成GRAB机器人轨迹和MotionHub文字是用户指定的来源替代，编码中的FIR四元数分量滤波、signfix及当前机器人关键点实现属于明确标注的改编，不声称作者原版实现或完整Full-Mix。

来源文字作为源序列级弱监督保留，未宣称逐句独立证明真实接触、他人交接或物体开关状态。513条10秒截断与67条已知offhand文字风险保留在元数据中，不删除对应轨迹。263D表示不保证无损恢复29个关节角，原29关节状态另行保留。

裁剪实现补充：目前超过10秒的序列保留起始200个状态点，时间为0至9.95秒，并沿用完整源描述。论文III-B1和IV-A实施细节明确20FPS及10秒裁剪/补齐，但未说明裁剪起点、是否随机/多窗口采样，或裁后文字如何调整。因此“从头截取并保留整段原文”是本基线的具体实现选择，不是已经核实的作者实现。513/3161（约16.2%）是存在裁剪风险的轨迹比例，不是已测出的错误配对率；来源关联正确不保证裁后片段包含描述中的全部事件。补齐掩码处理有效长度，不能解决被裁掉事件的文字噪声。当前交付为遵循公开预处理步骤的来源弱监督基线，未作额外重标或改动轨迹。

数据构建完成不等于已经训练模型或证明所有上游使用权；使用时遵循GRAB、MotionHub、DexWeave及OmniRetarget的来源条款。历史`training_ready=false`与语义核验字段保留，记录的是更广的训练/语义状态；本次完成以`paper_scope_completion.json`中的数据构建范围为准。

来源：[DexWeave](https://huggingface.co/datasets/naichuan04/DexWeave)、[MotionHub GRAB文字](https://huggingface.co/datasets/ZeyuLing/MotionHub/tree/main/GRAB/hierarchical_caption)、[OmniRetarget](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)、[OMOMO](https://github.com/lijiaman/omomo_release)。
