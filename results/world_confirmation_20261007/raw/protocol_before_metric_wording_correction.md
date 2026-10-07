# 残差世界模型：新动作集单次确认（采集前锁定）

三种子开发triage均PROMISING，不能用选择过checkpoint的validation作独立确认。旧66参考已全部参与历史工作，原11test退休。新动作确认只检验同Kick生成分布，不代表跨任务泛化或实体接触。

## 新参考生成及固定选择

冻结官方OmniControl权重SHA886b722e95496f4ceafd704c4e1d0578888733d12ce67429677b80a91a9c36d0；原官方Kick文本/空间hint前32项，生成seed20261007、1次重复、196帧20fps、原1000扩散步。采用已有text_only加载适配，只移除不需要的训练数据加载。生成32条，不补抽、不换seed。原PyRoki retarget及官方refine源码逐字复制/hash相同，排除额外IK。

在任何物理结果产生之前，按原数字ID升序选前11条满足以下静态接口条件的参考：原输出未被retarget拒绝；215帧×27关节/根位置和215×4四元数有限，四元数模误差≤0.001；关节不超官方URDF上下限0.0001rad；grab_idx在215帧内；qRMSE<0.01rad且rootRMSE<0.001m的原66或先前接受候选近重复不接受。去重固定215帧时间对齐、不time-warp；root减去第20帧初始平移并反旋其yaw，q不改变。仅静态选择，不按控制或预测效果选择。不足11条则BLOCKED，保全全部32条和拒绝原因，不自动再次生成。所有32条区分静态拒绝、合格未选、选中，不能将未选合格输入算失败。选中目录和每文件hash写manifest并锁定。原库加载截取前200帧、原Kick事件重算保持不变；静态grab_idx不是实际任务事件评分。

## 单次物理确认

每collect seed42/43/44，2048环境480控制步，原冻结随机actor/std，原奖励/PD/termination；只更换锁定11参考。共2,949,120新的真实转移，不作PPO/辅助更新。各1024clean/push按reference/初phase0.2s rank交替，step50..59 ±80N torso worldY；外力/方向/时钟不进actor。完整记录done、reset、forcebuffer。确认数据永不进入warmup、norm、辅助训练或checkpoint选择。

对各seed对应开发后冻结的六预测器（两个Ridge、Direct、Residual正确/未来动作置零/错配）仅推理一次，无任何拟合或重选择。确认前冻结六模型/checkpoint、ridge系数、input/target norm、schema/decoder源码hash。开发保持四reference折OOF，不因确认改为三折。每新reference每域最多1024个窗口，独立固定采样rng=20261007+seed；少于1024就全取，任一reference/域少于256合法窗则BLOCKED，不减少动作分母。10历史/10未来必须同reference/episode，恢复仅step60..109且属于施力起始episode，H10不含未来施力。无置换排除：错误模型已在开发集拟合，确认只比较同一窗口。reset排除仅保证预测定义，所有真实reset/中断事件另报，不能称存活成功率。

主预测指标H10根坐标全身URDF FK误差：每窗在全部link位置及xyz维度计算RMSE，先在每reference内平均窗口RMSE，再对11reference等权平均；沿用开发主统计。paired bootstrap10000次reference为单位，3seed分别报告。clean与recovery均要求正确残差同时比两个冻结Ridge及残差未来动作置零改善≥10%、95%CI下界>0；对Direct配对误差比95%CI上界≤1.05；对错配改善CI下界>0。3seed全部通过才打开原nominal预测门。保留root线/角速度、gravity、根高、足滑、contact AP/Brier/类别比例副指标。不能用根坐标FK证明世界目标、真实击打或控制收益。

若仅recovery通过而clean失败，原nominal预测门保持FAIL；需另行锁定扰动域PPO假设，不悄悄改变原协议。任一确认失败，不换模型/阈值/动作/seed重复test。

通过仅说明预测器可进入后续PPO消融；还需当前Kick实际仿真本地可视Gate、五臂三seed等真实PPO预算/因果控制结果及后续冻结MuJoCo。所有已有否定结果保留。
