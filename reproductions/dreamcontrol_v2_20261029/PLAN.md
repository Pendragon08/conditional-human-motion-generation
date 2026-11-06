# DreamControl-v2 仿真复现：执行计划

## 2026-11-06 12:33 UTC｜首批实际准入16条，闭合完成日志安全归档

- 首批Pick cohort_0008完成32条真实采样、16条运动学及原生几何准入，result SHA c8431040a424f78133d5af5168e99f375e714b2f9353d116e64935d640c7ea4d。全部raw/报告和准入状态/原生证明已独立逐SHA读回，证据reports/paper_v11_monitoring/cohort_0008_independent_byte_verification.json。当前Pick17/Pour1/搬箱2/推拉1=21/2000。第二批cohort_0009的32条raw已完成、worker1990480后优化，父1969082健康；质量1969083/SM1969084/视频1969085及Windows59168继续等待，未开展后续评估/新训练。
- 独立完成日志归档器1997516已通过五项自测并启动，S/completed_cohort_log_archive_v1/{status,manifest}.json(l)为索引。只处理本版全部阶段exit0且native_pass>0的已完成cohort日志；不改绑定生成源码、不动活动/失败日志。源及gzip独占F_WRLCK lease持有至释放，两遍完整gzipSHA/解压SHA、最终源身份/fullSHA后安全归档首份20,277,220B日志为2,241,603B，释放18,035,617B，额外独立读回再通过。gzip SHA7d3f952b2a2c012242c5dbf2dda8816cc4ac26c4bf1118e62e7f8ac35a73af7e，源码SHAb87760fc131d70201db445d2cd3ba609ca17395352dd088fb487e5d802cada02。原路径有.archived.json标记，完整字节在旁边.gz；不能因旧.log缺失重做批次。恢复src/restore_paper_v11_log_20261106.py，保持归档/拒绝覆盖不同内容。
- /data复核11,444,043,776B，root2,259,537,920B；本次小额释放仍不足全程容量，继续动态准备，正式11GiB及实际预算、生成/训练运行底线均不降低。C上次12.55GB/D5.56GB覆盖旧低容量段。详细收据reports/paper_v11_monitoring/PROGRESS_20261106_1233UTC.md。没有新增PPO/物体世界模型/在线闭环实验；老师建议和论文方向仅讨论。

## 2026-11-06 08:58 UTC｜首批32条采样完成进入后优化，v11增量发布已核

- paper_v11生成1969082运行正常，首批pick/cohort_0008的32条真实raw已完成，08:52进入后优化worker1972772；未完成原生准入，当前计数仍Pick1/Pour1/搬箱2/推拉1=5/2000。质量1969083、SM1969084及视频1969085均等待正确完整收据，Windows59168命令身份仍为v11渲染。勿重复启动或改绑定源。
- 首次v11发布在读取旧Git完整目录树时SSH响应超时，发生于创建树/提交之前，失败日志完整保留。新gzip响应中继只改变传输，GitHub请求仍在服务器发起，同一快照安全接续成功；提交02bae55a1cc7fb285a8ea7ac3f7629ef08069a1b，50条目、1,258,246源字节逐git blob验证，其他条目完整保留。收据reports/publication_inventory_20261105/paper_v11_code_20261106_receipt.json；已发布快照不重跑。
- 08:53只读全项目空间盘点：大于100MB文件均为受保护原始数据、最终权重或依赖，无可直接释放的大型候选；日志合计约98.9MB，不能凭后缀直接删除。/data11,463,716,864B；root2,260,025,344B，仍接近2GiB+100MiB保护线。当前C已实查12,553,674,752B、D5,557,526,528B，覆盖旧C低空间快照。未复制大型文件或删除任何源。后续按实际每批写入修正容量，正式训练最低11GiB和实际预算仍不降。


## 2026-11-06 08:34 UTC｜四类先导全部通过，paper_v11批量采样已启动

- 固定已采推拉20300007仅改变优化初值、完整重新拟合同一raw，运动学及独立原生180帧六项通过。result SHA48bf1ad044b6cad55cec8007c67efa6b0fdd201c5c827ddf0a91fb16549af2c4；状态ee42a8f998cc85b97bd69bcc0d87fea2eb07d1569727b47cf9318d1c10d0d332；原生证明1a1b8f133bb4909eccb9dbe7b636fbd55ff571581628423866f69c5315807869。腕位19.949→3.995cm、避障7.075cm、最小桌面距离+1.644mm；raw保真6.650cm/峰值21.303cm仍为诊断，不认证动态推拉成功。
- 独立CPU读回FK与保存点差5.958e-8m，输出对旧种子最大关节差0.9791rad、根位8.665cm；生产初值与成功诊断经过相同限位裁剪后逐值完全一致。旧13000053及同场景最近12条仅作优化初值；每条新raw完整重新拟合，未替换生成动作、目标或损失，未放宽论文筛选。
- paper_v11真实启动：生成1969082、质量1969083、SuperMimic1969084、视频源1969085。计划S/paper_pipeline_v11_launch/config.json SHAf60701d5ff638c3b0e5f3b43e7f0722cfbbce5cb68767427338e56f6bdd41c62。S/new_references_paper_v11/pilot_receipt.json明确all_four_native_passed=true；全部既有23条raw及拒收前缀已核，无重采，准入Pick1/Pour1/搬箱2/推拉1=5/2000。当前pick/cohort_0008首批32条实际采样worker1969093；候选不算准入。每类总1000候选及零准入批停止规则保持，不重启旧v10。
- 后续四类各500→evaluation_paper_v11五指标→supermimic_paper_v11新库原生训练、全2000身体跟踪与四项各100回合动态评估→四参考及四实测仿真视频。质量/训练现在仅等待；无新PPO、物体世界模型或在线闭环实验授权。用户本轮研究逻辑讨论不改变实验范围。
- Windows渲染59168已核为新render_report_videos_paper_v11.cjs（PID可能被系统复用，须看命令行），日志reports/paper_video_renderer_v11.*，输出D:/DreamControl_v2_paper_protocol_report_v11。旧渲染58412已退出。首轮输入待完成，视频尚未产出。
- /data08:32约11,469,127,680字节。按原保守1,396,736字节/候选、全通过假设，余1995参考估约2.786GB，超过现有生成底线上方约2.401GB；需继续按实际写入准备安全空间，正式训练更需至少11GiB及实际预算。无删除、底线不变、最终模型/原始数据/参考/所有冷档保留。新源码/证据待增量GitHub发布，旧上传勿重做。


## 2026-11-06 07:59 UTC｜推拉八条先导全拒，已有样本初始化诊断启动

- paper_v10推拉20300000–20300007共8条全部拒收，生成1946814、质量1946815、SM1946816、视频1946817及Windows58412已退出并保留failed_preserved。新协议真实raw共23条：Pick8/Pour2/搬箱5/推拉8；准入Pick1/Pour1/搬箱2/推拉0=4/2000。尚未批量、质量、新SM或新动态评估。禁止重启v10/扩大同配置先导。
- 八份后优化报告已逐SHA重读，最佳腕位各19.95–24.45cm仍超15cm。最小腕位分支的桌面穿透各5.40–16.64mm，部分腕避障不足3cm。此处5mm穿透仍为项目数值几何适配，搬箱/推拉不是论文同定义任务。证据reports/push_paper_v10_failure_20261106/readback_0651utc.json。
- 对固定已采20300007独立CPU逐帧FK：raw左右腕最大23.84/16.84cm；Stage A最大31.34cm，70帧左腕x=-.039/z=.759m而目标x=.241/z=.900m；Stage B最佳19.949cm/穿透6.013mm。原sampler单腕报告16.84cm与后优化双腕23.84cm为统计范围不同，不据此称坐标错误。推拉仍默认q0，初始化问题需对照验证。
- 仅只读旧推拉500库最近12条，匹配同场景/同180帧相位。旧13000053对当前目标腕位1.739cm/避障7.596cm、对新raw保真13.778cm，只作初值不计样本。S/push_initialization_diagnostic_v1/cpu_report.json SHA2ff7b2bbb3911bd982df1a99a342533bf88bc809eed48d76353df9741824342a。
- 隔离控制器1967045已启动，S/push_initialization_diagnostic_v1/{launch,status}.json、postopt/candidate_20300007/report.json权威。源diagnose_push_initialization_20261106_v1.py SHA328b2ffa5e62354176e64f727d0058eaa89b3560092fc00c6775a5c3c9b452c5，固定v10计划SHAe074e9b85998355a6588a0d1d711495014e587b724adaa5ee70a307b3986798a。同raw/目标/损失/筛选完整重新拟合，仅初值改变；运动学过才独立原生探针，随后独立读回、与种子区分并新版本安全接续。失败先读证据，不盲扫或重采。
- /data复核11,471,736,832字节；资源底线、旧PPO取消、受保护数据和最终模型保持。首次远端调用早于SCP完成报文件未到，未启动Python工作；SCP完成后仅一次启动成功，没有重复作业。新增诊断代码与证据尚待增量GitHub发布。

## 2026-11-06 05:20 UTC｜搬箱同样本初始化对照通过，paper_v10安全接续

- 05:23UTC本轮修正/诊断/接续证据已增量发布GitHub提交53f33b93b99e472030fb1883670a78d16724fb4c，55条目、1,232,156源字节逐git blob校验，其他条目保持。收据reports/publication_inventory_20261105/paper_v10_code_20261106_receipt.json。全部GitHub网络由服务器发起，不重复同快照。生成1946814继续推拉第二先导20300001后优化1947718；仍4/2000，资源正常。
- 前四条搬箱的通用q0初值使Stage A右腕卡在桌沿前下方，后优化最佳39.02–40.66cm，按论文15cm门槛实际拒收。固定20200000 raw/目标/损失/筛选，仅改为新通过20200004初值后完整重拟合，最终腕位5.994cm、避障3.580cm、穿透4.447mm，原生180帧六项通过。raw保真6.007cm/峰值21.722cm、腕角0.3630rad、躯干接触728.44N诊断保留，不认证搬运成功。
- 最终状态SHA2b9749e62dbaee45c7c2a865813efc4012ae5a1f43102bef0bfd1311a1923a64，后优化报告3ad19a5f16436ebed380939d1bf5741212666243cfda1684a39846745abb94bf，原生证明518b2f84069907cd08c72990841f750cd12ac13bfe519ba850b22a77a52ecbb6。独立读回、与种子区别及生产初始化CPU见证全通过，本机reports/box_initialization_20261106证据逐SHA核验。
- v8父1916624等待当前推拉20300000完成后有意SIGINT结束，所有完成前缀保留；推拉第一条22.264cm拒收，未重采/重拟合。v9接续在CPU见证因冗余Slerp归一化2.384e-6差异停止，未启动作业，不重启。v10同相位搬箱直接复用初值字节并重新按新raw拟合，其他条件/损失/论文筛选不变，所有旧版本证据保留。
- paper_v10已真实启动：生成1946814、质量等待1946815、SM等待1946816、视频源1946817。S/paper_pipeline_v10_launch/config.json SHAe074e9b85998355a6588a0d1d711495014e587b724adaa5ee70a307b3986798a。S/new_references_paper_v10/{status,progress,resume_receipt,pilot_receipt}.json权威，实际准入Pick1/Pour1/搬箱2/推拉0=4/2000，第二推拉20300001采样中。全前缀16条raw已核，各类先导8/总1000预算未变。
- 四先导全过后各500→evaluation_paper_v10五指标→supermimic_paper_v10新训练/2000身体及4×100动态评估→视频。等待不算已训练/评估。Windows渲染58412，render_report_videos_paper_v10.cjs，D:/DreamControl_v2_paper_protocol_report_v10。旧renderer59168已退出，无重复。
- /data11,479,408,640字节，当前生成正常、正式训练仍受11GiB与实际预算约束；无删除。旧PPO继续取消，扩散10000不重训。GitHub本轮修正和新证据待增量，已上传历史模型不重传。


## 2026-11-06 05:04 UTC｜搬箱第5条已过；按用户要求做同样本初始化对照

- 搬箱20200004通过后优化和独立原生180帧六项，腕位5.292cm、避障3.489cm、桌面数值穿透3.883mm；状态SHA83abd3d511771e1890bb2d9af49fab5520949b418365a8f1b8e16885bc111a51，原生证明SHA34e5f64cb88923a9c8a9d4b3f8fac53804c303e58ce3672357f23f1fc61def16。当前新库Pick1、Pour1、搬箱1，推拉0，共3/2000，未批量、未质量评估或新SM训练。
- 前四条raw腕位22.06–24.12cm，Stage A后右腕卡在桌前x约0.11m/z约0.73m，目标x约0.40–0.43m/z1.04m；Stage B最佳仍39.02–40.66cm。v8只有Pick/Pour使用已验证初值，搬箱/推拉仍通用q0。报告reports/box_initialization_20261106/ANALYSIS.md；原阈值及损失未改。
- 单条初始化对照控制1938522、worker1938529，S/box_initialization_diagnostic_v1。固定已采20200000，使用本轮已过20200004仅作初值，完整重拟合同一raw；无重采、旧样填充。CPU报告SHA9a5f4463087aa470e12ff84fafa3029359cc4df489e7d4e17684a326d2ddbe96。Stage A中间读回保真13.287→4.260cm、腕位44.578→15.977cm，最终及原生仍待结果，不预判通过。
- 为不修改绑定活跃源，接续监督1940249按真实身份暂停v8父1916624发新工作，但允许当前推拉20300000/worker1937140完成后优化；S/box_initialization_handoff_v1维持资源守护，完成后有意SIGINT父控制并保留所有前缀，旧status的KeyboardInterrupt不是算法崩溃。待对照全门槛及独立原生通过才启动paper_v9，重用全部raw/完成后优化/原生证明，推拉仅补未完成原生。v9源码已准备未部署，禁止直接重启旧v8。
- 两实际作业合计GPU约21.04GB，符合24GiB项目上限，单作业约10.52GB符合12GiB；/data约11.48GB，底线未变，无删除。旧PPO仍取消。

## 2026-11-06 04:08 UTC｜新Pour先导已通过，双手搬箱继续先导

- 新采样Pour20100001（第二次先导）运动学与独立原生180帧六项均通过，S/new_references_paper_v8/pour/cohort_0001/result.json SHA8c28e54a130ea5274819aac677488e977c91d1edd51023461d3f21abb706d6b4。最大腕位11.858cm、腕点避障最低3.449cm，桌面数值穿透1.992mm；原始保真RMSE11.525cm/峰值35.492cm、腕角0.9495rad、帧间腕角0.3599rad、肘目标RMSE18.078cm均保留诊断，不冒称动作质量或抓持/倾倒成功。
- 独立重读raw/状态/后优化/原生证明SHA全部一致；新raw的180帧右肘19与腕21提示均实际写入采样mask，条件源SHАf1bcf702296ffa664223e8a41f8072cb6b64f3affd21612f39ff62bf3bd2211a，不再只是旧raw后优化补肘。状态SHA97c6b74b3542b242c4d78eea8d1331830ab080d2540530b229c0fa1616bdf338，原生证明SHA4bc2032edcb8a17d8eb2c5e99e2c4215348b869ba703c9d51dc9ec880a60ad07；相对初值最大关节差1.1644rad、根位21.383cm。仅几何探针，物体仍为计划位姿注入。三份证据已下载本机reports/pour_paper_v8_pilot_20100001并逐SHA核验。
- 04:08UTC生成1916624正常，双手搬箱前两条拒收，第三条20200002真实采样worker1928997。新库Pick1、Pour1，共2/2000；推拉尚未开始，四项先导全部通过后才批量。原8次先导/1000总候选上限与论文筛选未改，无并行重复作业。质量/训练/视频仍等待。
- /data11,486,572,544字节，当前资源正常。只读盘点了本项目大文件/日志/归档目录，>100MB主要为受保护原始ZIP、最终模型和依赖；没有删除或转移。最近单条分配上界1,396,736字节，按1998条全通过的保守估计约2.79GB，可能接近生成底线，正式SuperMimic另需至少11GiB和检查点预算。预算为估计，继续以实际写入/资源判定，不能为腾空间动受保护资产。


## 2026-11-06 03:35 UTC｜v8 增量代码与证据已发布，先导按既定预算继续

- v8代码与独立读回/初值见证/启动收据已通过服务器GitHub中继发布，提交02d94b1b50696cde64fcd07f04fa292b68ca584e；48条目、1,100,713源字节，全部git blob逐字节核验且其他文件保持。先复核25份服务器运行源与本机镜像SHA一致。收据reports/publication_inventory_20261105/paper_v8_code_20261106_receipt.json。所有GitHub流量来自服务器，不重复运行已完成快照。
- 03:35UTC生成1916624正常，Pour第二条20100001后优化worker1919316，已完成的20100000按论文腕位/数值几何拒收并保留；第一条最佳腕17.006cm、桌面穿透7.066mm，对应报告SHA3ecbcf974e88f820323465dc1caaeff2ae394d3812b5492366aec7802fab8fbe。仍在原8次先导预算，未新增诊断或调整条件/阈值。新库仍Pick1/2000，质量、训练、视频继续等待完整前置证据。
- GPU10,515,120,128字节/RSS2,142,707,712字节符合现有上限，/data11,489,738,752字节。生成控制正常，未清理文件或重启任务。


## 2026-11-06 03:18 UTC｜论文协议 Pick 先导恢复通过，v8 接续 Pour 新采样

- 既有新采样 20000004 的隔离初始化对照已通过运动学与独立原生120帧检查。S/paper_pick_seed_diagnostic_v2/result.json SHA d6e2668eed726c3f914ddb7e0549404b309b604a13f4ed1944db818541ccdcc2；状态 SHA31b7502f55a66ee0efff1169d7e197f264084af4d667cc05aa0ba1fd47c24340，原生证明1090b42a102e180712d57818dc7b3361abc0747c646c74f411b3801d4736b21d。最大腕误差38.692→8.439cm，腕点避障3.915cm；腕角0.3108rad、原始动作保真RMSE14.917cm、峰值37.251cm保留诊断，不能称质量已改善或抓持成功。
- 独立CPU重新读取raw/状态/原生绑定并计算FK，和保存点最大差5.94e-8m。相对相位对齐旧440初值，最终最大关节改动0.1281rad、根位8.405cm，确认重新拟合了同一raw。S/paper_pipeline_v8_launch/independent_pick_readback.json SHA6e3b994e665ff10c69810953c4b2327fc916789e0ba14a1d66c7d3781a3891a5。旧姿态仅初值，未新增采样或把旧动作填入库。
- 修复验证与生产初始化不一致：Pick按同一已核近邻/相位方法选初值；Pour按已通过v7原生验证的姿态作相位对齐初值；均完整重拟合各自新raw，不改条件、目标、损失和论文15cm/3cm及数值几何检查。新模块CPU复算重现Pick诊断初值，并确认Pour同相位输入恒等，S/paper_pipeline_v8_launch/initializer_cpu_witness.json SHA9f50527b85c269f3203a82d5491ab614a6417a0f1cff623d661a9a74a13cb8fe。后两任务沿用v7初始化，仍须先导验证。
- v8已真实启动：生成1916624，质量等待1916625，SuperMimic等待1916626，视频源1916627。S/paper_pipeline_v8_launch/config.json SHAeb3c79702a08589d7eb73a40605e1f0d1515254420bdfbd20bfbf9de72d2a0c2，权威 S/new_references_paper_v8/{status,progress,pilot_receipt,resume_receipt}.json。保留Pick八条raw与原拒收证据，只将20000004引用新通过证明，已准入1/2000；03:18UTC Pour20100000真实采样worker1916637。四任务全过先导后再批量各500，每类8先导和1000总候选上限不变。旧v7及v5/v6失败不重启，绑定源不改。
- 后续严格依次新库完整→S/evaluation_paper_v8五指标→S/supermimic_paper_v8训练/全部身体跟踪/四项各100动态回合。新等待器不是开训。原Viser视频源S/report_video_sources_paper_v8；Windows渲染59168，src/repair_prior29_v2_20261105/render_report_videos_paper_v8.cjs，输出D:/DreamControl_v2_paper_protocol_report_v8，日志reports/paper_video_renderer_v8.*。SSH临时超时已恢复；本机接续61484正常完成，reports/paper_v8_connection/remote_launch.json有服务器收据，勿重复启动。
- /data 11,491,360,768字节，当前生成资源正常，正式训练仍须至少11GiB及实际剩余预算；未删文件或降低资源底线。最终模型/旧275权重和冷档已发布状态保持。本次v8代码与新证据仍待补GitHub增量发布；所有GitHub网络继续从服务器发起。


## 2026-11-06 01:26 UTC｜v7 Pick 八条先导全拒，隔离初始化对照已接续

- S/new_references_paper_v7/status.json 已 failed_preserved：Pick 8/8 拒收，后三类未进入先导，新协议0/2000，禁止重启v7或增加同配置先导。原raw、输出、报告均保留。
- 条件诊断1914254已确认原生成退出后启动后优化worker1914522，S/paper_pick_seed_diagnostic_v2/status.json phase_running/postopt。只改已采20000004的求解初值为相位对齐旧440，重新拟合同一raw；没有修改目标、损失、15cm/3cm筛选或几何检查，没有新增采样。全部运动学过才自动独立原生探针；不自动把旧种子计入新库。


## 2026-11-06 01:24 UTC｜新先导继续，最终权重 GitHub 读回完成

- 新协议 Pick 前7条拒收，第8条20000007后优化进行中；新库仍0/2000，后三类尚未进入先导。候选20000004原始腕点最大误差12.748cm，Stage A原始动作拟合RMSE16.895cm，后优化最佳腕点38.692cm，说明至少该条误差在后优化中扩大。未改变论文筛选值，未计作合格。
- 仅CPU对20000004比较同场景最近12条已验证旧姿态、按接触相位对齐；选旧440只作优化初值，腕目标最大6.102cm、相对新raw RMSE18.377cm，不是新参考。报告S/paper_pick_seed_diagnostic_v1/cpu_report.json SHA a409534c6a604189442cda8aef6eb4319ea73836d0699540962edb1a4f75d96f。v1等待器把pilots_in_progress误判为无需诊断，已退出，未做GPU；v2复用已核CPU证据修正状态分支，不重采。
- 条件诊断控制器1914254，S/paper_pick_seed_diagnostic_v2/{launch,status}.json；脚本SHA95fcd1101979a979644dd5820868d5878927d494c9ac12150ef07e06e93ce3f6。只在Pick全部8次失败、原控制器退出后，同一20000004用旧440初始化重新拟合原raw；原条件、损失、论文15cm/3cm和几何检查全不变。若Pick先导通过则退出；通过运动学才独立原生探针。禁止并行重复启动或修改活跃源，不自动把诊断填入库。
- 修复扩散最终10000权重已完整上传、整TAR和4成员SHA读回核验通过。asset715377849，503853568字节，TAR SHA b7d8fb3053e5723d87403f0c58becf6725869267c5e5fd5a2957328734aaef78；模型原SHA保持。远端R/publication_snapshot_20261105/repaired_final_v1/receipt.json SHA435940d48cf826ff24ce081edf16b061e8229de52685cb9153a292d0d58c9b23，本机reports/publication_inventory_20261105/repaired_final_v1_receipt.json逐SHA相同。无源释放，不重传。


## 2026-11-06 00:46 UTC｜先导与发布实际状态

- 新协议前两条 Pick（20000000/20000001）真实采样后均拒收；最佳最大腕位分别75.54/36.11 cm，且桌面数值穿透检查未过，不是因已移除的保真/关节窗拒收。当前第三条按原8次先导预算继续，新库准入仍0/2000。旧10100005仅验证通过，不计新库。先导如耗尽须保留失败并诊断，不重复启动同配置或追加预算。
- 增量代码/证据 GitHub 提交 ffc1418f65ce178f23d4893979075ab2650f3ee0，80条目逐字节核验，其他仓库内容保持；收据 reports/publication_inventory_20261105/repair_code_20261106_receipt.json。修复扩散最终10000权重独立上传1904106，R/publication_snapshot_20261105/repaired_final_v1/{status,receipt}.json。当前上传/完整读回中，不删除源、不重复传；最终需 receipt published_readback_verified。
- Windows原Viser渲染等待器63996真实运行，日志 reports/paper_video_renderer_v7.{log,err}，仅等待正确新输入；尚无新参考/仿真视频完成。


## 2026-11-06｜论文协议 v7 复验通过，真实新采样已开始

- 已按用户最新授权移除论文未给出的原始保真、非腕位移、关节/根修正窗、腕角和帧间变化硬拒收；保留原动作拟合、平滑、脚稳定目标和完整诊断。公开腕目标误差 <15 cm 与腕点避障 >3 cm 分开，数值完整性/关节限位/独立 FK/穿透检查保留。历史“不得放宽旧门槛”不再适用于这些自行加入的限制。
- Pour 10100005 原样本重拟合及独立原生六项检查均通过，S/paper_protocol_validation_v7/result.json SHA 305347a6cc4b4b275a44b2541be6e74606893396f7fec058239f928a73fed318。最大腕误差 7.253 cm、桌面距离最低 4.152 cm、相邻帧腕角最大 0.06833 rad；腕角 0.39439 rad、原始保真 RMSE 3.884 cm 和肘目标偏差 15.810 cm 均保留诊断。原生是姿态几何探针，不是抓持成功。新增肘提示在该旧 raw 中只用于后优化验证，不计入新库。
- 新采样控制器 1899660 实际运行，S/new_references_paper_v7/{status,progress,pilot_receipt}.json 为准，计划 SHA a36fcedbfccbaeb8cd065457f72e76c601d233c6bd59759db32481c7fb68db74；新 ID 从 20000000 起，首次 Pick 已真实采样并进入后优化。无旧样本填充。四类先各最多8次先导，通过后逐批各500，候选上限1000/类。
- 质量等待器1899674、SuperMimic等待器1899675、视频源等待器1899676已启动；依次等待新库完整、质量收据和同轮评估结果，不是已训练或已评估。固定分词215条回放一致，新 Pour 文本分词已完成，SHA acacac6498c1d3e2718375e7ea50f1da227bb6b60d339fd7e53fa0c43eb3b8cf。
- 新库五指标→原生 SuperMimic→全部身体跟踪及四任务各100回合动态评估；四参考、四实测仿真视频输出 D:/DreamControl_v2_paper_protocol_report_v7。成功率与交互点误差分别记录。Pour 评分读每条接触时刻。旧 PPO 不恢复。纸面未公开的29DoF坐标适配、Pour角曲线及搬箱/推拉任务配方继续显式标注，项目留出不是论文 Full-Mix。
- /data 当前约11.50GB，生成运行门槛未变；下游正式训练仍需至少11GiB及实际剩余写入预算。C盘低于保护线，不追加大复制；D盘现约5.63GB，后续仅按完整可恢复副本及源身份/独占lease规则处理确实闲置产物。所有原始/参考/活动/最终模型及旧冷档保护。


## 2026-11-06｜用户授权按论文条件和筛选重建（覆盖历史自加门槛）

- 用户明确要求取消自行加严限制。论文 V-D 腕目标误差 <15 cm，腕点避障 >3 cm；后者不能当腕目标阈值。原始动作保真、非腕位移、根/关节修正窗、腕角及帧间变化改为诊断指标，仍保留姿态、平滑、脚稳定与避障优化；保留关节限位、有限值、独立 FK、数值穿透检查。论文未给出的角度曲线、alpha_l 和 29DoF 场景坐标适配显式标注。
- 独立 `src/repair_prior29_v2_20261105/paper_protocol_v7` 修改 Pour 肘提示与腕姿共享相位；先在保留的 10100005 上验证，再新四类先导与各500。现有 raw 加肘目标仅验证后优化，不能声称该 raw 生成时已有肘提示，不能算新协议生成样本。旧成果保留，旧 v6/PPO 不恢复。
- 当前尚在检查/部署准备，不能称已生成或已开训。新库完整后五指标→SuperMimic→四项动态评估和视频。资源与正式训练磁盘门槛保持，GitHub 全部流量继续服务器发起。


## 2026-11-05 23:44 UTC｜Pour 已采候选对照结束，仍未通过准入

- 候选 `10100005` 使用旧 `11000435` 仅作初值，完整拟合新 raw 后仍拒收。最佳腕位置分支的最大误差为 6.683 cm / 0.24979 rad，超过原门槛 3 cm / 0.20 rad，且关节时间连续性未过。相对新 raw 的保真 RMSE 4.306 cm。报告 SHA `905639e8e1aa7e51d80cc49860299f9b85ad6874db84d12db9461966e28f2687`，本机只读回执 `reports/pour_existing_contrast_v3_readback.log`。
- 新库仍 Pick 1/500，Pour、双手搬箱和推拉箱各 0/500。Pour 八条先导已拒收，后两类尚未进入先导。质量评估、SuperMimic 重训、四项动态仿真和视频等前置条件齐备再接续。此前的生成及等待器已退出。后续先从现有样本只读核对 Pour 的空间条件、腕姿目标和阶段时钟，不重复同配置试跑或放宽门槛。

## 2026-11-05 22:42 UTC｜从已采 Pour 中启动一条对照，无新采样

- 八条只读比较已回，`reports/pour_existing_candidates_v1.log`。按倾倒阶段右肩肘和全身与同场景、同相位旧物理解的偏差，10100005 最小，肩肘 RMSE 13.92 cm、肘 15.10 cm、全身 11.77 cm，旧种子 11000435。10100000 相应为 16.22/17.56/12.72 cm。10100005 自身原始腕位 9.48 cm 通过 12 cm 预筛。10100007 原始腕位 12.45 cm 未过预筛。旧近邻对新腕目标接近只是几何初值证据，不能计新合格。
- 按上述证据只用已采 10100005 做一次完整 raw 重拟合，复用相同近邻初始化求解器 SHA `5704cd44643d1bf0d60fa1a824e8c82f824d277a29ac02703f57f0a8cbc561f4`，同一目标/场景/时钟、最终阈值及预算不变。控制器 PID 1893357、worker 1893361，计划 SHA `0ea2432bfd38769c1568fb53136a342d594ccd2d5e9d460dd3fcc78242555120`，`S/pour_existing_contrast_v3/{launch,status}.json`、`postopt/candidate_10100005/report.json` 为准；仅全门槛过才独立原生探针。勿重复部署，不重采或对 10100000 继续盲扫。23:11 UTC SSH 多次超时，Windows 只读接续 38020，日志 reports/pour_existing_contrast_v3_readback.*，不得将网络错误判作求解失败。合格仍 Pick 1/2,000。
- /data 最新 11,501,813,760 字节。修复目录大于 50 MB 的文件只有需保护的最终权重；全项目大头是原始/参考/已用依赖及保留成果，没有据此释放任何文件。


## 2026-11-05 22:15 UTC｜Pour 固定基准修正仍拒收，转候选间对照

- 单条 Pour 10100000 的可行性修正完成，报告 `S/pour_feasibility_refinement_v2/postopt/candidate_10100000/report.json` SHA `b3dd6fb8335880673fee8d7dde8daedbc609564ff139e0f218bd3a075f8f98f1`。原始保真 RMSE 从 5.092 降至 4.878 cm，关节投影/连续性和其余原门槛过；腕位 6.201 cm、朝向 0.268 rad 仍超原 3 cm／0.20，未做原生探针。
- 原状态、原 Stage A 基准下的独立 CPU 全身逐帧搜索 `S/pour_wrist_pointwise_feasibility_v1.json` SHA `97ebc89b468a0e127161c1fd6250b200c92d1cd51f6102f90e5f1268474a3c89`。70/110 两关键帧的最小综合残余仍分别约 3.69/3.59 cm、0.229/0.226 rad、非腕峰值 10.08/10.18 cm（限 8 cm），腕俯仰触物理或 Stage B 窗边界。未找到同时可行，不能据局部搜索断言全局不可行。两项对照都只读旧成果或保留状态，无新样本、准入、训练或视频。
- 不再对同条盲改权重。现在仅比较已采八条 Pour 原始腕位与倾倒阶段肩肘和同场景近邻历史姿态的差异，寻找能检验候选特异性的对照。Windows 只读接续 34236，`reports/pour_existing_candidates_v1.{log,err}`，SSH 暂时不稳定，未重复部署或启动新 GPU。只有比较证据支持才用另一条已采候选做有界对照，不能放宽门槛/预算或用旧姿态填充。合格库仍仅 Pick 1/2,000。


## 2026-11-05 21:44 UTC｜Pour v6 先导耗尽，沿固定拟合基准验证约束修正

- v6 Pour 8/8 拒收，旧等候器因上游失败均停止并留存收据。Pick 已准入 1 条，其他三项仍 0，未触发质量、重训、动态评估或新视频。禁止重复启 v6/旧等待器、重采八条或降门槛。
- 单条 10100000 按旧近邻作初值重拟合仍拒收。报告 `S/pour_verified_seed_diagnostic_v1/postopt/candidate_10100000/report.json` SHA `50739d3f171db8a1117ba1e76d7ea2c700420b4f1a32b66a45d51c326f34d0cf`。最佳第三分支腕位 6.614 cm、朝向 0.296 rad、保真 RMSE 5.092 cm／峰值 15.18 cm，关节投影 0.001116 rad、帧间最大步 0.3065 rad，均超对应原门槛。其他门槛过。独立 CPU FK 显示 110 帧（倾倒保持末段）同时是腕位与腕角峰，70 帧右肘是保真峰，20→21 帧右踝俯仰是关节连续性峰。
- 只复用同候选最佳拟合状态做一次针对超限的可行性修正，保留同一 raw、Stage A 基准、桌面、任务时钟、关节窗、所有原验收阈值。新增求解残差对腕位、腕角、保真峰与帧间动作仅在超出略严于验收的带时罚，非腕及原碰撞/脚底约束仍在。全部实测门槛过才原生探针；仍失败先读证据，不重做同配置或盲扫。Windows 接续 62628，`reports/pour_feasibility_refinement_v2.{log,err}`，权威 `S/pour_feasibility_refinement_v2/{launch,status}.json` 和 `postopt/candidate_10100000/report.json`。勿重复启动，无重采样。


## 2026-11-05 20:17 UTC｜首条 Pick 通过原门槛及独立原生探针，接续其他任务先导

- v4 旧解仅作初值的重拟合结束，原报告 SHA `80bb37f4ae16d3b7187f8b809b799ee811e85ff0f93be719a576b5f211f2413d`。第三分支只有腕位 3.314 cm 未过 3 cm，原始保真 2.854 cm，其余门槛均过。
- 对同条 10000007 仅对超界腕位做最小范数根平移，41 帧最大 3.19 mm，目标/关节不变。重新核全部原运动学/碰撞/保真阈值后最大腕位 2.995 cm、朝向 0.110 rad，独立真实原生姿态几何探针的五项全部过。结果 `S/pick_wrist_mm_projection_v1/result.json` SHA `7a8541c64e139d749bd795920892bd63da7e024384581d2388119a34070a3a40`。独立 CPU 原始保真 2.887 cm，相位对齐后与旧种子最大关节差 1.131 rad、根位差 6.87 cm，真实重拟合且状态 SHA 不同。`independent_source_readback_v1.json` SHA `09b04f5a60bdf190b352db60ce1dc2b0778d4ac434c3508bd02253f965638404`。新库现 1/2,000，只有一条 Pick，无抓持成功或整库结论。
- v6 接续保留已有 Pick 八次原始采样及七个拒收，10000007 仅引用新独立通过收据，无重采、重复计数或旧样本填充。生成先做剩余三类逐项先导；四类过关才继续各500。后续 Pick 由近邻旧解作种子，完整重拟合每条新 raw，只有腕位唯一未过且最小根位修正 <4mm 时才试同一公式，并重核所有原门槛。后三类后优化保持 v5。先导 8/类，总 1000/类不变。v5 保留失败并停止。
- Windows 接续 PID 65172，日志 `reports/generation_after_pick_probe_v6.{log,err}`。权威 `S/generation_pipeline_v6_launch/launch.json`、`S/new_references_v6/{status,progress,pilot_receipt}.json`，不得重复部署。等待器 `evaluation_controller_v6`、`supermimic_repaired_v6`、`report_video_sources_v3` 均按完整真实收据触发，Windows v3 渲染在远端启动后接上。未完成先导或库不触发下游训练。无新 PPO。旧检查点、归档和11GiB/实际写入门槛保持。


- 20:41 UTC 回读：v6 远端 PID 1879655、评估 1879656、SuperMimic 1879657、视频 1879658，计划 SHA 94dcaeb8493dc55dc0e59590ab8f41b6cfd9aaffcc79ac39bc9e3e265428af7e；Windows 渲染 47436。八次 Pick 原始记录完整接上，1 准入。Pour 前三条已拒收（其中第三条源报告可能先于控制器汇总），按当前预算继续，另两项仍等待。邻近旧 Pour 状态只读对照，三条新 raw 的最佳近邻 RMSE 12.15–12.72 cm，旧姿态对新腕目标的误差约 0.46–1.08 cm / 0.009–0.036 rad。是可达性和候选初值证据，禁止用旧状态充数；尚未改当前其他任务求解器。/data 11,508,932,608 字节，按已观测上界投影剩余候选写入约 2.50 GB（不含额外拒收），需继续核空间，不能降运行或训练门槛。

- 21:11 UTC：Pour 前 6 条拒收，第 7 条后优化，合格仍仅 Pick 1/2,000。第 1 条 Stage A 原始保真 3.52 cm 且 raw 腕位 8.21 cm，但后优化最小最大腕位 11.68 cm、最小最大腕角 1.438 rad。旧 Pour 近邻可对齐任务腕目标的只读证据已核。已备独立的条件式单候选诊断：只有 v6 Pour 8 次全部耗尽并已失败退出，才从已采的 8 条中选 Stage A RMSE 最小且通过预筛的一条，以相同场景、相位和经 SHA 验证的旧 Pour 姿态作初值，完整重拟合新 raw 并复核原门槛。v6 Pour 如通过则诊断退出，无新增采样、并行 GPU、重复计数或降标准。Windows 接续 61840，`reports/pour_verified_seed_diagnostic_v1.{log,err}`，权威 `S/pour_verified_seed_diagnostic_v1/{launch,status}.json`，勿重复启动。

## 2026-11-05 19:16 UTC｜提前腕姿约束未过，旧物理解仅作种子对照

- 腕姿感知 Stage A 单候选复验 10000007 已结束，仍拒收。报告 `S/pick_wrist_anchor_diagnostic_v3/candidate_10000007/report.json` SHA `216cf8875c734b41251354885287922a0d134d9ad7ca0a9cd92e9a330ac9be33`。A 保真 9.98 cm。B 三分支最小最大腕位 30.43 cm，最小最大腕角 0.499 rad；保真最小 8.64 cm，不能用提前加权作为恢复方案。
- 只读核旧原生准入库中物体中心最近 12 条。旧手腕位置与朝向本身能通过新 3 cm/0.20 rad 子门槛，但相对各自旧 raw 的重建 RMSE 12.93–40.53 cm，0/12 通过新绝对保真与腕姿联合子集。报告 `S/pick_legacy_reference_comparison_v1.json` SHA `a1858a80903d6e1da54423a865c405333a8db034a665c9b2f8a9d1d96f3037f6`。这是历史协议差异和可达性对照，不否定旧准入，也不能视为新合格。
- 将这 12 条只读按新候选接触第 49 帧和抬升第 89 帧对齐。已有参考能贴近新腕目标，最大腕位约 0.71–1.32 cm、朝向约 0.022 rad；与新 raw 最接近的旧 ID5 仍有保真 RMSE 10.43 cm。报告 `S/pick_legacy_warm_start_readonly_v1.json` SHA `8f32511895f0a816f1a3ff49730afdb14cd68b4999c2443548ea3f062e76fa4c`。因此旧状态只能作有效姿态的求解种子，不能替代新原始动作或输出。
- 仅同条 10000007 的隔离后优化诊断使用旧 ID5 作 A 初始状态，随后以新 raw 重拟合，全桌面约束、原目标、关节窗与最终所有门槛不改，无新采样、准入或训练。Windows 接续 PID 48244，`reports/pick_legacy_seed_diagnostic_v4.{log,err}`。权威 `S/pick_legacy_seed_diagnostic_v4/{launch,status}.json` 和 `candidate_10000007/report.json`。启动收据已确认远端 1878099，计划 SHA e2e32dc77e9eb359373a8fe72be59e3d7ece781b91f1c1d210ad499fe8af63c7；19:41 UTC SSH 暂未读到结果，Windows 只读重连 31412，日志 reports/pick_legacy_seed_diagnostic_v4_readback.*。不重复部署，若仍失败，停止对同一候选重复改权重；若通过则核原始保真、与种子区分并独立原生探针。合格库仍 0/2,000。


## 2026-11-05 18:44 UTC｜全身逐帧搜索完成，检查腕朝向拟合基准

- 10000007 第 104/119 帧同时核腕姿、每点保真、非腕和根位的固定根朝向搜索无满足子集的结果，`S/pick_wrist_joint_feasibility_v2.json` SHA `7d3919029f9259e02f1559597c29e40d3de708e2c98c80798a8283e6c7f8618e`。固定根姿与 ±0.50 rad 窗下朝向残余约 0.73 rad。放开到全物理关节范围并允许根位移动的参考分支可到 3.19–3.29 cm、0.210–0.216 rad，但仍需约 1.51–1.52 rad 关节改变，且非腕峰值 9.3–9.9 cm 超 8 cm。该参考分支不用于准入。
- 进一步对同两帧开放全部 29 关节和根旋转，维持原 Stage B ±0.50 rad、根位 5 cm、各点和腕姿门槛，仍没有找到满足子集的解。两帧最佳折中腕位 7.50–7.65 cm、朝向 0.620 rad、非腕峰值 13.7–14.2 cm，腕俯仰触窗。报告 `S/pick_wrist_fullbody_feasibility_v3.json` SHA `a6c98b04a77334f671fccc1f76f479440069e020d62ee6aa873eae72b532b409`。局部诊断不证明全局不可行，无碰撞或连续性认证。
- 后阶段自由度搜索未解决，故只复用同条 raw 试验在 Stage A 拟合时加入腕朝向误差带，Stage A 仍拟合 raw 点且不吸收任务腕位置目标。复用分阶段桌面初始化和连续性残差。原 Stage B 修正窗、碰撞和所有最终验收阈值不变。Windows 接续 PID 36968，`reports/pick_wrist_anchor_diagnostic_v3.{log,err}`，权威 `S/pick_wrist_anchor_diagnostic_v3/{launch,status}.json` 与 `candidate_10000007/report.json`，控制器 1876477 已回读后优化进行，计划 SHA 259f6c114973e9b72cb41f4d0d072e5a1654c3f2ed0c958742a70e973a21e107，勿重复启动。若新拟合基准无效，不继续盲扫权重。若通过全部验收，先核 raw 保真和独立原生探针才恢复。
- 合格库 0/2,000。没有新样本、指标、训练或视频。旧产物、已完成归档和磁盘门槛保持。


## 2026-11-05 18:15 UTC｜腕部复验仍拒收，转逐帧可行性核查

- 同条 10000007 的受控腕姿 v2 复验结束，报告 `S/pick_wrist_initialization_diagnostic_v2/candidate_10000007/report.json` SHA `dff8f21f5917f183e7275a0a0c5579ecb0ef8c6e14637c8a917081f7be429c4f`。最佳腕位分支最大误差 10.29 cm、朝向 0.861 rad、帧间腕转 0.255 rad，均超原 3 cm／0.20／0.25 门槛；同分支关节修正 0.00123 rad 超 0.001，非腕 RMSE 3.25 cm 超 3 cm，保真 4.84 cm 通过。合格仍 0/2,000，无新采样或下游运行。
- CPU 独立逐帧逆运动学 `S/pick_wrist_pointwise_diagnostic_v1.json` SHA `e2b3a46ac3a6cde7ec818bae1aa03f0aee5cb5b73d0b171b5fa01e03009a4333`。第 104/119 帧姿态最难。在固定根朝向、Stage B ±0.50 rad 关节窗口下，纯腕目标折中仍偏约 5.5–5.7 cm、0.69–0.70 rad。全关节范围可以逐帧匹配腕姿，但示例解需要约 1.43 rad 关节改变，且非腕最大移动约 19–20 cm、原动作峰值偏差 22–23 cm。仅为点位对照，不证明整段必不可行，也不作准入证据。
- 已准备只针对第 104、119 帧同时搜索腕位/朝向、每点保真、非腕及根位原门槛的 CPU 诊断，固定根朝向，三种关节/根位置搜索边界，不做采样、碰撞仿真或训练。本机 SSH 接续 PID 62304，日志 `reports/pick_wrist_joint_feasibility_v2.{log,err}`；以 `S/pick_wrist_joint_feasibility_v2_launch.json` 和 `S/pick_wrist_joint_feasibility_v2.json` 为准，勿重复启动。若找到满足这些必要约束的解再定位时间连续性和碰撞，找不到只能判定当前受限搜索未找到，不代表全局不可行。
- 不重启 v5、扩大原样本预算或降门槛。/data 最近 11,514,916,864 字节，仍低于 11 GiB 正式训练底线，已释放档案保持。


## 2026-11-05 17:43 UTC｜同条 Pick 初始化对照完成，腕部姿态仍未过

- 候选 10000007 对照使用同一 raw SHA，分阶段只在初始化去掉桌面残差，后续恢复原值并执行相同验收。Stage A 保真 RMSE 5.57→3.05 cm，最小的最大右腕位置误差 19.61→10.70 cm，说明桌面加权初始化实质影响局部拟合，但未证明可准入。对照报告 `S/pick_table_initialization_diagnostic_v1/candidate_10000007/report.json` SHA `49d4879fed4583079e3f8e593dc9f2bc47234e9b2ab59f3126b0582569db5c3b`。
- 最接近位置门槛的分支仍有最大腕位误差 10.70 cm（门槛 3 cm）、腕朝向 1.042 rad（0.20）、帧间腕转 0.351 rad（0.25），非腕修正 3.238 cm（3 cm）与关节修正 0.504 rad（0.50）。同一分支保真 4.58 cm 已低于 5 cm。结果仍为 0/2,000，勿启 v5、原生探针、评估、训练或视频。
- 仅同一个候选的受控 v2 诊断在相同分阶段桌面初始化基础上，加入弱腕朝向初值和显式帧间腕转残差，不改末端目标、验收门槛或用户预算。无重采样。Windows SSH 重连 3232，日志 `reports/pick_wrist_diagnostic_v2.{log,err}`。权威 `S/pick_wrist_initialization_diagnostic_v2/{launch,status}.json`；重连收据已确认远端控制器 1874375，计划 SHA b9e0b5b5535fb98d6317c904e3a14e3617b9691b67c6c890362779b646739b2a，勿重复部署。若通过，先核全门槛再做一次独立原生探针。若失败，按逐帧位置及朝向证据判断几何可行性，避免无根据扩大预算。
- /data 最新 11,516,129,280 字节，仍未达正式 11 GiB 底线；713907427 的双遍验证和两个闲置源释放已完成，最终模型、旧成果均保留。


## 2026-11-05 17:20 UTC｜Pick 先导未过，保留现场并做单候选诊断

- v5 Pick 8/8 拒收，7 条在原始腕点预筛后未通过后优化门槛、1 条在原始腕点预筛拒收。参考库仍 0/2,000，其他三项未运行。生成、质量、SuperMimic、视频源及 Windows 渲染等待器均已停止并留存错误。没有新参考、指标、训练或视频成果。禁止重启 v5 或直接增加同配置的先导数量。
- 对七条保留的 Stage A 状态独立 CPU FK。右腕最大偏差常出现在抬举阶段，生成点 x 约 0.33–0.43 m，求解 x 约 0.11–0.12 m，且 z 约 0.75–0.83 m，低于或贴近桌面。桌沿局部极小值是待验证的解释。保留的 10000007 Stage A 保真 RMSE 5.57 cm。仅复用该候选试一次分阶段碰撞加权初始化，原最终桌面/地面、手腕、关节、连续性及保真准入阈值全部保持。无新采样、训练、策略或准入。Windows 重连进程 46380，日志 `reports/pick_table_diagnostic_v1.{log,err}`，远端启动与结果以 `S/pick_table_initialization_diagnostic_v1/{launch,status}.json` 为准；重连收据确认已派生远端控制器 1873231，计划 SHA 2c00248583a9b92f894a10159822c1b97afc994e821eb881342c6969cb70dd28；结果尚待回读。单项通过运动学检查后仍须独立原生探针，才考虑恢复四项生成。
- 归档 713907427 已完成两次全 TAR、两成员 SHA 与元数据双向校验。原释放脚本在 Python 路径接口处退出，源未动；修复后核双遍收据、精确来源 SHA、独占内核 lease 和原最终权重 SHA，释放 1,007,682,668 字节，`R/checkpoint_archival_20261104/repaired_diffusion_intermediates_v1/verified_source_release_v2.json`。旧失败状态保留。本机释放脚本 `src/release_verified_repaired_diffusion_intermediates_20261105_v2.py`。释放后 /data 11,517,550,592 字节，仍小于正式训练 11 GiB 底线。保留原始、参考、活动和最终权重。


## 2026-11-05 16:10 UTC｜修复参考已开始生成并接续视频、质量与 SuperMimic

- 修复扩散正式完成 10,000 更新，完整模型与全部 234 个 Adam 状态读回通过。最终 SHA `f73da1fdc88d033c5b5fd277aabd9d280c2aa1d4cbe94db09589feab8c491fd6`，保留 `S/runs/grab_omni29_mdm_floor_l2_guided_v2/step_000010000.pt`。已有两个 Pick 候选真实扩散生成，最新见证仍 0/2,000 合格，四类各 500 目标不变。
- v2 采样接口与 v3 显存分配故障均保留。v4 显存修复后验证任务投影三次 `ATb_norm=nan`，返回同一个 A 解，原拒收不足以判断扩散质量。修复零位移处的范数求导，原 JAX CPU 实测梯度从 NaN 变成 0，保持科学阈值。已安全停止 v4，复用前两个候选 raw 并按哈希验证。v5 PID 1864015，`S/new_references_v5/{status,progress,pilot_receipt}.json`，配置 SHA `5d2cfca7b6da4cc9f6b47fb36c63db05ddb6cf8f26244a8189f2486b7a81403a`。真实候选通过逐任务原生探针后才批量；每项最多 8 次先导尝试、1,000 总候选。
- 质量评估 v5 PID 1864016，`S/evaluation_controller_v5/status.json`，固定 HumanML 和几何评估 raw／重编码／后优化及全候选／准入，报告 FID、R@4、Diversity、Control L2、Skating Ratio。项目留出和论文 Full-Mix 分开。新原生 SuperMimic v5 PID 1864017，`S/supermimic_repaired_v5/status.json`，等新 2,000 清单及质量收据。复用 8912 环境、72,000 真实向量步，从头训练，完成后进行全库身体跟踪和四项各 100 首回合实际动态仿真，报告任务成功率及交互点误差。旧后续 PPO 仍取消。
- 视频源控制器 PID 1864018，`S/report_video_sources_v2/status.json`。参考每类最低准入 ID，复用原 DreamControl Viser 与原 29 身体/14 手指关节适配。下游每类取对应 100 回合评估的第 0 回合，保存 50 Hz 实测机器人 43 关节、根和物体位姿，在同一源状态回放，保留实际成功或失败。Windows 渲染 PID 44644，`src/repair_prior29_v2_20261105/render_report_videos_v2.cjs`，产物 `D:/DreamControl_v2_repaired_report_v1`。使用旧参考作无头录制预检成功，8 个新版视频待其实际上游完成。
- 旧 Pour 的 90 个中间点在双遍归档校验和源锁验证后已释放 550,945,092 字节。修复扩散的两个非活跃中间点已上传 GitHub，双遍读回待完成，最终权重始终保留。训练维持至少 11 GiB 与实际检查点预算、运行 8 GiB+356 MiB、项目 GPU 24 GiB/RSS 64 GiB。用户授权按需释放已有完整可恢复副本的非活跃文件，保护原始数据、参考、活动/最终模型和他人文件。旧结果保留。


## 2026-11-05 15:39 UTC｜新授权修复库 SuperMimic 与下游仿真接续

- 本轮用户授权顺序：修复扩散正式 10,000 更新及最终权重验证 → Pick、Pour、双手搬箱、推拉箱各 500 条真实准入参考 → 新参考轨迹质量评估完整出报告 → 使用新 2,000 库训练独立原生 SuperMimic → 全库身体跟踪与四任务各 100 回合冻结策略 PhysX 评估。新训练从头开始，沿用旧原生预算 8912 环境、72,000 真实向量步（600 早期 + 续训至 28,748 原生迭代）。后续技能 PPO 仍取消。
- 新接续控制器已在服务器 S 启动，PID 1857736，SHA 5441c45de456bc5e3ae6643bc3b17d49efbb22ba2b276e889d55619d7f839732，状态 S/supermimic_repaired_v1/status.json，当前 waiting_for_reference_quality_evaluation。原生、PhysX 既有环境及固定训练/评估逻辑复用，四份旧入口逐字节复制到新隔离目录，完成语法与预算导入预检。旧模型和任务保持原目录。新正式训练尚未开始。
- 质量报告、生成清单、参考哈希和每类计数一致后才转换原生库；再完整 CPU MotionLib/FK/时间检查和 8912 环境容量见证。正式启动需空闲至少 11 GiB 且实际检查点预算通过，运行底线 8 GiB+356 MiB，单作业 GPU12GiB/RSS32GiB，项目24/64。当前 /data 约 10.99 GB，低于未来正式训练启动门槛，若届时不足将等待。不可降低门槛或释放原始数据、参考、活动/最终模型或未经校验副本的文件。
- 最近实际扩散 9,710/10,000，生成控制器仍等待最终权重。原冷档日志 11 分块全部上传核验，SHA c491df125d039fe7357595176b90e576bcbc94a6f5321e8f32259344b4fdf1d4。第二 Pour 归档包双遍读回完成，最后 5 个源文件仍在。Pick 分词 v2 下载继续，沿用旧分词完整回放确认。新结果待评估与补提交。


## 2026-11-05 14:36 UTC｜正式训练已运行，发布中继等待恢复

- 15:17 UTC 更新：训练最新见证 8,430/10,000；第二个归档包第二遍读到 664,797,184/1,057,892,352 字节。剩余 138 个旧权重的 133 个分块全部上传并核验，重建归档 4,438,420,480 字节，SHA 2d12558f1a906cbb98b1e3b0f74503fb77ac2c792ef594ceba40fb69ae7973fd。旧 275 权重上传已齐。冷档日志包上传 PID 56180。Pick 分词 v1 下载超时，保留日志，v2 在原隔离方式下放宽下载时限，PID 1854750；评估控制器仍等待分词和生成收据。

- 14:40 UTC 更新：正式训练 6,460/10,000。服务器 SSH 已恢复，评估控制器 1836549 等四类准入清单与分词收据，固定分词后台 PID 1836550。重连监督器 34504 派生上传 PID 11780，已验证 48 个分块。第二个归档包第二遍已读 258,736,128/1,057,892,352 字节。

- 归档 712860202 的 85 个 Pour 中间点已通过两遍整包及全部成员 SHA，逐文件再做源身份、完整 SHA 和独占内核租约检查。安全释放 520,337,370 字节，Pour 最终权重 SHA 前后相同。服务器收据 `checkpoint_archival_20261104/github_pour_readback_20261105_v1/source_release_asset712860202_v3_receipt.json`。原 v2 汇总因 SSH 断开标记失败，保留为历史。
- 释放后实测 /data 可用 12,003,934,208 字节，超过正式启动门槛 11,983,126,528。训练控制器 1820275 已启动守护 1832242、正式 worker 1832245，CUDA 0。14:32 UTC 最近见证 5,940/10,000 更新，5,000 次检查点已读回验证全部 234 个 Adam 参数状态，SHA `d36f2cdb9335dc7aafbd6126105976980ea30f1dbba442bcf7c7fd5189c1497c`。生成控制器 1823251 等最终检查点，按四类真实单样本准入后各 500。
- 剩余归档 712850886（5 个 Pour 中间点及同包其他成员）以服务器后台 PID 1832620 两遍独立读回。第一遍通过，第二遍最近读到 181,403,648/1,057,892,352 字节。其源文件保留。收据 `checkpoint_archival_20261104/github_pour_readback_20261105_v1/remaining_asset712850886_v3_receipt.json`。不再运行全 90 份的旧释放脚本。
- 正式评估控制器和固定版本 Pick 分词脚本已在本机准备。服务器 SSH 在 14:32 后再次断续，部署尚待重连。分词只在服务器私有内存临时目录运行 spaCy 3.8.11 与 en_core_web_sm 3.8.0，回放比对所有既有分词后补齐 Pick。评估控制器核对 2,000 条准入清单和分词收据，再自动运行固定 HumanML 与几何评估。
- 权重上传已有 34 个 32 MiB 分块经 GitHub 验证。旧 PID 952 和第一次续传 PID 41188 已退出。重连监督器 PID 34504 已启动，按退避等原 ZeroTier SSH 通道恢复，依次部署评估支持并继续剩余分块。日志 `reports/publication_inventory_20261105/repair_reconnect_v1.log`，查该 PID 和派生上传进程，避免重复启动。完整可恢复权重仍 137/275。后续技能 PPO 不恢复。


## 2026-11-05 12:06 UTC｜修复版已通过运行见证，正式训练等待磁盘

- 12:15 UTC 已核验 Git 提交 374fe33c402e226d9740b39933082a032c44b39d。既有复现快照 1,323 文件，1,326 个新增或更新条目逐一核验，保留仓库其他 11,186 个条目。新修复版源码与后续实测结果仍待补提交。收据 reports/publication_inventory_20261105/git_publication_receipt.json。

- 四项修复已进入独立新目录 `repair_prior29_v2_20261105`。既有环境通过 7 项契约检查和 18 项后优化及评估协议检查。真实 GPU 小测完成 8 次优化更新，完整模型、Adam、随机状态逐项读回一致。小测检查点 503,841,302 字节。服务器 `reports/gpu_smoke.json` SHA `98febabc1833231a63e14ab06c6087c3715fa54d2f17bcb59b09dd90fc8280af`。
- 训练控制器 PID 1820275，`training_controller_v1/status.json` 当前 `waiting_actual_checkpoint_disk_budget`。正式启动要求 /data 至少 11,983,126,528 字节空闲，基于每检查点 576 MiB、3 份计划权重及 2 份额外预留计算。实测约 11.484 GB，正式更新为 0。控制器达标后只启动一次 10,000 更新。
- 生成控制器 PID 1823251，`new_references_v2/status.json` 为 `waiting_for_formal_checkpoint`。计划 SHA `68eb2f5195be1c44d384ea83edfa3b10948b491355c66aa5cd57ec4408996102`。先逐类做真实采样、保真后优化及物理探针验证。每类最多 8 次单候选质量尝试，执行异常立即停止并保留证据，累计上限 1,000 候选。通过后逐批累计各 500 条。未生成新参考。启动见证见 `reports/repair_prior29_v2_20261105/verified_server_startup_20261105_1202.json`。
- 评估缓存入口已部署，沿用固定 HumanML 评估器、586 条留出集和原几何指标，分开记录原始、重编码、后优化阶段及所有候选和准入集。已见证一条留出特征和一条既有分词的 512 维嵌入。新 Pick 实际提示缺固定版本分词，四任务文本检索待补。公开 Full-Mix 与项目结果分开。
- 旧 PPO 取消继续有效。归档读回 PID 1819280，`checkpoint_archival_20261104/github_pour_readback_20261105_v1/verification_receipt.json`。asset 712860202 第一遍全量通过，第二遍进行中，asset 712850886 两遍待完成。未释放源文件。两遍完整 TAR 与逐成员 SHA、最终源身份和独占内核 lease 全部通过才可释放精确的非活跃中间点。
- GitHub 结果补充包 asset 713243201 已验证完整，权重分块 PID 952 正常。上传出口为服务器，本机经 ZeroTier 中继。完整可恢复权重仍为 137/275。上传及归档核验的当前状态以发布收据为准。


## 2026-11-05 11:30 UTC 最新用户授权与取消（覆盖全部历史队列段）

- 用户已明确取消后续技能 PPO。1573490 于 11:22:53 UTC 停止且同身份子进程均退出，搬箱、推拉正式训练未启动；不得自动恢复队列。J/active_controller.json 指向 user_cancelled_after_pour_20261105_v1/receipt.json。原始控制器状态另存，完成模型、日志及失败结果保留。
- Pick PPO 3000 与 Pour PPO 5000 均完成，100 环境均 0/100。Pour final4999 SHA442f0453f6a4f8d66e0dee9d15f1466f722c4366aba739bb0f41e4e1416f87ff，100env report SHA04a1e05d527982ee3b44ff306754f79725504277ed3c56244018376990cf7e9a，接触/抬升/倾倒/复正均0。现有 SuperMimic 结果与旧2000参考全部保留。
- 用户新授权修复审计四点，重新训练条件扩散模型，再生成四类各500新G1参考。明确选择先10000次更新后评估。独立 src/repair_prior29_v2_20261105 实现中，旧模型/参考/绑定源码不动。新代码尚未通过完整服务器/GPU见证，不得称已正式训练。既有环境robot-diffusion@74af067a暖复用，不安装或改公共环境。
- 修复目标：一致的世界/canonical高度与正逆变换；固定公开空间引导日程和训练/生成提示覆盖；保留原生成动作的后优化及显式形变/物理硬验收；分阶段、分筛选口径的评估协议。Full-Mix数据不齐，不能冒称论文同口径；四固定语义提示不足标准32语义组检索，保留旧口径比较并显式报告适用性。审计 reports/diffusion_pipeline_audit_20261105/EXPERIMENT_AUDIT.md。训练更新10000不是论文公开值，1000是DDPM时间步。
- GitHub上传继续，仓库Pendragon08/conditional-human-motion-generation，release434987063仍草稿。远端8TAR及Pour新增17文件/11权重补充已验证，共137个完整权重已上传；138个本机冷档权重分块上传未完成，总库存275不能称全传完。以reports/publication_inventory_20261105收据为准。新四视频final_v4已完成，原Viser+29身体14指输入适配，非物体动力学成功。
- 用户要求尽量不耗本机VPN：GitHub上传/API从服务器发起，本机只走到10.147.17.32的ZeroTier SSH；底层是否走VPN不作保证。保留全部冷档，不大复制。上传网络故障从已核分块接续，不重传已完整内容。
- 新训练仍须GPU12GiB/RSS32GiB、合计24/64、正式至少11GiB及真实完整checkpoint预算、运行8GiB+356MiB。当前/data约11.48GB；C约0.8GB、D1.23GB/root2.32GB均紧张。Pour90份完成中间点550945092字节已在GitHub归档，但尚待服务器两遍全TAR/逐成员读回及最终F_WRLCK lease，不得直接释放。原数据、任务参考、current/final模型、他人资产、全部既有冷档保留。新SuperMimic/在线闭环/物体世界模型未获训练授权。

## 2026-11-05 08:25 UTC 最新接续

- 原生 SuperMimic、2000 条身体跟踪、四项各 100 回合的项目物体评估和五项生成指标均已完成。不得因下文历史状态重启或重评。物体任务均 0/100，身体跟踪通过不证明抓持或物体任务成功。
- 用户要求的原 DreamControl Pick 配置复测已完成，结果为 0/100 成功、0/100 右手接触、交互点误差 0.255011 m。使用固定公开版本的桌面、物体材质、质量扰动、重置及高度成功计数，保留 29 关节资产、参考初始化、配对摆位和首次回合窗口。不是论文所有条件的完整同口径复现。详见 reports/DREAMCONTROL_UPSTREAM_EVAL_ALIGNMENT_20261105.md 与 D:/DreamControl_v2_upstream_pick_20261105/pick_100env_aligned.json。其他三项缺少同定义公开任务，不以抬箱或抽屉任务冒充，不重复已完成项目评估。
- 已查明现有参考视频未应用 hand_closed，因此没有展示真实手指开合。后优化约束腕点和腕姿态，不认证掌指接触或抓持。固定 256 条配对诊断的 FID 为原特征 1.6704、同流程重编码 1.6543、后优化 4.4243，证据 D:/DreamControl_v2_upstream_pick_20261105/fid_roundtrip_256.json。不能将项目 586 条留出基准的五指标直接当论文 Full-Mix 同口径成绩，也不能称全部指标越高越好。原始参考、视频和结果全部保留。
- 只继续现有 PPO 队列 1573490。08:25 UTC 的 Pour 子进程 1613025、父进程 1613019 正常，实际 leaf 已达 4205/5000，之后自动 100 环境评估及搬箱、推拉各 5000/100 环境。源文件、门槛、场景和预算不改。只在完成、明确失败或需用户行动时通知。
- 本轮 /data 11,609,104,384 字节，运行底线足够，但低于下一个正式启动的至少 11 GiB。继续动态计剩余写入。C 约 1.115 GB 已低于 1 GiB+100 MiB 保留线，D 约 1.296 GB，服务器 root 约 2.329 GB，不能盲目追加复制。之前原生 26000/28000 两个中间点已完整冷存，索引 R/checkpoint_archival_20261104/native2000_final_run_intermediates_v1.json；不得因原路径归档标记重跑。现有大型剩余文件主要是原数据、依赖和当前/最终模型，保持不动。若后续空间阻塞，仅处理明确可再生缓存或完整已核副本的已完成非活跃产物，保留全部冷档及恢复说明。
- 在线闭环、接触世界模型和其他创新仍仅讨论，现有任务结束后由用户确定，不自动加实验。

## 2026-11-05：四项生成参考轨迹五指标评估完成

- 读取已验收的四项各 500 条最终参考及配对的原始扩散输出，完成 FID、R-Precision Top-1/2/3、Diversity、Control L2 和 Skating Ratio，未重新生成样本。使用原 DreamControl 公开 HumanML3D 网络和指标函数，权重采用原评估器的公开推理格式转换。复现侧 586 条 GRAB+Omni G1 留出特征作为共用 FID 对照，文本检索每条用原条件提示词及 31 条留出描述。与论文 Full-Mix 的评估集分别报告。
- 最终参考整体 FID 4.3705、R@1/2/3 为 20.25%/38.75%/46.05%、Diversity 1.3768、Control L2 0.02991 m、Skating Ratio 0%。配对的原始扩散输出整体分别为 1.4175、26.90%/40.60%/46.20%、1.3325、0.33411 m、7.32%。脚滑率为公开几何判据。生成 Control L2 和策略执行 Control Error 各自独立。
- 本机 reports/generated_reference_evaluation_20261105/summary.md 与 receipt.json 保留四类及整体两个阶段的结果、方法和哈希。服务器 generated_reference_evaluation_20261105/geometry_v1 与 embeddings_complete_v1 为已完成评估目录。

## 2026-11-05：冻结 SuperMimic 的四项物体任务评估

- 按用户要求，下游任务仿真分任务报告 Success Ratio 和 Control Error（米）。在现有 Isaac Sim 4.5 PhysX 场景各执行 100 个首次回合。从各 500 条训练参考对应的物体摆位中固定随机种子抽样，机器人按参考首帧初始化，物体在回合内由重力和接触动力学驱动。
- 冻结策略由原生环境读取最终 last.pt，保留原 MDP 观测、历史、未来目标及身体动作映射。PhysX 使用项目的 43 关节手部资产与原生身体驱动参数，双手开合由参考计划驱动。训练更新 0，评估前后最终权重 SHA256 一致。
- 正式结果均为 0/100 成功，平均交互点误差为 Pick 0.2537184 m、Pour 0.2480644 m、双手搬箱 0.2266409 m、推拉箱 0.1383181 m。每项 100 个回合均达到误差取样时刻。推拉分为首次推接触目标和拉回再接触目标，误差分别 0.0906571 m 与 0.1859792 m。误差取实际手腕到预设目标的 L2 距离，双手及多阶段先在回合内平均。物理接触覆盖独立记录，Pick 有 4 个回合在某时刻检测到右手接触，其他任务满足所需手接触的回合均为 0。
- 原生 2,000 条身体跟踪评估独立保留。本次为当前配对参考场景的冻结策略与手部计划组合评估，Pour 为无液体模型的瓶体代理。首次小测的旧 PhysX 根状态接口错误保留，v2 修正后小测和正式 400 回合通过执行与归档。GPU1 PPO 队列继续。
- 服务器归档 Q/comparisons/native2000_frozen_object_tasks_100env_v2.json。本机 reports/native2000_object_task_evaluation_20261105 中有收据、汇总及四份逐回合报告，收据 SHA256 65a0a80bec0a0f631060e910bf673f6e54af2f6c8f87c01f2a6e9d7dcd60d0b7。后续扩展方法继续保留这两个任务指标及其评估口径。

## 2026-11-05：SuperMimic 四类参考评估完成，原生归档已补齐

- 主训练达到 28,748 次原生迭代、72,000 次真实向量步（641,664,000 环境转移）。实际 Adam 更新为世界模型 287,480、策略 28,248、价值网络 564,960。最终 last.pt 保存在服务器，SHA256 为 571b207875610946f1065bf25bdf1dac78cb8c37f1691fc001324b1da64776ee。
- 29 身体关节跟踪评估逐条运行了训练参考库的 2,000 条动作。各类均为 0/500 触发失败，平均归一化跟踪回报依次为 Pick 0.9950922、Pour 0.9966018、双手搬箱 0.9949496、推拉箱 0.9937348。此评估范围是训练库跟踪覆盖，评估无域随机化，物体交互成功率和未见轨迹泛化须单独评估。
- 训练与评估均正常完成。自动汇总因 Q/comparisons 目录不存在而失败，已核对 2,000 条逐条记录、训练状态与最终权重后建立目录并补齐 canonical receipt。Q/comparisons/native_universal_four_tasks2000_v1.json SHA256 3ba4b8b0d9bb1153dbac238d48cc6d14e4c061e60535ed1b6b374135cc35d03c。原生评估 report.json SHA256 ef184738d810f05ea07439d309f3fe9290cb99f5e379c234c60f4e732e62d379。
- 原控制器原始 status.json 和 traceback 保留。N/active_native2000_controller.json 已指向 formal2000_sequence_v1/recovered_completion_status.json，D 的活动运行索引和 Q/progress 同步完成收据。原生不重启、不重做评估，原训练与评估源码均不修改。
- Pour PPO 和后两项既有接续队列继续。后续正式作业仍须满足启动空间门槛，按实测预算核对 /data，保留原始数据、最终模型、活跃作业文件和所有冷档。新闭环或操作世界模型只作为研究讨论，待既定任务结束后决定。


## 2026-11-05 00:20UTC：2000参考最终验收完成；原生600段已过并进入主训练，Pour PPO并行

- 四类最终各500、共2000参考已完整验收：B/mixed2000_final_v1/completion_receipt.json SHA90574d8e1c5469630d93b31dc0b33570bdf613ab2351f4421bdca1e9c4772126；混合清单SHA6d35b617780968b45ef9d05c2014df17708d6b45bb7158438a13ec6c49df36cf、实际原生加载报告SHA809ebc1f2af40c1cdec847661aff3e58a3b0bc3493faed12d06970bba69675d3。本机reports/four_task2000_final_20261105已下载并逐文件核SHA。真实MotionLib2000条/821000个50Hz状态、逐条未来参考与边界、6000 MuJoCo FK对照通过，最大位置差1.987e-6m。不是策略跟踪或物体成功率。
- 批量1482880与混合1491803已正常完成退出，不重启。新增三类从1534个新候选及6小样接纳504/504/500，恰选各500，多余和拒绝保留；已有Pick500及CSV复用，没有重建。Q/sequence版本8、progress、B/plan/active_pipeline及N/gate已对真实2000收据同步，历史在Q/before_reference2000_handoff_20261105保留。
- 原生1571248已通过实际新2000库8912环境65步容量（579280转移）、完整资源和检查点预算，自动启动GPU0。early600_v1真实完成600迭代、1565向量步/13947280转移，WM6000/策略100/critic2000，完整last.pt模型/优化器序列化读回通过，218547681字节，SHA86f4c61f9289abbb0201c09b0a75fbdc93a99352adabed22573a9730e22ee1ea。旧early Python1678437正常结束，主main_resume_v1 Python1680552已启动；实际读leaf/CSV/控制器，勿重做600或旧500验证、勿修改绑定源。恢复为模型和优化器，非replay/RNG精确断点；含两次65步热身的总72000向量步与最终28748原生迭代预算不变，之后全部2000身体跟踪评估。29身体输出不认证手物体任务。
- GPU1 Pour PPO1613025/组1613019已自动正式运行，8912环境/独立500库/5000更新，历史快照1864更新；实时看B/task_ppo_20261104/pour/ppo_8912_5000_v1/status.json、metrics和1573490队列。Pour完成及100环境评估后自动双手搬箱、推拉箱，各独立5000/100环境，原生继续GPU0。Pick已完成3000及0/100归档，不恢复旧训练、不把预算完成称任务成功。
- 原生一度因约0.27GB检查点预留缺口等待；本轮将39份已完成日志和94张历史检查PNG完整冷存本机training_space_archive_20261105/completed_logs_and_review_png/v1/payload.tar，完整352251304原文件字节，tar SHA c491df125d039fe7357595176b90e576bcbc94a6f5321e8f32259344b4fdf1d4。两遍逐成员SHA/完整tar SHA、服务器最终源身份与完整SHA、独占kernel F_WRLCK lease持有至释放源均通过，所有字节仍可恢复。索引R/checkpoint_archival_20261104/training_space_completed_logs_and_review_png_manifest.json及released/receipt；本机src/restore_completed_logs_and_review_20261105.cjs completed_logs_and_review_png v1 [exact_member]恢复且保留tar、拒绝覆盖不同内容。原始运动、当前/最终模型、任务状态、活跃日志和绑定源未动。
- 启动时/data约12.389GB通过实际12.308GB合并预算，完整checkpoint后资源仍动态监控，不将旧等待字段当当前不足。C约2.24GB/D1.39GB，保留所有冷档与既定底线。每作业GPU12GiB/RSS32GiB、合计24/64、正式11GiB及合并预算、运行8GiB+356MiB均不降低。在线闭环仍仅讨论稿，未批准实验，不加队列。

## 2026-11-04 14:47UTC：Pick500训练/评估已归档为0/100；空间冷归档完成；在线闭环仅为讨论候选

- 新Pick500 PPO真实完成3000更新，final model_2999.pt SHAed45d9d1f63b64ace4667964f02683280da1224b25a1ca88c2ca06487a653e31；同500库SHA01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572。Q/evaluations/pick500_final_3000_v1/report.json真实100环境只读评估0/100，报告SHA282d74ebb4093a571a6464129147cd040c3be2a2b69686e976f0ff2ed8ab3580，Q/completions/pick500_reference500_v1.json已由原控制归档。训练预算完成与任务成功严格分开：不能称Pick已成功，也不能单归因于参考数量或离线规划。原始结果/最终模型保留；后续需要独立诊断跟踪、接触和抓持，不能未经证据归因或重启旧预算。
- PPO队列1573490已自然转为waiting_task500_scene_admissions，当前等Pour达到500；原生1571248仍waiting_final2000_receipt，批量1482880/混合接续1491803健康。依最新用户授权继续三项独立PPO与原生共享策略，不因一次成功率0伪称训练异常、删除结果或跳过记录；运行源、初始化策略和训练预算均未改。
- 完整安全冷归档已释放约1.90GB原文件字节：六份已完成原生诊断权重（不是当前/正式模型）与未参与当前训练的11424条AMASS派生NPZ，以及已完成Pick500的60份中间点；最终2999、当前扩散10000、旧Pick最终2999、全部GRAB/Omni原始/当前训练数据、任务参考与他人文件未动。两份local tar分别在本机training_space_archive_20261104/native_and_inactive_amass/v2、completed_pick_intermediates/v2；逐成员两遍SHA与完整tar SHA通过，服务器逐文件source hash/inode/device/mtime再核，持有Linux F_WRLCK独占lease证明无其他打开FD至移出源，所有字节均可恢复。/data本轮约12.64GB，C约2.66GB/D约1.39GB，仍动态核两作业预算与剩余数据，不能放松门槛。
- 归档索引R/checkpoint_archival_20261104/training_space_{native_and_inactive_amass,completed_pick_intermediates}_manifest.json及released.jsonl/receipt，完整本机manifest映射原路径与SHA。恢复用本机src/restore_training_space_archive_20261104.cjs <group> v2 [exact_member]，不覆盖不同内容、保留本机完整tar。AMASS11424的文字/拆分/统计/来源清单及原作者PKL不变，motions/*.npz现为冷存；需要使用该数据时先恢复，不重新转换、重定向或重训。其数量/验收结论保持，只是存放位置改变。当前四技能生成仍用原3161训练的先验，与该冷库无依赖。
- 原生六份旧验证权重现有.archived.json标记；活跃原生控制器在初始化时已经测量并保存预算，不再读取它们，训练使用未来新early600/main权重。后续若另起新版本控制器需这些旧尺寸见证，应按索引恢复或读已核归档尺寸，不重跑旧验证。首个打包尝试因JS数字不能精确保存纳秒mtime而在移出任何源前安全停止，旧inventory/小型不完整tar/failed_preserved.json保留；v2以十进制字符串保存完整身份，未放宽任何校验。
- 最新用户要求：完成现有所有任务之后，再考虑在线闭环扩散规划，先给具体创新方案、再确定计划。本轮仅文献核对与候选技术讨论，reports/ONLINE_CLOSED_LOOP_PROPOSAL_20261104.md；未批准实验计划，禁止自动加入训练队列、修改当前先验或提前启动创新实验。当前四项、原生与PPO任务按既有队列继续。

## 2026-11-04 14:06UTC：用户要求已落实为双GPU自动训练接续；三项新PPO真实小测通过

- 用户最新明确：四类各500、共2000参考最终验收后立即进入原生SuperMimic开训；当前Pick PPO及100环境评估结束后，另外三项独立PPO接续，与SuperMimic并行。GPU0原生共享跟踪器，GPU1按Pour→双手搬箱→推拉箱训练独立策略，各5000更新沿用sequence原预算；Pick仍3000，不改或重启。Q/sequence版本7、progress和B/parallel_training_20261104/plan.json已更新；不恢复暂停技能、不做真机。
- 原生CPU等待控制器1571248已运行：N/active_native2000_controller.json、formal2000_sequence_v1_launch.json、formal2000_sequence_v1/status.json权威。新D/src/native_tracker2000_controller_v1.py SHA0fec81ab66d4c9a477047134a1378e4f186a393bf36c43b4541872b837471f84及worker/eval/完整原生源码均绑定，勿修改或重复启动。当前waiting_final2000_receipt，没有原生正式训练；真实最终2000收据→同库8912环境65步容量→合并磁盘/检查点/显存门槛→早期600原生迭代完整模型/优化器核验→同检查点续至28748→原生全部2000逐条身体跟踪评估、分四类归档。两次65步热身均计入72000向量真实步/641664000转移；不把native更新数当PPO轮数。原生仍为29身体输出，物体交互不由该跟踪评估认证。
- 三项新PPO场景/奖励/成功指标已在隔离B/src/stage3_supported_ppo_{core,env,train,manifest,queue}_v1.py实现，复用只读G1身体/手动作接口，未修改Pick绑定源。对象真实受重力/接触驱动，绝不注入计划物体运动。Pour检查抓取抬瓶、真实瓶体旋转>60度、复正放下释放（瓶体代理，无液体输运模型）；搬箱要求双手对物体的过滤接触、实际抬升/搬移/放下；推拉要求实际推动、释放、重接触、返回。瓶质量0.1kg、箱1kg为明确项目近似。Actor保持102实时输入、29身体+双手31动作，参考只奖励/初始化。
- CPU已核物体不动不会成功、推拉缺释放不会成功及三条有序正例。三项各2条既有已接纳参考加载后，均真实4环境2次PPO更新（各192环境转移），独立完整模型/Adam保存读回通过，物体重置误差0、私有参考ID变更时Actor差0；三项见证B/task_ppo_20261104/three_task_integration_receipt_v1.json，各integration_v1/status.json，不重复。这不是正式训练或任务成功率。
- PPO自动队列1573490已运行：B/task_ppo_20261104/active_controller.json、queue_v1_launch.json、queue_v1/status.json及queue_v1.log权威；当前waiting_existing_Pick_PPO_and100env_evaluation。先等待旧Pick正式训练/评估完成，再每项自己的500真实场景接纳（可早于其余任务）、恰500独立清单CPU加载→8912环境128步容量→与原生作业合并检查点预算→8912环境5000更新→100环境新连续对象位置只读评估归档。每任务独立目录/策略/最终权重，失败保留停止，不盲重启整队列或重做已通过小测。此队列与五个新任务源已绑定勿编辑。
- 保持每作业GPU12GiB/RSS32GiB、项目24/64、正式11GiB及两作业剩余检查点合并预算、运行8GiB+356MiB。原生私有/dev/shm缓存上限显式从1GiB改2GiB，真正启动检查共享内存至少4GiB，暖复用且不改HOME/不安装。当前/data约10.79GB，低于两作业正式启动所需约12.31GB，数据生成仍正常；控制器会等待，不能降低保护或拿当前有余量冒称可开训。后续在最终库到齐前按已授权完整冷归档非活跃产物/明确自有可再生缓存准备空间；原始数据、当前/最终模型、他人文件和无完整副本产物不得删除。C盘本轮实测约4.59GB空余，D约1.39GB；动态重测并保留1GiB+100MiB。

## 2026-11-04 12:50UTC：新增2000库原生入口，仅CPU准备，未启动GPU验证或训练

- 隔离D/src/native_tracker2000_worker_v1.py（SHA4e08aa55047f82ee85c4e2150f4d25864660e496d7926b0b7215699885f31c8d）仍整份执行固定原生train.py，无替代优化循环。实际最终2000收据/清单/加载报告哈希在导入Torch前检查；正式阶段另需真实2000库8912容量报告与gate。当前没有最终2000收据，CPU门禁正确拒绝提前运行，未导入Torch、未占GPU、未改任何活动源/任务。
- 入口准备采用早期600原生迭代保存完整模型/优化器并读回，随后同检查点恢复至28748迭代。两次各65真实回放热身计入后，仍恰72000向量步/641664000转移；相较单次28774方案减少26策略更新尝试，不冒称更新次数相同。WM-only500计入，名义WM287480/策略尝试28248/critic564960；实际Adam步骤及原生因梯度非有限跳过的策略更新分别记录。恢复不含replay/RNG。这里只是入口设计，600阶段尚未实际执行。
- 证据N/formal2000_worker_preparation_v1/status.json和cpu_gate_audit.json，旧单次预算与CPU前稿formal2000_preparation_v2保留。尚须完成真正2000数据、容量、正式保护/剩余检查点磁盘核算与启动、全2000分任务只读跟踪评估；本入口并未自动启动或把gate改true。每作业12GiB/RSS32GiB和合计24/64、11GiB正式磁盘门槛不变。当前批量和Pick PPO继续正常；CPU准备不单独通知。

## 2026-11-04 11:21UTC：释放旧失败SuperMimic-v1的清理挂起进程

- 发现旧自写v1的Python1444248在已记录nonfinite policy gradient、包装退出后仍卡在Isaac清理，非正在训练。核失败状态/日志哈希、进程启动标识及独占进程组后，仅向该PID发SIGTERM，正常释放；已再次核NVIDIA列表中该PID消失。未删除任何文件、未重启训练、未改失败状态或绑定源码。证据D/parallel_pick500_20261104/retired_v1_cleanup_20261104T1119Z/evidence_before.json及receipt.json。
- Pick PPO1436632、保护1442525、批量1482880、混合接续1491803、日志归档1484965的启动标识保持不变且继续健康。仍等四类各500与2000混合加载真实收据后才启动原生通用SuperMimic；不恢复旧自写v1/v2。此为旧失败进程资源收尾，不是新的训练故障或数据完成通知。

## 2026-11-04 08:05UTC：批量健康；2000混合库的自动CPU验收接续已接好

- 2026-11-04 09:13UTC补充：N/formal2000_preparation_v1/budget_source_audit.json已按固定原生train.py源码静态核算，仅CPU准备、未开训。单次28774原生迭代、UTD0.4、初始65真实向量步=72000真实向量步/641664000转移；WM-only500原生迭代计入，名义WM287740、策略尝试28274、critic565480。原生代码对非有限策略梯度会跳过opt_pi.step，正式必须另读真实Adam步数，不能把原生迭代数当成功更新数。重建回放/RNG不精确恢复，每次原生恢复新增真实热身必须计入预算；示例一次偶数迭代边界恢复并再热身65步，最终原生迭代改28748才仍为72000真实步。buffer128/pi_batch2048区别于默认1000/4096已列。正式仍待2000真实收据、新2000库8912容量、早期可恢复完整优化器快照、实际检查点预算/保护和全2000只读分任务跟踪评估；不重做旧500容量，不只改ready，不将原生29身体跟踪称手物体任务成功。
- 不修改运行中的五个批量绑定源。1482880继续滚动生成/审计/拟合/原生探针，当前历史快照Pour66/搬箱34/推拉2（102/1500），推拉首32批正在后优化；实际数量只读W/controller_status.json。前三个完整cohort均32/32接纳，不能把这些局部通过率外推为全部任务完成。Pick PPO1436632继续，历史快照1139/3000；健康计数不单独通知。
- 新隔离stage3_supported_mixed2000_prepare.py与stage3_supported_native_library_check.py已用四类各2条真实既有样本验证转换契约。8条源20Hz→原生MotionLib实际50Hz加载3284状态、每条未来10帧/终点保持/时间边界通过；24个原生FK与独立MuJoCo对照最大位置差8.41e-7m、旋转矩阵差4.25e-6。Pick既有CSV只读软链接复用，新增三类另写CSV；双手、物体计划、文字与任务原元数据保留在原NPZ及清单，未改旧500库、未新增训练。证据B/mixed_contract_preview8_v1/native_loader_check_v1/report.json，SHA dc2a7ebc022cbe68c4124a00b6c19cf2390b646b3f76b79504f9030f182da7de。此8条接口验证不是2000最终验收。
- CPU接续控制1491803已启动，B/mixed2000_completion_v1_launch.json、mixed2000_completion_v1/status.json及B/mixed2000_completion_v1.log权威；当前waiting_all1500_scene_admissions，不是第二采样/求解器，不占GPU，不重复启动。它等待W/completion.json真实完成，再独立重读四类各500来源/噪声审计证明/状态/原生证明/时间/关节/全局去重，构建B/mixed2000_final_v1；之后用TextOp原生环境32a40546在CPU真实加载全部2000，逐条CSV独立复算、20→50Hz插值、未来目标/跨clip边界和6000个MuJoCo FK对照。成功才写completion_receipt.json status=four_task2000_reference_dataset_completed；当前gate保持false，接续不启动训练。失败留完整前缀新版本安全续，不盲重做。
- 接续源绑定：preparer SHA1ceab746be1863e4b42c76995be74057972d0b97e16c28993f3fb1a5ed0960ef；原生加载核验SHA81149da96a4acd406987d79ef8d38ec6cc0050e8dec45e5d293ad68d10532c2e；接续控制SHA2a75f1c02906193f1d8144257f35ef8df2ae17cc8770ea83a6be3d29418ad90d。不得修改活跃绑定源。最终2000真实收据后通知用户，落实原生通用跟踪正式预算/保护/评估/空间，才开始共享权重正式训练；不把身体跟踪当物体交互成功。
- 为覆盖剩余数据和PPO检查点，已将停止的旧自写SM-style v2的early96/failure3424两份完整权重无损gzip冷存到/home/ouyanghao/dreamcontrol_v2_cold_archive_20261104/retired_sm_style_v2_v3，逐文件源SHA、独立完整解压SHA、归档SHA和源inode/size/mtime复核通过；最终持有Linux内核F_WRLCK文件lease证明无其他打开FD后才移出原源。完整原始字节1143614108保留，释放/data约1.14GB，实际/data约11.37GB、root约2.548GB，均动态重核；root保留2GiB+100MiB。索引R/checkpoint_archival_20261104/retired_sm_v2_manifest.json，源路径附.pt.archived.json；恢复用B/src/restore_retired_sm_checkpoint_20261104.py --name对应文件，拒绝覆盖并保留压缩档。不删冷档、不据标记重跑旧失败/验证。PPO当前与最终模型、扩散最终、原生验证权重、原始数据和他人文件均未改。早期两次/proc扫描因不可读FD表停止且未移出任何源，空目录保留并记录原因；最后改为内核独占lease校验，不放宽完整副本与无打开FD要求。

## 2026-11-04已修正推拉小样并通过原生验收，1500补齐批量正在运行

- 两条原推拉候选已修正且保持原目标/时序/全部验收门槛。102000采用push_postopt_v5，腕最大8.384mm、脚sole最高4.714mm；102001经v7更严格CG/LM精度与迭代、腕/支撑权重，腕最大9.296mm/旋转13.953mrad、脚sole2.096mm。B/push_final_repair_cpu_audit.json独立重读raw/实际初值/29状态/FK/限位/手计划/时钟通过。v5/v6/v7各尝试与失败结果保留，已通过者不重做，v7结束1477092不重启。
- Pour2/搬箱2经native_pilot_v1（1477733），推拉2经native_push_pilot_v1（1478894）在各自瓶/接收容器或箱体场景180帧真实PhysX几何/接触探针全部通过；六条逐文件/源/原生证明绑定复读在B/bulk_v1/pilot_admission_audit.json。机器人与物体均按参考/计划姿态注入，1次0.005秒接触探针/20Hz源帧，不是PD或真实抓取/倾倒/推拉成功。未重做人到机器人重定向，未过滤源GRAB/Omni。
- B/bulk_v1滚动批量控制器1482880（source SHA61d512d9c1ce87e09d1b0bc2d400c44447ad70041c699c43837c48428960deb0）已实际运行，controller_launch/status/log及source_binding.json是权威；每批32条，逐批DDPM→GPU噪声/条件/文件审计→复用JIT的两阶段29DoF拟合/必要时严格数值细化→独立状态/FK/目标审计→相应原生场景探针→原生证明/去重复读后入库。首个Pour32/32全部通过，当前源场景接纳快照Pour34/搬箱2/推拉2共38/1500（最终混合训练库尚未完成），当前实际以controller_status为准。
- 原32候选包装1479787与首两条拟合见证1480330均正常结束，控制器已采纳并复用候选11000000/11000001的已完成状态，只拟合余30，不重复启动或求解。新ID按Pour11000000+、搬箱12000000+、推拉13000000+分区，各任务总新候选上限1000；通过不够则新ID补采样，不重复轨迹凑数。目标各500包含已验收pilot2条，过量合格者保留、最终恰选500。
- 控制器源已绑定，不编辑运行源/不并行再起控制器/不重启旧目录。B/active_pipeline.json与plan.json指向bulk_v1/controller_status.json，Q/progress和N/all_four_reference_gate.json已同步；SuperMimic仍等待四类各500及2000混合真实加载，批量控制器不会启动训练，结束状态all1500_scene_references_admitted_final_mixed_loader_pending仍需完成独立最终清单/时间关节映射/2000CSV与原生MotionLib加载才通知最终完成。
- 暂估剩余全生成验收10–16小时（不含RL/SM训练），W/preprocessing_eta.json记录：实测候选中位15.441秒/条，暖后优化中位6.533秒/条（30条）；纯1500候选约6.434小时。70–90%后续通过率只是敏感性假设，含补采样/逐批验收和0.5–1小时最终混合加载，按新批实测更新，非承诺。共享资源/磁盘可能变化。
- 数据运行仍守8GiB+456MiB、单作业GPU12GiB/RSS32GiB、项目24/64；新批前空余不足底线+200MiB自动等待。1484965只对已完成cohort且关闭的日志做完整gzip双源SHA/解压SHA核验后冷存，索引completed_log_archive_manifest.jsonl、原路径.log.archived.json；全部日志字节保留，不触碰任何数据/模型/检查点，也不压缩失败/活跃日志。仅守数据构建预算，11GiB不是构建门槛；SM正式启动预算仍保持。Pick PPO1436632继续总3000，最新快照967，实际读leaf。

## 2026-11-04本轮最终阶段：4条小样运动学通过，原生适用未验；推拉2条需继续适配

- v4五条重拟合已正常完成（1472102/1472103，不重启），独立CPU原始状态/实际初值hash/投影/29关节FK/时钟/手计划/腕位置朝向复算通过。Pour100000/100001及搬箱101001运动学通过，与保留101000合并为Pour2/搬箱2；最大腕位置约5.65/7.31mm。两条推拉仍未过：102000最大腕32.7mm>30mm，102001腕42.3mm且脚未贴地。门槛不放宽，两条留证继续诊断可达/支撑与场景约束，不盲重算4条通过样本。
- B/staged_kinematic_pilots.json明确非最终参考库、正式训练禁止；包含4条已审状态与2失败、全部源/文件/目标/审计绑定。B/active_pipeline.json标记已正常结束、当前active_source_jobs=[]，不是仍有求解进程；Q/progress和N/all_four_reference_gate.json独立记录generated=6、kinematic=4、最终接纳=0。总最终库仍Pick500+其余0。
- 下一步优先给4条小样补各自瓶/接收容器与箱体的原生手物体场景适用检查，解决两条推拉的腕/支撑几何，再用已验证任务配置按通过率分批生成各500。原生姿态回放与实际接触动力学成功分开；不复制候选或先宣布1500批量。SuperMimic待四类2000最终清单，Pick PPO继续；旧v1/v2/v3/v4过程和所有结果均保留。下面v4运行中为历史快照。

## 2026-11-04当前执行v4：五条用已有近目标状态热启动，门槛保持

- B/pilot_postopt_v3已正常完成，5条虽限位通过，但腕/脚检查0/5通过，全部保留且不接纳。加强软限位后从中立态重新优化的结果没有满足任务目标，不能据限位通过称已修复。已通过的v2搬箱101000继续保留、不重新求解；六条原始DDPM候选和独立审计均不重做。
- 当前新v4保护1472102/实际Python以B/pilot_postopt_v4_guard.json和同名launch/log为准，src/stage3_supported_task_postopt_v4.py及guard_v4。只做原五条失败：仍用限位惩罚10000和全部原有严格门槛，改从v2已绑定raw状态作为优化初值，初值q仅在进入求解器前压入界内；最终状态仍须通过1mrad数值修正、3cm腕位置/0.20rad旋转/脚根地面规则，不把裁剪初值当合格结果。raw文件hash、实际初值hash分别保存，失败不覆盖。
- B/active_pipeline.json、plan.json与Q/progress已同步v4；先查真实状态，运行勿改源/不重启v3。完成需独立重读最终raw与状态/限位/FK/位置朝向/手时序，再进行新任务原生场景适用；未通过前不放大至1500批量。后三类合格库仍0，SuperMimic仍等2000，Pick PPO继续。下面v3运行中为旧快照。

## 2026-11-04当前接续：六条候选已生，保留一条运动学通过，五条仅加强限位重拟合

- B/pilot_postopt_v2已正常完成六条180帧全轨迹求解，不重启v2。独立CPU审计independent_state_audit.json确认：搬箱101000完整29状态/FK/时钟/腕旋转/手计划通过，最大腕位置误差5.345mm、旋转5.137mrad；其余5条原始限位修正需求1.316–13.381mrad超过原1mrad规则，拒绝属实，源候选/原始求解状态/日志保留。101000不重复拟合，不计原生最终接纳。
- 当前新v3保护1470858/实际Python以B/pilot_postopt_v3_guard.json为准，pilot_postopt_v3_launch.json/log/status。src/stage3_supported_task_postopt_v3.py只把软限位惩罚1000提高至10000，1mrad投影门槛、腕位置3cm/朝向0.20rad、脚根/地面等规则不变；只重拟合失败ID100000/100001/101001/102000/102001，复用已审6条DDPM，不重复生成或重做101000。先查实际完成前缀，运行不改源/不重启。
- B/active_pipeline.json、plan.json和Q/progress已更新当前真实v3阶段，下面1469298/1469299“运行中”是已结束v2的历史快照。六条候选不是1500合格条数，后三类最终接纳仍0；继续v3结果独立CPU重读及各任务原生场景适用检查，通过小批后再扩500。SuperMimic等待2000，Pick PPO继续。

## 2026-11-04后三类参考已实际试生成，六条候选完成独立审计，29DoF后优化运行中

- B=R/stage3_supported_references_20261104。Pour、双手搬箱、推拉箱各2条真实DDPM候选已生成（ID100000/100001、101000/101001、102000/102001），pilot_v2/candidate_audit.json逐文件/源文字/20Hz180帧/目标掩码/姿态计划/初始GPU噪声与特征哈希复算及6条跨任务去重通过。三类均20Hz/约9秒，复用固定10000步先验与原统计，无源数据重定向/逐条物理筛选。不是1500批量完成或六条最终合格参考；后三类最终计数仍0，Pick500不变。
- 新任务不是更换Pick名称：Pour有2秒抓取、3.5秒抬起、4.5–5.5秒世界X轴-75度倾倒、恢复直立和放置；搬箱左右腕共同接近、抬高0.14m、搬移0.10m和放下/释放；推拉箱包括前推0.10m、释放、重新接触并拉回同一项任务。当前是项目桌面小范围场景，物体期望路径只是计划、不伪装真实仿真。位置hint进DDPM，腕朝向/手二值保存为后优化与接口计划，不声称DDPM接收朝向条件。脚hint使用正确10/11踝点，旧Pick不修改。
- 当前后优化保护器1469298/实际Python1469299，B/pilot_postopt_v2_launch.json、pilot_postopt_v2_guard.json、pilot_postopt_v2/status.json和同名.log为权威。src/stage3_supported_task_postopt_v2.py SHA e3a708077e6e1ad48fd8ed9c586aa32221f93528b74197ea2579b142f818576d；29身体、180帧、两腕位置与朝向、双脚支撑、地面/桌面、软限位1000与1mrad数值投影规则，复用已验证JAX/PyRoki5%GPU0。首次JIT/求解运行中，不重复启动、不改绑定源或重做六条DDPM。
- 前置pilot_v1仅初始化因CUDA可见性设置与已有物理索引契约冲突失败，无候选；queue_v2按原契约取消重映射后生成六条。postopt_v1因时间扩180后smoothness仍119对而未开始求解；v2改179对，失败日志/源码/目录均保留，不是放宽筛选或重装环境。采样控制1466648及独立审计1467300已正常完成，不重启。B/active_pipeline.json和plan.json记录真实阶段，Q/progress已同步。
- 接续先完成六条后优化，独立重读29关节/FK/腕位置旋转/限位/脚根/哈希与任务时序，再实现各任务原生手物体场景适用检查。通过后按实际通过率/空间分批扩每类500，失败新ID/seed补齐；不得在未通过pilot前直接冒称1500条已批量入库。SuperMimic仍等待2000最终清单及混合加载；Pick PPO1436632持续正常不改不重启。数据底线8GiB+456MiB、单作业GPU12GiB/RSS32GiB、项目合计24/64维持，当前/data约10.34GB。

## 2026-11-04最新用户变更：四类参考齐备后再一起训练SuperMimic

- 最新用户明确“生成完了再一起做supermimic运动跟踪”。此要求覆盖下面“先用Pick500正式训练、随后扩库”的旧顺序。原生通用SuperMimic正式训练必须等待Pick、Pour、双手搬箱、推拉箱各500条最终合格参考，共2000条，混合输入统一验证后再启动。当前原生500条短测结果全部保留，不升级为正式训练，不重跑已有验证。
- 2026-11-04实际服务器核查：Pick500/500已最终验收；Pour0/500、双手搬箱0/500、推拉箱0/500（最终合格库数量），后三项任务条件/场景适配及生成参考仍未完成。其他旧人体重定向、作者已有源轨迹和native诊断模型不能计作新增任务的扩散生成参考。累计500/2000，剩余1500。当前没有新的原生SuperMimic正式训练作业。
- 权威前置条件N/all_four_reference_gate.json和wait_all_four_receipt.json，training_allowed=false；D/active_native_universal_run.json、adaptation_status、N/status.json与Q/sequence.json版本6、progress.json已同步并保留before_wait_all_four_20261104旧快照。下一步优先实现后三类真实任务条件与小批生成/后优化/适用验证，再各扩至500，独立源/目标/时钟/状态/文件证据及新版2000混合清单，不能只改Pick prompt/复制轨迹凑数。
- 29身体DoF、源参考20Hz/最长10s保持；原生Actor可以输入历史本体和未来参考，四类共享权重，双手/物体接口仍需明确实现与验收。Pick PPO保持独立正在运行，PID1436632/总3000更新，政策变更快照605；不等待剩余1500、不暂停不重启。旧SuperMimic-v1/v2失败和原生已完成验证均保留。
- 正式入口后续必须重读四类各500最终清单及其哈希、真实混合加载/关节顺序/时间网格、资源与检查点预算；四类齐备前不因单Pick500验证通过或空间满足而自动开训。其他论文技能继续暂缓，Nymeria暂停，不做Sim2Real。

## 2026-11-04原生通用跟踪器：完整主训练器、编译与8912批量优化已实测通过

- 用户选定一套通用参考条件跟踪策略；四类技能共享权重，输入本体历史和未来参考，不再要求SuperMimic采用PPO的102实时无参考输入。PPO仍分任务，当前Pick1436632保持运行，快照563/3000仅供定位，实际读leaf。原生29身体输出与双手/物体交互仍须分开，不把身体跟踪通过当抓取成功。
- N=D/native_universal_20261104。完整未改train.py SHA b94569605bd3ff96a49d4903bb791bb54bc4695a03f1914c8447c0ec7dd1e6bd已真实运行。smoke_author_v1通过64env/500参考/5迭代，Adam实际WM5/策略4/critic80，4864真实转移、970个Inductor生成kernel、218547617字节checkpoint读回一致。作者及fresh-agent环境kernel32a40546均通过，不重做。
- 顺序控制器1461436已正常完成，不重启：capacity_8912_v1真实8912env/64步/570368转移通过；benchmark_8912_v1真实8912env/2048批/H32/WM每轮10批/10迭代通过，Adam WM100/策略9/critic180、793168转移，970kernel、last.pt SHA f03be218339d63303d76f26c6888105344a8d67722669a2e086d251f59efdc4e、218547681字节。预热8迭代中位1.63827it/s是短测，不作正式完成时间或性能优势承诺。GPU进程8.89GB/RSS4.13GB低于限额。完整证据N/native_compiled_benchmark_receipt.json、各guard/leaf/日志，勿重复短测/容量/benchmark。
- 原生整份evaluate.py已对10迭代测试模型只读评估全部500轨迹：evaluation_entry500_v1_guard.json passed，零学习、源checkpoint不变，500/500跟踪终止、归一化回报0.15333。只是原生评估入口验证，短测模型未正式训练，不能把入口通过说成跟踪效果合格，也不是Pick交互评估。此入口不重跑。
- 原生实际恢复见证resume_8912_v1已通过并正常退出（1463190/1463191，不重启）：从benchmark原生last.pt恢复到12迭代，真实WM120/策略11/critic220及1408096累计转移（原793168+重建回放64步570368+5步44560）。新last.pt SHA72fd5222de6f0a282c29b6d34e016fbd9b5663c2669c29dbfdc18829d1a61560。上游checkpoint没有replay/RNG，明确是模型/优化器恢复、重建回放，不是精确断点。N/validation_complete_receipt.json绑定全部五类验证，D/active_native_universal_run.json指向已验证待正式启动，不重复验证。
- 正式原生通用策略尚未启动。后续须落实正式预算/保护/评估接续及空间检查；当前/data约10.36GB低于11GiB启动线，不降低门槛。按原生UTD0.4可用28774迭代+65步初始采样精确对齐72000真实向量步/641664000转移，500个WM-only原生迭代计入；此设置须单独写正式计划并核实，不把原生迭代等同PPO3000。buffer128/策略批2048为已实测容量配置，区别于上游默认1000/4096，明确记录。原生与PPO观察/目标/手物体不同，不冒称严格同条件算法对照。
- 原生正式库先用现有Pick500，后续Pour/双手搬箱/推拉箱各500形成独立新版混合库，不等待其余1500再开始。当前500/原始数据/旧失败/检查点全保留；如空间不足，仅可清理自有可再生缓存或完整已核冷归档的非活跃权重，不触碰PPO当前结果/最终模型/他人文件。

## 2026-11-04最新明确选择：SuperMimic使用一套原生通用参考跟踪策略

- 用户已回答澄清：选择“一个通用跟踪策略：输入各任务参考动作来驱动机器人，最接近原生SuperMimic”。此授权明确允许SuperMimic Actor输入本体历史与未来参考，覆盖旧102实时无参考Actor限制；PPO仍按Pick/Pour/双手搬箱/推拉箱独立训练，当前Pick PPO1436632继续。SuperMimic先用现成Pick500，后续其他三任务参考形成独立新版混合库，四类共享跟踪器，不等于无参考自主任务成功。
- 停止沿用旧自写WM/BPTT适配：v2在3424实际步/30514688真实转移、1343策略更新后再发散，单一梯度Inf已留证，完整故障恢复点resume_failure_3424.pt SHA8ae346f6670f3c0b7ef18c35797f0296635c7d6e9c216067ffb3507829ae2845，571807054字节已独立核。旧v1/v2/失败/检查点全保留，不重启旧优化器或旧评估1448750，不再修补旧适配循环。旧数值/评估控制器已退出；PPO控制器1442525仍负责PPO3000及其100环境评估。
- 新目录D/native_universal_20261104，handoff_receipt.json真实保留故障+PPO不中断；preparation_receipt.json记录500条源20Hz/60000状态CSV输入、WXYZ→xyzw明确转换、29关节顺序与读回。source/supermimic/train.py整份字节与原生zita-ch固定0525a23一致，SHAb94569605bd3ff96a49d4903bb791bb54bc4695a03f1914c8447c0ec7dd1e6bd。world-model/策略/critic/回放/优化主循环不再自写。仅复用TextOp已验证IsaacLab3资产兼容env.py/robot.py，MotionLib源FPS默认30→20以正确读本项目20Hz参考，其余数据时钟不擅改。原始参考与原始TextOp项目均只读。
- 原生环境沿用TextOp Python3.12/Torch2.11+cu126/Triton3.6、Newton/MuJoCo-Warp，不重装。新canonical环境spec32a40546；GPU0、每作业12GiB/RSS32GiB、项目合计24/64，数据8GiB+356MiB底线保持。作者8x8 seeded CUDA已通过，独立照文档kernel_freshdoc_v1与完整500库/原生编译批量2048、32步BPTT、BF16/TF32的5迭代优化见证须真实通过，不仅凭import或源码含torch.compile称已接通。入口D/src/native_tracker500_worker.py和native_tracker500_guard.py，新输出独占，失败留证新版本修复，勿盲重启同目录。
- 原生基本跟踪器学习29身体动作，默认含10历史/10未来参考；它尚无双手二值策略输出和目标物体场景。源hand_closed/物体位置仍留原始参考元数据，手部/物体交互单独接入验收，不把原生body29短测称Pick抓取成功或完整31动作正式任务。当前仅原生接入验证，正式库容量/预算/原生保存恢复与独立跟踪评估仍需补齐；不照旧自主Actor评估器评测新网络。
- 2026-11-04实时GitHub API：当前TextOp来源zita-ch/supermimic_vibed 808stars/85forks，Motphys/Uni-SuperMimic240stars/19forks。前者更多，两者均为非官方复现，不用star作性能/正确性证据。用户仅要求查Uni，无授权改用Uni替代当前来源。

## 2026-11-04用户核查：Pick扩散确实有空间条件，但原始空间误差较大

- 已只读重读最终500参考的原始DDPM候选/target_points/target_mask与来源hash，全部有文本和空间目标；22点索引0=骨盆每120帧、21=右腕每120帧、4/5=左右膝各24帧（每5帧一次），不是脚部空间hint。右腕2秒目标=物体中心+[0,0,.04]、4秒=[0,0,.20]并保持；物体XY+-0.06m。hint/hint_mask确实进入CMDM控制分支，GaussianDiffusion实际在mean去噪步骤调用self.guide做空间MSE梯度，不只是后优化目标。腕朝向、脚部支撑、桌面/地面碰撞另由29DoF后优化处理。
- 全500原始扩散右腕2秒目标误差中位0.418744m、4秒0.330961m；同最终参考经后优化后分别中位0.001535m/0.002463m（2秒最大0.004510m）。不能把视频毫米级精度说成扩散直接达到；空间输入存在不等于原始生成已准确满足目标。不据该原始误差否定或重建已验收后优化参考，不修改/中断当前两套RL。服务器O/diffusion_spatial_condition_audit_v2_20261104.json，本机reports/pick500_parallel_20261104同名报告，核查无GPU采样、无额外物理筛选。

## 2026-11-04最新：v2通过600步诊断并恢复正式训练

- 数值修复v2已真实完成8912环境600步/500WM热身/214策略更新及完整序列化；随后隔离新正式v2已运行1080实际向量步（906WM/406BPTT），超过旧480步失败前缀。D/runs/pick500_8912_72000_parallel_v2/Python1449777、包装1449772，控制器1448225。恢复见证W/numerical_repair_v2/recovery_verified_receipt.json，正式96步完整恢复点SHA b346550196342fb240924e146f8e78b5eecf54a2fd6217cf0b14777d7abe04c0已读回、逐字节SHA复核。属于观察到的稳定恢复，不是全部训练完成或任务成功；不冒称v1断点恢复。
- PPO1436632/GPU1正常快照352/3000，SuperMimic v2在GPU0继续，500参考/两作业预算/资源保护/100env评估不变。D/adaptation_status与Q/progress已同步v2运行。旧v1的failed状态、旧评估退出及旧修复等待段为历史，不能据此重启任何已退出作业；今后读新v2 leaf及1448225保护状态。

## 2026-11-04接续优先：SuperMimic数值诊断，PPO不受影响

- v1正式SuperMimic已在最后确认480真实步后出现nonfinite policy gradient并留证停止；最后166策略更新不代表成功或已完成。v1无首个持久优化器checkpoint，失败计算单列，不能声称恢复。PPO1436632/GPU1继续，总3000，控制器1442525继续保护和最终100环境评估。
- 仅在新v2源/目录做数值修复假设：全局梯度范数用FP64，普通梯度裁剪等价，不掩盖NaN或Inf。CPU验证通过；实负载600步/8912环境/500WM热身诊断正在GPU0，控制器1448225，W/numerical_repair_v2/status.json（W=D/parallel_pick500_20261104）。须真实超过旧480前缀/166策略更新并完整序列化通过才允许新formal_v2，失败保持结果不循环重试。不能把CPU小测或尚在跑的600步诊断说成数值故障已解决。
- 条件新正式run D/runs/pick500_8912_72000_parallel_v2总72000实际步，独立新作业而非恢复v1。正式96步完整早期检查点，随后1500宏轮/最终/异常紧急保存，预算仍按4完整＋60小策略点及PPO剩余预算，启动11GiB/运行底线不变。对应100环境评估等待PID1448750；v1评估已退出，不重启或重复。计划和状态上面的接续段覆盖下文v1健康并行的历史快照。

## 2026-11-04最新更正：同Pick500库的PPO与SuperMimic并行

- 已见证两正式优化器同时实际更新：PPO184/3000；SuperMimic真实168向量步、541WM/41BPTT，500WM热身已完成且计入64真实采样步。收据D/parallel_pick500_20261104/parallel_training_started_receipt.json。两者原生更新数不可直接相等比较，统一641664000真实交互预算。冷归档需要某成员时用本机src/restore_parallel_archive_member_20261104.cjs传manifest的准确member恢复，保留本地tar并校验远端内容/空间且拒绝覆盖。

- 用户更正“切换优先级”：两张GPU并行比较。PPO原PID1436632/GPU1/run pick500_8912_3000_v1始终继续，无暂停或重启。SuperMimic使用GPU0、同500库SHA01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572、同8912环境/场景/奖励/重置/102实时Actor/31动作；明确为已授权的Isaac4.5 PhysX世界模型+BPTT改编，不是原生CLI或PPO改名。新run为R/stage3_supermimic29_20261103/runs/pick500_8912_72000_parallel_v1，真实72000向量步/641664000转移对齐PPO3000×24，64步热身计入，原生优化更新单列。
- 并行控制器1442525：D/parallel_pick500_20261104/controller_v1/status.json（D=R/stage3_supermimic29_20261103），active指针O/active_completion_controller.json及D/active_pick500_run.json。旧1427653只交出保护控制，PPO进程/进度及绑定源码均保留，不能重启旧控制器或串行再起一份SuperMimic。最新formal已实际启动，模型热身/策略更新以SM run/status.json和metrics为准；不要把初始化/容量测试冒称策略正式更新。
- 保留每作业GPU12GiB/RSS32GiB；并行两作业合计24GiB/64GiB，按实际独立进程组统计，仅新比较作业先让出资源，不干预他人。11GiB正式启动、8GiB+356MiB运行底线均保留，还按实测571806990字节完整SuperMimic检查点与PPO剩余检查点合并计预算。四任务各500/独立PPO范围不变，不恢复旧八项；SuperMimic是同Pick库额外对照，不算新任务。
- 已真实完成新的500库4env96步（16WM/12BPTT/完整模型优化器回放序列化读回）和8912env96步（相同16/12、855552转移、456864768字节回放、571806990字节完整检查点内存序列化读回）；对象重置误差0、Actor私有参考改变差0。证据D/parallel_pick500_20261104/integration500_v1和capacity8912_v1。不是任务成功率；不用再次运行这些入口。
- 旧已完成但清理挂起的本项目容量进程1327291已依据smoke_completed/32步及命令绑定释放，不触碰其他作业。为双训练检查点腾空间，4542个旧本项目Omni派生npz及1个新接线测试检查点共507494082字节完整流式冷归档到本机parallel_archive_20261104/inactive_previews_and_diagnostic.tar，逐成员两次SHA及服务器inode/size/内容/无打开FD再次核对后才移出服务器；源数据/当前500库/训练数据/当前与最终模型保留。索引本机parallel_archive_20261104/manifest.json、R/checkpoint_archival_20261104/parallel_inactive_manifest.json；原目录有冷归档标记，完整字节保留，勿重复生成旧预览或清理本机归档。
- PPO3000后原生100环境评估/归档由并行控制器接续；SuperMimic完成72000实际步后独立100环境评估由PID1444616的D/parallel_pick500_20261104/eval_followup_status.json接续，完成收据Q/comparisons/pick500_supermimic_parallel_v1.json。各自成功率与完成训练预算分别报告。仅仿真，无真机，Nymeria暂停。正常训练/常规计数保持安静。

## 2026-11-04：Pick最终500条已验收，8912环境3000更新PPO已开始

- 从507条原生接纳中按candidate ID确定性选恰500，manifest SHA01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572，O/final_library_v1/reference_bank.json。全628原始候选GPU噪声复算/源目标/时间/去重审计完毕，选中500逐文件状态/原生证明/FK/限位/目标重读通过；20Hz、每条120状态约6秒、29身体关节加双手二值。默认ReferenceBank500真实加载及8912环境128控制步通过，物体重置误差0、Actor私有参考变更差0；final_library_audit.json status passed。不是PD跟踪或抓取成功。无需再生成补样、重复最终库或容量。
- 已真正启动新Pick PPO：1436632，S/runs/pick500_8912_3000_v1，8912环境/总3000更新，当前快照27更新，实际以status/metrics为准。新pointer O/active_pick500_run.json，保护/评估接续控制器仍1427653、completion_controller_v1；六个训练/环境/评估/最终化入口源已绑定，不修改运行源码或重复启动。实读第一正式checkpoint model_0.pt SHA1820678bf4243f5ec87084d8ef98df7cbf3c3e48a67622e293a06533e2514d1d，iter0=1更新、真实优化器状态、有限权重、500库哈希/102实时Actor/31动作/29DoF均通过，完成收据usable500_training_started_receipt.json。训练与最终100环境评估尚未完成。
- 为保持11GiB启动预算，58个本项目非活跃检查点已完整复制到本机并两遍SHA校验、服务器再核内容/inode/device及无打开FD后冷归档，释放2004750460字节。包含两个扩散中间点、六个SuperMimic接线/恢复诊断点、旧Pick v3的50个中间点；原始GRAB/Omni/人体源、当前扩散最终10000、旧Pick最终2999、当前新训练模型/日志/报告/参考均留服务器，两个受保护最终模型哈希前后未变。全部检查点内容仍完整保留，未丢弃结果。
- 归档索引：R/checkpoint_archival_20261104/manifest.json；本机checkpoint_archive_20261104/manifest.json列每个旧路径→C/D本机文件/SHA/尺寸，C位于本项目checkpoint_archive_20261104，D位于D:/DreamControl_v2_checkpoint_archive_20261104。源路径附.pt.archived.json。需要旧中间/诊断权重时先用src/restore_archived_checkpoint_20261104.js按manifest相对路径校验恢复，不因文件归档而重新跑已通过见证，也不能把归档标记当权重。正式最终模型不需恢复。
- 启动前实际/data约12.08GB，满足11GiB，不是降低预算；本任务GPU约8GB/RSS约16GB低于12GiB/32GiB。当前正式状态看leaf和controller.resources，controller顶层旧free_bytes/shortfall_bytes仍是此前等待空间的历史值，勿误判未开训；library_progress里的formal_training_started=false也是原生阶段旧快照，以新active_pointer/leaf/实测checkpoint为准，不改旧证明装完成。后续控制器核model_2999后真实100环境评估归档，预算完成与高度型成功率分别报告。其余四任务顺序与暂缓任务保持，SuperMimic额外同库对照不替代四项PPO。


## 2026-11-04：原500候选审计通过并顺序补样；最终库到PPO评估接续已排入

- 原496候选已全部生成、独立逐文件/时间/目标绑定/GPU初始噪声复算496/496通过；与pilot4合并核ID0..499、噪声/特征全局去重500/500通过。原采样1402368与优化1405357已正常结束，不因PID消失而重启。顺序队列1424784已自动启动补样128（ID500..627），采样PID1425966、后优化PID1425968，目录candidate_replacement128_v1与postopt_replacement128_v1；registry已更新真实PID，native控制器1411803继续串行接纳。当前414/500，只看实际library_progress；补样不是最终接纳。
- 新stage3_pick500_finalize_library.py已准备CPU最终500构建/源噪声审计证明/每条状态与目标/FK/限位/原生证明及去重全量复算，恰选500个已接纳ID，保留多余与失败结果；目前未运行，须真正500及候选审计齐备才执行。输出final_library_v1/reference_bank.json和cpu_audit.json；CPU通过仍需默认ReferenceBank500和8912环境128步实测，不能仅改ready。
- 新stage3_pick500_evaluate.py真实4环境32步读取2更新接线检查点测试通过，实际新物体XY随机位置、102实时Actor无参考、31动作、29DoF、策略/检查点不修改、零训练更新；仅评估入口验证，不是新正式策略成功率。默认正式100环境评估用新seed20261105、当前桌面+-0.06m连续新物体位置，参考仅初始化/无学习时奖励，接触目标从实际对象计算；主要成功指标仍height>0.95m且过接触时间，同时记录持续抬起时长和腕误差，非手指抓握认证。证据O/evaluation_entry4_v1/report.json和guard，勿重复。
- 最终接续控制器PID1427653，O/completion_controller_v1、同名launch/.log及active_completion_controller.json，源码stage3_pick500_completion_controller.py SHA c7a3b1c758528a323a952090a7471461e8fe63b1f7570a46b22582be9478d9ef。实际状态waiting_reference500_and_source_audits；它等待原生500+全628源审计，真实CPU最终库→默认500库8912env128步→结合实测生成final_library_audit.json→等待11GiB/实测检查点预算→另目录Pick3000 PPO→真实100env评估归档。不是已开始正式训练。六个入口/env/core源已绑定，不并发改源码或重复控制器；阶段失败留证，不重启旧队列。原生/补样/最终控制器均各司其职，不互相替代。
- 正式运行若满足预算将用S/runs/pick500_8912_3000_v1和O/active_pick500_run.json，保留旧2参考pointer/模型。守本项目合计GPU12GiB/RSS32GiB、启动11GiB、运行8GiB+356MiB；不足启动预算会明确等待，绝不降门槛或删无备份检查点。实测500库/容量/正式训练/最终评估仍待完成，不把接续代码当结果。


## 2026-11-04最新授权：近期改为四项数据支持任务，每项500，Pick优先

- Pick顺序补采样队列已排入：PID1424784，O/replacement_queue128_v1及同名launch/.log，源码stage3_pick500_replacement_queue.py。当前waiting_original_sources；原496采样与488优化完成且实际退出后，独立核496文件/目标/时间/GPU噪声并与pilot4全500去重，再启动ID500..627共128个新seed候选和通用v3后优化，不并发重复旧模型。registry已原子登记两个新目录和完成路径，原生控制器将正常等待并接纳；新真实PID由队列启动后登记。此批按当前缺口补样，不是已完成500参考；如仍不足需新ID继续，保留失败前缀。数据8GiB+456MiB、合计GPU12GiB/RSS32GiB保护不降低，只能停止该队列新启动组。


- 当前执行任务改为Pick、Pour、双手搬箱、推拉箱，各500条最终接纳参考，合计2000；推拉箱作为完整一项任务500条、一套PPO，不拆成各500的两项。G1保持29DoF，20Hz/最长10秒；每任务独立PPO和100环境物理仿真评估。先完成Pick500并训练其总3000更新，不等待其他1500。旧八任务/4000为历史扩大范围：Punch、Step&Punch、DeepSquat、DeepSquat&Pick、DrawerOpen、VerticalWipe按用户要求暂缓，保留原计划和所有结果。
- Q/sequence.json、progress.json与O/plan.json已在旧控制器退出且取得队列锁后更新，旧版本以before_supported_four_20261104命名备份。实际Pick接纳以O/library_progress.json为准，当前快照351/500；未重复或中断原采样/求解/原生批次。SuperMimic已授权同库对照单列，不计第五个技能/新增500，也不替代四项PPO。
- Pour、双手搬箱、推拉箱仍需各自29DoF场景、双手/腕姿态条件、物体接触与真实成功标准、小批生成/仿真/PPO验证，之后再各扩到500。不得只改Pick名称或prompt就冒充搬箱推拉；使用当前固定扩散先验及对应数据动作支持，不把论文未实现任务标ready。


## 2026-11-04：当前GRAB＋Omni先验不能直接假定覆盖论文八技能

- 用户要求看清目标是否到位，目标对照视频v6已完成、独立核360时刻目标/原生位置/NumPy FK误差及视频哈希，真实中英数字画面已审。交付reports/pick29_targets_final_20261104；这是后优化参考手腕到位，物体固定，不是抓取成功，正常退出不重复。失败v2/v3/v4及v5字体问题保留记录。

- 根据用户质疑，核真实训练清单5fc1a0...及随机使用的全部candidate_captions：训练2091=GRAB903+Omni1188。pick/grasp/grab/lift检索1490条（GRAB900/Omni590），pour16条；drawer、punch/boxing、squat/crouch均0，wipe两条为擦眼镜而非竖直擦拭。Omni训练只有7类大箱子抬起/移动/旋转/推拉/踢动原文。只是文字覆盖线索，零关键词不证明动作绝不存在，有关键词不证明完整任务成功；不新增逐条源语义/物理筛选。完整报告reports/current_prior_task_coverage_20261104/caption_retrieval_all_captions_v2.json。
- 论文八技能默认Full-Mix还包含AMASS/HumanML3D及Nymeria；已配对11424条未加入当前固定扩散检查点，MDM预训练残留不能当已核机器人覆盖。此前直接按论文八任务安排当前小基线全部扩展，范围过大。八项各500保留长期目标，Pick继续且优先PPO；其余批量需先证当前先验支持/小批验证，或独立版本补训练，再扩展，不自动热换当前模型，也不把推箱等新候选擅自替换用户的论文任务库。


## 2026-11-04最新确认：八任务各500；Pick库完成后优先独立PPO，提供参考视频

- 视频已交付：本机reports/pick29_videos_20261104含18秒双视角合辑、三条各6秒单片及哈希来源report，已检查实际机器人画面，勿重复生成。是参考姿态回放，不是PPO执行或抓取成功。
- 新隔离PPO入口stage3_pick500_ppo.py使用Pick500EnvCfg；4环境/2更新接线和检查点读回通过，证据O/ppo_integration_v1/status.json。无需重复；后续完成final_library_audit.json（真实500及manifest绑定）、最终库8912环境128步capacity，再准备新正式保护启动和100环境评估，不调用旧固定物体入口。原11GiB启动/8GiB+356MiB运行及GPU12GiB/RSS32GiB保持。

- 用户再次明确每任务独立PPO：Pick、Punch、Deep Squat、Deep Squat & Pick、Drawer Open、Pour、Vertical Wipe、Step & Punch各500条最终接纳参考，共4000。当前Pick已接纳255条，动态进度以服务器library_progress.json为准；其余七项尚待任务适配与生成。SuperMimic同库对照不计第九项，不另生成500。
- Pick500最终清单、加载/环境/PPO检查与既定资源保护满足后，优先在新目录训练Pick总3000更新，再100环境评估，不等其余3500条。旧两参考PPO结果/检查点全部保留；其余任务以按任务PPO为主，SuperMimic保留为已授权同库对照，不把原生迭代数与PPO更新数直接等同。
- 报告视频单独受限回放作业report_video_v1，先查launch/guard/log，不重复启动。只从已接纳库选3条不同目标位置，使用标定29DoF带手Isaac资产，20Hz双视图，明确参考姿态回放非策略执行/抓取成功；对象固定初始位置，不伪造携物轨迹。


## 2026-11-04：Pick批量原生场景验收已接通；未覆盖AMASS源头信息已分流

- 新原生批次脚本stage3_pick500_native_batch_v2.py按最多32条已完成且运动学通过的生成候选运行PhysX120帧注入回放，逐条核原文件/特征/目标位置/生成噪声绑定及原生报告，避免重新求解。当前累计原生接纳224/500，实际进度以reference_scale500_20261104/library_progress.json为准。已保存每批inputs/child/exit/summary/资源日志和含全部已接纳轨迹的partial_reference_bank.json，校验机器人状态内容非重复；计数不等于PD跟踪/抓取成功。
- 自动验收控制器包装PID1411802、实际PID以active_native_controller.json为准，目录native_drain_controller_v1。先查status/日志/active指针，不手动重复启动native_batch或控制器。它串行批量验收，等待新后优化结果；达到500时等待独立最终清单/训练入口检查，或初始来源耗尽时报告补采样缺口，不直接开训。源/优化器/进程路由native_pipeline_registry.json；后续追加新seed补采样时以原子方式登记新producer_dirs、optimizer_record_paths、optimizer_terminal_dirs、active_jobs，不改正在运行源码。原采样PID1402368、后优化PID1405357保持。
- 新原生批次保护器只计本项目对应进程及子进程，实测合计GPU峰值5940183040字节、RSS约6961971200字节，低于12GiB/32GiB；磁盘8GiB+456MiB。若保护触发只停止新原生批次自身，保留证据，不干预他人/旧采样优化。几何/碰撞是生成参考适用检查，不额外筛源GRAB/Omni。
- AMASS差集251名义ID涉及184个现存候选SMPL-X165列文件。只读小字段与NPY头读取（不解压完整pose、不更改源）显示全部声明120Hz；按固定HumanML源码9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23的整数步长及头部裁剪/区间规则，108条在界、143条超范围，不能据此直接启动全量重定向或擅自更改FPS。详见uncovered_source_header_probe_v1.json；后续先查超范围来源/时钟与逐文件映射，再选未覆盖且可用的小样做GMR+JAX/PyRoki29DoF，不用旧7状态滑窗低效算法重新跑40天。已完成11424条发布名义60→20Hz配对集不重复。


## 2026-11-04：ReactiveBFM11424条名义60→20Hz配对集已完成及独立技术验收

- 复用11424条现成29DoF轨迹与34132句原文，全部完成20Hz/最长10秒/22点263D及完整29状态转换，0失败；665378548字节运动文件，CPU一线程约305.5秒。每条文件/来源运动/文字绑定/20Hz网格/四元数/形状/统计重读通过，32条同采样器/FK/编码器复算最大差0；确切索引源组跨划分重叠0，精确特征跨划分重复0。按源路径分组保留全部轨迹，训练8146/验证738/测试2540，保留作者与HumanML原划分字段；仅训练侧264224特征帧重算统计。
- 增补时空条件统计与真正CPU数据加载器，三个划分分别4条实际数据加载/有效帧掩码/完整29状态/文字/263D/22点条件归一化通过。combined_stats_sha256为7f2dceda1e94c8649d1cf7aab48c8a384163dae6bab07db824fba8e4e9b692c1；主清单SHA256 8ee660fcc10c01a0f42225a091cc94f9df9bc9b37f337b51831f5f1d33f25538。数据目录reactive29_pairing_work_20261104/publisher60_20hz_full_v1，统计loader_v1/combined_stats.npz，最终收据completion_receipt.json。原构建report中的independent_file_audit=pending为原始时点，实际后续独立审计及加载已通过；不回头重跑或覆盖原报告。
- 这是明确采用发布方60Hz声明转换的名义时间版本，不把历史20Hz敏感性差异掩盖或声称逐条真时钟已核；采用发布原文弱监督，294条文字非逐字相同仍原样保留且标注，未新增逐条语义/物理筛选。未合并或重训正在生成参考所用模型，500任务参考两条GPU流水线保持。此11k既有覆盖无需再重定向。
- HumanML非镜像14614名义ID中尚有3190不在此发布集：1191属于单列HumanAct12非AMASS源、1748需现存归档/源解析、251在服务器找到名义原始候选路径；只是可用性/名字线索，不证明身份或时间，差集和逐条来源uncovered_source_triage.json保留。后续优先对确实未覆盖且来源/时间可用者做GMR+JAX/PyRoki，不重复已处理11k、不启动旧GRAB源重定向。


## 2026-11-04最新要求：生成参考继续，同时并行处理AMASS/HumanML3D

- 用户明确参考生成保持，AMASS/HumanML3D文字—29DoF G1参考数据并行推进。沿用用户此前指定的ReactiveBFM公开11424条，不重复对已覆盖内容人体重定向；未覆盖名义ID另列coverage，后续仅对确实需要的原始源确认差集/时间后再GMR+JAX/PyRoki。当前500任务参考采样PID1402368及后优化PID1405357保持，已训练扩散先验不更改。
- 已核作者当前数据README Git blob04ebf854bf9c6deaac9623509c4b49c6794c7041：根xyz/xyzw+29关节、Z-up、所有g1_36dim声明60Hz。CPU低优先级一线程包装PID1408020独立构建publisher_nominal60_to20Hz_29DoF_weak_text，目录reactive29_pairing_work_20261104/publisher60_20hz_full_v1；最多10秒、22点/263D、完整29状态、发布原文全部保留，并记录HumanML原文/来源成员哈希及文字差异。60Hz仅发布说明的名义时间依据，不冒称逐条真实录制时钟；历史20Hz敏感性试验仍保留。未新增人工语义/物理筛选，不启动新扩散训练，不改变当前3161数据/模型。
- 全11424条各自转换后读回，按确切索引源路径进行test>val>train分组、保留官方与发布方切分字段、仅训练侧重算统计；不能把路径/ID当真实录制身份。输出预算1GiB、磁盘8GiB+456MiB底线，CUDA隐藏不占GPU。publisher60_20hz_full_v1/status.json、同名.log和launch收据为准，运行不重复；完成后watch_reactive29_pairing_audit_20261104.py自动独立核所有文件/来源/文字/20Hz网格/统计及32条同编码器复算，检查independent_audit_v1.json和automatic_audit_exit.json，失败保留实际前缀新目录续。
- 本轮论文核对：v2附录V-E/TableIX明确共用运动跟踪奖励并加任务项；Pick物体高度>0.95m，Deep Squat & Pick>0.65m，Drawer位移>=0.05m等；Deep Squat/BimanualPick额外约束根平面线/角速度。v2称RL沿用DreamControl，原公开README以--task单独训练/独立任务检查点，支持按任务训练技能策略；v2没有公开统一8任务Actor联合训练方案。共享PPO算法/MLP结构不代表共享同一策略权重，也不是500条各训一次。


- 后优化续做最新记录：v2在ID11的右腕pitch出现0.001222407rad越限，超过原1mrad最大投影规则；保留7条已完成ID4..10及raw/log，ID11记录为拒绝，未放宽阈值，也不重复求解。v3包装PID1405357只处理余488个未做ID，输出postopt_remaining488_jit_v3，源码stage3_pick500_postopt_follow_v3.py，--exclude-records引用v2七条、--exclude-ids11。个体数值越限现在形成拒绝收据并继续队列；环境/内核错误仍失败留证。采样PID1402368继续；最终500合格需补采样被拒绝项，不能500候选直接当完成。

## 2026-11-04：四条新Pick参考已通过完整几何/场景链路，余496条边采样边后优化

- 当前新的500目标库已接纳4条（非500完成）；四条JAX/PyRoki后优化、微小限位投影后的独立NumPy FK及全碰撞网格地面检查、每条120帧真实PhysX几何回放均通过。新物体位置与参考目标一致，原生物体重置误差0；8环境32控制步四次重置覆盖4参考ID，Actor改变私有参考ID时观测差0。默认500条训练入口正确拒绝四条试库，未启动PPO/SuperMimic；几何回放不等于PD跟踪/抓取成功。
- 可复用JIT求解器stage3_pick500_postopt.py：首次46.46秒，后续2.44/2.64/2.84秒。新src/stage3_pick500_postopt_follow_v2.py持续读取完整采样记录，包装PID1404446，输出postopt_remaining496_jit_v2；先查收据/进程/日志/status/records，不重复。继续现有采样PID1402368，candidate_remaining496_v1；二者各限5%GPU，磁盘底线8GiB+456MiB。后优化库仍需原生回放再计接纳，不直接改标志。
- 新隔离stage3_pick500_env.py与stage3_pick500_replay.py绑定每参考物体位置，避免沿用旧固定物体使动作失配。最终库需500不同源噪声/特征及机器人状态内容、文件哈希、场景回放及可加载证明；旧资产/旧两参考PPO不修改。完整证据reports/stage3_reference500_20261104，服务器reference_scale500_20261104/native_pilot_v1_launch/report.json。
- 用户本轮仅询问联合PPO是否可行，未授权切成多任务统一Actor。当前仍每任务一套策略，每任务500参考一起参加该任务PPO（不是每条分别训），PPO通过仿真交互更新。若做统一8任务策略还需任务条件、多任务场景和奖励/采样机制，不能简单把4000条放进Pick环境。按Pick实测纯采样500约2小时、4000约16小时（单路当前5%GPU；不含其他7任务尚待实现的场景/条件、补采样及最终检查），不能把它当4000合格完工期限。


## 2026-11-04最新用户要求：每项技能500条生成参考后再进行PPO

- 用户明确每个验证子任务需500条生成动作用于PPO。执行范围仍为8项技能，目标共4000条通过任务适用检查的不同参考；500原始候选不等于500已接纳参考，不复制文件凑数。保留既有两条参考的3000更新PPO及1/100评估，另建500参考Pick基线，再在同一500参考上做SuperMimic对照。先扩充Pick，不在两条旧参考上启动正式SuperMimic；后续任务先各自500条再训练。现有2条只属小规模基线，不能说论文规模复现已完成，也不能将1%全部归因参考条数。
- 已实现隔离分批采样入口stage3_sample_pick29_batch.py：复用最终10000步扩散先验，每条独立初始噪声seed/hash及生成特征内容hash，20Hz/120状态，按当前桌面XY+-0.06m变化接触目标；每批落盘清单与进度，受磁盘/5%GPU保护。先4条真实采样验证，再按实测扩大；后优化与原生场景回放后计入500接纳数，失败补采样。扩充对象位置必须同步后优化目标、原生场景和训练重置，不能沿用旧固定物体位置使参考失配。
- 500参考计划及当前计数见本地reports/stage3_reference500_20261104/plan.json和服务器stage3_pick29_20261103/reference_scale500_20261104/plan.json。新500参考训练与旧2参考结果严格分开目录/清单，不覆盖检查点。29DoF、31策略动作、102实时无参考Actor、20Hz参考、50Hz控制及每次Pick3000×24真实交互预算保持；继续执行共享GPU/磁盘保护、只做仿真、Nymeria暂停。11GiB仅正式训练启动门槛，不阻碍受限参考采样构建。


## 2026-11-04：Pick PPO已完成3000次更新及100环境评估，转入SuperMimic适配

- Pick最终model_2999.pt SHA256 bc38324f751513e0f72da5d17eb47cacd96fef2a172ef20388c3156e094b6d7e；独立核对续训251..2999连续，累计3000更新，模型102输入/31输出有限。原训练、保护器和交接控制器均正常退出，不得重启Pick或将自然退出当故障。正式100环境评估1/100满足当前项目高度判据，不代表论文分布/真机成功；结果保持原样归档。
- 原评估接触时刻统计的浮点跨界诊断有误，独立新目录按整数控制步复测；同种子/模型仍1/100，75例到接触时刻，腕目标误差平均0.27223m。原主评估不覆盖，诊断报告pick_contact_diagnostic_3000_v2。低成功率如实保留，不能将完成预算当任务成功。
- 当前顺序pick_supermimic→其余七技能。TextOp原生IsaacLab3/MuJoCo-Warp场景无手/物体且Actor含未来参考，不能直接替换当前PhysX任务。选择复用固定SuperMimic源码的网络/归一化原语及世界模型+BPTT优化策略，在已有Isaac4.5运行时适配；不重装、不修改TextOp项目。属于明确标记的SuperMimic-style改编，不冒称原生CLI或作者实现。保持同PhysX资产/场景/奖励、29身体+14手指物理关节、31策略动作、102实时Actor且无参考；参考仅用于奖励头/初始化。
- 真实8环境128步/32世界模型更新/1次BPTT及独立照文档4环境64步/16模型更新/1次BPTT均通过；观测与原任务最大差1.19e-7，六项模型/优化器保存读回一致。证明有限接线，不证明任务成功、全程或8912环境资源。runtime_bridge_spec.json c7724ac5保留；其smoke_scope描述作者8环境版本，独立命令实际为4环境版本，二者已明确区分。
- 新online训练循环、32行有界回放和完整断点恢复正在验证。4项CPU回放顺序/回环/恢复/损坏拒绝测试通过；8912正式预算仍3000×24=72000真实向量步、641664000转移，世界模型热身计入，想象步不计；正式未启动，不能据原型通过标记就绪。检查stage3_supermimic29_20261103/adaptation_status.json及最新reports，勿重复启动已有见证。预留11GiB启动和8GiB+356MiB运行底线、12GiB显存/32GiB RSS；当前/data约11.55GB低于启动线，需真实资源核算与合规空间安排，不以降低预算放行。
- 本地证据reports/stage3_supermimic29_20261103；原数据、全部结果和检查点保留。其余29DoF任务尚待适配，不用Pick改名。只做仿真，Nymeria暂停，不重做数据/重定向/额外源语义筛选。


目标：以已有 DreamControl/Isaac Lab 工程和已归档控制基线为基础，重现论文中机器人空间数据构建、条件扩散先验训练、参考轨迹生成及后处理、PPO 策略训练和仿真评估。不执行真机部署。论文采用 OmniControl + MDM 初始化；从头训练人体 MDM 不是本复现目标。

本计划采用论文 [arXiv:2604.00202](https://arxiv.org/html/2604.00202) 和 [原 DreamControl 代码](https://github.com/GenRobo/DreamControl/tree/ec28b508313d6837de52c68247fbc8cf81fe4935) 为依据。作者 [DreamControl-v2 仓库](https://github.com/GenRobo/DreamControl-v2) 当前呈现为项目网页；未确认其含训练代码和完整权重。未公开的实现细节必须明确记录为近似或待确认，不得填成已核实参数。

## 执行顺序与验收

2026-11-03 最新新增计划：**当前Pick PPO训练及100环境物理仿真评估完成后，先实现SuperMimic框架替换并完成同一Pick任务训练/评估，再继续其余七项技能**。复用TextOp已经配置的SuperMimic源码和环境，不重装、不改动TextOp项目；当前可用候选是`textop-reproduction-20261024/step34_20261029/user_vibed_native_env_20261103_v1/venv`及其固定提交源码。具体接续和匹配配置见服务器`stage3_supermimic29_20261103/handoff_plan.json`，本地`reports/STAGE3_SUPERMIMIC_HANDOFF.md`。8912环境、29身体关节+双手31动作、102实时Actor输入、同参考/场景/奖励/重置/评估、50Hz控制及20Hz参考保持一致。按3000个匹配轮次×24个控制步，对齐每环境72000步、名义641664000条真实环境转移；SuperMimic原生迭代、世界模型更新和策略更新分别记录，不能把原生3000次直接当PPO3000轮。算法特有参数逐项说明，不能保留PPO损失却改名SuperMimic。原生SuperMimic未来参考Actor、29维无手动作和世界模型需适配当前自主交互任务，尚未实现、不称可直接开训；完成验证后优先用于后续任务。保留现有PPO基线与检查点，当前训练不重启，不并行启动另一正式训练，所有任务终点仍为物理仿真评估。

2026-11-03 最新用户明确：**第四步Sim2Real Deployment不复现，全部任务以物理仿真平台上的验证与评估为终点**。继续现有Isaac Lab / Isaac Sim（PhysX）的29DoF任务训练、100环境评估和结果归档；不准备或执行真机部署，不以硬件实验作为完成门槛。仿真评估报告如实记录任务成功率、测试条件和失败情况，不将仿真表现表述为真机有效。当前Pick总预算3000及后续任务自动接续保持执行，不重启正在运行的训练。

2026-11-03 最新用户将**Pick总训练预算改为3000轮**（包含已完成更新，非追加3000轮）。以最近完整检查点恢复模型、Adam和观测归一化；在独立新目录续训，保留原5000预算运行的日志及全部检查点。活动运行、目标轮数、最终检查点和保护器以stage3_pick29_20261103/active_pick_run.json为准；评估/自动接续须读取该配置，不再硬编码model_4999或5000完成条件。切换期间budget_change_request_3000.json标记处理中，勿重复启动或并发切换。

2026-11-03 最新用户授权：**完成首个任务后自动衔接其他任务复现**，无需逐项确认。以论文Table III的8技能为范围，执行顺序为Pick→Punch→Deep Squat→Deep Squat & Pick→Drawer Open→Pour→Vertical Wipe→Step & Punch。每项完成计划训练、检查点复核和100环境评估后归档真实成功率，再推进下一项；训练失败/资源保护停止需先留证修复，不能假装完成。其余29DoF任务适配尚待实现，不用旧27DoF任务冒充。服务器stage3_skill_sequence_20261103为执行记录，后台控制器等待现有Pick训练结束并自动评估、交接下一项；已按本次明确要求创建dreamcontrol-v2接续自动化，每30分钟继续本聊天工作，覆盖历史“旧自动已删除，不自行重建”限制。维持29DoF、源数据不重复处理、共享资源保护、不做真机和Nymeria暂停。

执行进展（2026-11-03）：第二阶段首轮10000更新已完成，29DoF Pick两条生成参考已通过独立运动学/原生碰撞回放；4环境2轮PPO及8912环境32步容量检查通过。Pick已按最新用户预算从旧model_250.pt（251次已保存更新）恢复至总目标3000，当前PID1343796、screen dcv2_pick29_3000_v3_20261103，目录stage3_pick29_20261103/runs/pick29_8912_3000_resume_v3，已核255次实际更新。先查看active_pick_run.json及对应保护器，不重复启动、不重做已通过数据/重定向/参考审计。旧5000运行停止且结果保留；预算切换v1/v2学习前错误已留证修复，当前v3通过真实更新及完整恢复复核。训练结果和成功率待评估统计，不能把启动或损失降低当成功。

2026-11-03 用户授权：第二阶段完成后继续第三阶段Isaac Lab/PPO，首个任务已明确选定**Pick**。使用29身体关节+左右手二值开合，共31动作；Actor只读取实时本体与物体根坐标位姿，参考轨迹只用于训练奖励/初始化，不作为Actor必须输入。正式目标8912并行环境、5000次PPO迭代；先验证真实29DoF带手资产、关节/动作顺序、观测不泄漏参考、场景参考校准、奖励/重置有限性及小规模PPO，再测容量，满足第二阶段完成及合格Pick参考和共享资源条件后自动衔接。不能将只完成扩散优化步数当成参考已通过，也不拿27DoF旧策略直接充当29DoF。Nymeria暂停、不做真机部署；保护其他任务、原数据/结果/检查点。

2026-11-03 最新用户明确启动第二阶段Diffusion Prior Training，覆盖此前仅构建数据、不启动训练的阶段限制。使用已交付3161条GRAB+Omni、20Hz/22点/263D及完整29DoF状态，以OmniControl架构、MDM权重初始化、零输出Realism Guidance分支训练；文本与时空条件、x0/DDPM、CFG及采样空间梯度引导按公开代码和论文接入。现成OmniControl已训练空间分支不能冒称MDM从头初始化；后处理输出29DoF，不锁腰。只对生成后的参考轨迹做场景适用筛选，不对GRAB/Omni源数据追加物理过滤。先实际GPU训练/采样与检查点验证，再独立目录启动可恢复基线；保留8GiB磁盘底线、既定训练预算和共享GPU限制，保护其他任务。未公开v2细节明确记录改编，不回到逐条语义门禁。

2026-11-03 最新用户再次明确：**只按论文中构建文本—G1参考轨迹数据的步骤处理，不做其他额外处理**。采用已配对的来源文字作为弱监督，执行20Hz、最多10秒、22点/263D、朝向规范化、来源拆分与统计及可加载性核验；G1保留用户指定29DoF。取消逐条语义重标、接触证明、额外物理筛选等前置要求，不以历史`caption_accepted`或`training_ready=false`代表本次数据构建未完成；这些字段保留其原有独立语义/训练状态。全部3161条纳入主基线，风险仅保留为元数据，旧2609条阶段不作主数据集。当前只构建数据，完成时通知，不启动训练。

2026-11-03 用户明确：**当前只构建和验收数据集**，数据集构建完后通知用户；暂不启动训练或以训练11GiB预算阻碍数据构建。持续执行既有磁盘安全余量、原始数据和共享资源保护；训练入口和训练资源留待后续训练阶段。

2026-11-03 最新用户要求：优先用 GRAB 的 DexWeave G1 与 MotionHub 文字、OmniRetarget 既有机器人轨迹和文字构建混合基线，尽快迭代；按论文20Hz和22点/263D，G1使用29DoF并保留29关节状态。暂将 ReactiveBFM 等大规模数据构建后移；GRAB/Omni不做人到机器人重定向、不作论文未用的物理动作筛选。数据来源、裁剪后的文字、时间、许可与资源保持如实标记，满足既有条件后启动训练。

2026-11-03 用户再次明确：直接复用 ReactiveBFM 公开的 11,424 条 AMASS–HumanML3D 文字及29DoF G1轨迹，核对时间、文字并构建条件扩散训练数据；不再对这部分重新重定向。GRAB 继续使用 DexWeave 现成G1轨迹和MotionHub文字，按用户要求不做物理动作筛选。追踪两来源的重定向方法、关节顺序、坐标和资产差异，记录适配与混合训练的限制；源文件和他人数据只读保留。

2026-11-03 用户进一步明确：原始 GRAB 人体动作预处理不再需要；继续 DexWeave 现成 G1 轨迹与 MotionHub 文字的处理，保留已下载原始数据，不做人体重定向或GRAB物理筛选。

2026-11-03 用户澄清：暂停原始 GRAB 人体数据的预处理和人体重定向；**继续**使用 DexWeave 的 G1 轨迹和 MotionHub 文字构建训练数据。现成机器人轨迹的下载、配对、核验和转换均继续；保留原始人体数据，不作 GRAB 物理动作筛选。此前“暂停所有 GRAB 处理”的理解撤销。

2026-11-03 用户指定 GRAB 采用 DexWeave 的 G1 轨迹与 MotionHub 对应文字构建训练数据，检查现有原始 GRAB 的覆盖和文字一致性；不对该来源执行物理动作筛选，按轨迹身份、关节、时间、文本和许可逐步适配。

2026-11-03 用户最新要求：优先查找并复用已重定向的 G1 轨迹，避免对已覆盖的 AMASS/HumanML3D 和 GRAB 重复重定向。先核对 29 自由度关节顺序、时间、文本匹配、来源及适用许可，再适配训练格式；未覆盖部分按授权继续处理，其他项目数据只读保留。

2026-11-03 用户再次明确：按论文文本—G1动作先配对、转换至20 FPS、22关键点/263D并接入条件扩散训练；机器人关节数采用29DoF。记录源文件、原文与时间、29关节状态并保留空间约束支持。GMR+JAX/PyRoki及29DoF是用户指定改编；GRAB/Omni不做物理动作筛选。严格区别技术预览、文字语义核验和最终可训练数据，条件满足后直接训练。

2026-11-03 用户澄清：遵照论文，不对 GRAB 和 OmniRetarget 做物理动作筛选；不因机器人难以跟踪而剔除这两个数据集。继续检查文件技术完整性、时间及文本匹配；其他人体源的适用筛选与后续仿真评价按授权推进。此前对 OmniRetarget 做物理筛选的指示由本条替代。

2026-11-03 用户最新明确选择：采用 G1 **29 自由度**，允许偏离论文锁腰27自由度设定；人体 GMR、JAX/PyRoki 优化、OmniRetarget 输入、训练与仿真均以29DoF资产和关节顺序为目标，不再将腰roll/pitch锁住或删列。论文使用27DoF，但未解释锁腰动机。此前27DoF试验仅作历史记录，不进入新训练。重新验证官方OmniRetarget 29DoF与所选资产的一致性及文本配对，能直接使用时不再重定向。

2026-11-03 最新补充：以 GMR 建立人体到机器人运动，再采用 JAX/PyRoki GPU 优化器进行后续优化；这是组合近似，不能声称 GMR 本身由 JAX/PyRoki 实现或作者原版。重定向后参考 TextOp 跟踪和原 DreamControl 筛选代码设计可复算的跟踪/物理筛选，保留筛选理由及源数据，再构建运动—文本训练数据。核验 OmniRetarget 官方轨迹的机器人及关节配置，判断是否可直接使用；不因型号名相同就假定 29DoF 与锁腰 27DoF 等价。优先完成接线和实测，资源与质量真实满足后训练。

2026-11-03 用户最新指示：清理本项目 AMASS/HumanML3D/GRAB 的已处理数据，保留全部原始人体轨迹和原始归档；人体轨迹全部使用 GMR 重新重定向。停止此前PyRoki重定向，不把GMR称为作者v2原始方法。OmniRetarget既有机器人轨迹不再重定向，保留并继续其文字配对、质量与训练验收。按用户授权仅删除已明确识别的派生数据，来源与其他项目材料不可误删；范围、清理日志及尚待分类项目见STATUS。

2026-11-02 用户最新指示：先集中处理 HumanML3D / AMASS、已有 KIT 和 GRAB；继续复用已有归档，按来源、原始时序、文本、机器人转换及适用质量逐批核验。OmniRetarget 与其文本处理暂缓为后续工作。较小可用子集的训练仍以既有授权的来源、质量、资源门禁为准；不等待完整 Full-Mix 或额外存储。

2026-11-01 用户新要求：HumanML3D 尚未完成，算法实现与现有候选文本、物体轨迹同步推进；在固定版原 DreamControl 基础上逐项接通扩散训练/采样、27 自由度后优化、物体条件控制与 Isaac 仿真评估。未公开的 v2 环节明确标注为近似或未完成，测试结果与真实训练、数据验收分开。已完成旧 DDPM/CFG、空间约束和旧 PyRoki 的隔离接线测试，既有 GPU 批处理及技术审计已完成；继续在资源保护下审阅数据、实现可复算验收。只有来源、许可、文本/时间、质量与资源条件实际满足后才自动训练。

2026-11-01 当前执行重点：使用适用许可的数据，完成可追溯的人体来源到机器人、文本时间和分组统计的最小可复算子集，并实现真实质量验收；官方测试协议与辅助划分分别保留。最新 OMOMO 技术预览已只读适配原 DreamControl 数据加载接口并完成 train/val/test CPU 冒烟，但 `1,717` 条均未获文本接受且真实来源许可/时钟/物理质量未验收；实际 `--run` 失败关闭检查已确认拒绝且不写入运行目录。因此不得开训。八候选人体预览也不代表 HumanML3D 全集或合格数据。三份既有 OmniRetarget 归档已与固定官方版本的指针哈希及本机/服务器文件比对一致，尚未因此获得 OMOMO 上游数据与文本的完整授权。此前完成的 MoSh 253 窗和裁后 20 Hz/263D 导出不重复；按 STATUS 最新证据继续处理，而非仅改训练标志。用户已明确允许与其他任务共享有足够余量的 GPU，须限制本任务显存、不停止或中断其他任务；资源在启动时重新核查，下文历史资源数字不代表当前值。

### 0. 固定来源和资源

- 固定论文、原代码版本、运行环境、G1 资产哈希，以及数据来源、许可、帧率和坐标约定；保留原项目未提交更改，不覆盖原有结果。
- 运行只读资源与数据清单：GPU 空闲、磁盘空间、数据可读性；为新工作使用独立目录、日志和检查点。
- 验收：每个阶段的输入来源可追溯，形成固定配置、哈希和状态记录。当前两张 A6000 有其他任务，`/data` 可用约 21 GB；容量与 GPU 需在启动前再次检查。

### 1. 统一机器人空间数据

1. 审计可用原始 AMASS/HumanML3D 与既有 G1 动作；分别记录人体动作、文本映射、G1 轨迹和原始样本的对应关系。不得把不同项目的切分直接等同于论文 Full-Mix。
   - HumanML 官方划分中有 874 个映射源路径跨划分，名义索引存在交叉的情况须在核对实际索引与时间含义后处理；官方协议和来源分组评估要明确区分，见 STATUS。
   - 两条已确认标记帧对应的 SSM 候选已完成 22 点位置镜像与 20 Hz 线性时间网格见证；尚需抗混叠、姿态旋转、文本语义、其他源及机器人验证，见 STATUS。
   - 已生成不改动官方文件的保守来源互斥辅助清单并独立检查；它仅隔离完全相同的索引路径，训练资格及别名/录制级泄漏仍待验证，不能替代官方评估。
2. 对人体原始数据构建基于 G1 的关键点匹配、关节限位、足部约束与优化；如果输入已有 29 自由度 G1 动作，须显式重优化为腰部 roll/pitch 锁定的 27 自由度，而不是直接删列。记录每条轨迹的优化前后误差和失败原因。
3. GRAB、Nymeria、OmniRetarget 可获得时分别处理：Nymeria Xsens→SMPL 拟合与有时间戳的交互片段检索；GRAB/OmniRetarget 按论文口径处理。保留原始数据许可和来源；没有数据时不伪造 Full-Mix。
   - GRAB 1,335 条已有逐成员字节哈希和主体/资产来源清单，并完成独立复核（见 STATUS）；该清单尚不包含已证实真实时间、机器人轨迹或无泄漏划分，后续处理须保持原成员关联。
4. 20 Hz 采样、10 秒上限、论文表 VII 的 22 点、HumanML 263 维编码。仅用训练集计算归一化统计；按原动作来源分组划分，检查跨集合重叠。
- 按论文 III-A 的顺序，从 AMASS/HumanML3D 和 GRAB 的原始人体参数重建人体关键点并重定向 G1，之后才做 20 Hz 处理；已是机器人轨迹的 OmniRetarget 单独按机器人格式适配。论文对人体动作使用 12 个髋、膝、踝、肩、肘、腕关键点，目标包括相对角度、相对位置、脚滑和人机比例；作者基于 JAX/PyRoki，论文未公开足够细节来确定完整优化目标、权重。当前 CPU 的 GMR 初始化与 12 点精化是近似原型，须继续实现并验证论文描述的要素，不能称为作者实现。
- 2026-10-30 用户指出 GMR 不符合论文：经复核 III-A1 和 III-A2，OmniRetarget 机器人轨迹是 Full-Mix 的额外数据源；作者描述的原始人体动作重定向采用 JAX/PyRoki，并未说明所有原始人体轨迹使用 OmniRetarget 算法。正式复现主线优先实现 PyRoki 路径；已有 GMR 结果仅为探索性基线，不作为论文方法或正式训练数据。DreamControl 一代 `TrajGen/sample/retarget_helpers/_utils.py` 有 PyRoki 局部/全局匹配、脚接触、脚滑及碰撞目标，可研究适配，但一代代码、任务条件和权重不等同 v2 作者实现。需对照关节对应、输入坐标、时间、损失与依赖版本，在隔离环境验证；当前找到 PyRoki/JAX/JAXLS 源或库，但所探测 Python 环境缺少 yourdfpy，尚未通过导入或实际求解。
- 论文 III-A4 明确对 GRAB 和 OmniRetarget 不施加所述可行性筛选；对解码失败、不有限值、缺失引用等技术异常单独留证和隔离，不把实验性质量阈值作为论文已有的数据删除规则。物理诊断用于发现错误、描述局限及后续控制研究，不要求每条参考运动先证明在简单 PD 下能稳定跟踪才开展任何试验。
- 验收：每种来源均有数量、数据哈希、关节单位/顺序、时间和坐标系；随机样本的 FK、特征反解、关节范围、关键点误差、脚滑/接触检查可复算。生成 AMASS、AMASS+Nymeria 和 Full-Mix 三组数据及对应拆分；缺失来源时仅记录可执行子集。

### 2. 机器人空间扩散先验

1. 复用 OmniControl 的条件 Transformer、Realism Guidance 和 DDPM 训练代码；对 MDM 权重作键名、形状、加载覆盖率检查，确认新控制分支初始化及训练参数。不能将只包含人类先验的旧 OmniControl 检查点标为 v2 训练完成。
2. 建立机器人空间数据加载与文本、时空控制 mask，训练前检查归一化、有效帧、无效帧遮罩、文本/运动配对和训练/验证隔离。独立记录权重初始化与真正的断点续训；原训练脚本将模型文件名数字视作 resume step，且优化器恢复在该版本中被注释，不能把权重初始化误当完整恢复。
3. 根据论文采用 AdamW、学习率 `1e-5`，将其他未公开或代码差异的超参数列明；论文表 V 的预训练实验约 `102k` 迭代。分别训练三种数据组合及零样本一代基线。
- 验收：真实训练数据和日志、检查点、训练步数及随机种子完整；重载检查点并独立生成样本；在对应留出集评估 FID、R-Precision、Diversity、Control L2 和 Skating Ratio，并明确采用的特征提取器、条件和样本量。对比论文表 I、II、V，不将代理值写成论文指标。

### 3. 条件采样与 G1 参考轨迹

1. 根据论文附录配置各任务文本、时空约束、CFG 和空间梯度引导，直接在 G1 表征上采样；保持论文区分的任务和目标分布。
2. 从 263 维表示优化还原为 G1 关节角与根姿态，并根据任务添加手部状态；实现自动筛选及任务特定后优化：目标误差、关节/根高度、足部稳定、障碍物和手腕姿态等。论文只披露部分阈值，其余标为明确配置和近似，不冒称官方数值。
3. 逐任务记录输入、随机种子、生成量、保留量、剔除原因、生成/优化前后误差和输出哈希。参考规模按论文具体实验固定；500 条不是所有实验都必须达到的单一判据。
- 验收：轨迹能加载到 Isaac Lab，动作形状、约束、手部时序和任务场景/物体位置经检查；未通过的轨迹不能进入正式训练。保留物理真实性的边界：启发式筛选不等于动力学验证。

### 4. 无参考输入的仿真 PPO

1. 在隔离的 Isaac Lab 工程中复用一代 G1 动作、场景、跟踪奖励和任务奖励，适配论文八项：Pick、Punch、Deep Squat & Pick、Drawer Open、Pour、Vertical Wipe、Deep Squat、Step & Punch。原项目中缺少的任务需补充实现，不可把功能近似的旧任务直接改名。
2. actor 接受本体感知和任务目标等可获取观测；参考运动用于训练奖励。逐函数排查隐藏的 `motion_lib` 依赖，包括手状态和目标函数；固定物理状态/目标、改变参考，验证 actor 观测和输出不变。任务奖励仍可读取参考或任务状态，并分别测试。
3. 用同一数据来源、预算和随机化范围进行 PPO 与正式评价。论文文字报告每任务 2,000–5,000 轮、约 8,912 并行环境；数字与一代设置不同，须记录论文原文并核对代码及机器条件，不静默改成 8,192。按论文表 III 的 100 个随机环境报告成功率与 Control Error；另保留接触/实际物体运动诊断。
- 验收：每个任务配置、轨迹、训练日志、模型、成功判据和评估视频可追溯；8 项与论文表 III 比较，记录失败和差异；同口径评估数据量缩放及必要基线。

## 目前状态

- PyRoki URDF 与旧 Pick USD 的左右肩局部 z 差约 10 mm 被固定腰关节坐标差补偿；S5 离线完整链中肩/肘/腕链接原点一致至约 `1.32e-8 m`，腰与躯干原点仍相差约 9–10 mm。碰撞形状、质量和真实 Isaac 环境尚未等价验证。
- 已只读检查旧 Pick 指向的本地 G1 USD：实例展开后有 27 身体关节、14 手关节及 32 个碰撞 API；静态顺序测试不替代真实 PhysX 顺序、碰撞过滤和接触验证。
- Isaac 动作管理器源码在模拟资产上与实时观测完成 CPU 接线；旧 Pick 左手开/闭目标相同，需结合真实 USD、任务意图和 v2 方法核实，不能把该模拟检查当实际仿真验收。
- S5 倒水片段已开展源物体姿态同步、固定张手与杯子凸包的独立静态距离区间检查；杯口凹部被凸包填平，不能由近零区间断言真实接触，仍须真实表面、场景和动力学。
- 初步检查 SMPL-X 姿态反射与官方关节点镜像：同主体两段各三帧存在毫米级残差；对其中三帧，静止骨架不对称和关节层级 FK 已解释所观察的骨盆相对残差，人工对称骨架仅为诊断。其他主体、模型版本和局部坐标仍须核对，不直接把该假设用作官方或训练姿态。
- 在单个约30 Hz SSM 关节点样本上对比了严格20 Hz时间网格的滤波与线性插值，并通过直接滤波求和及合成信号检查；结果仅为抗混叠技术见证，姿态、骨长、接触、真实时间与跨来源表现尚待检查。
- HumanML 原/镜像文本已逐行检查左右与顺逆时针字面映射及原始时间标签；两类具体文本异常已确认（`upright`→`upleft`、一例 `left-to-right`→`left-to-left`），不等于全部语义异常。NaN 标签在旧加载器中转为零的行为不能直接视为真实时间。官方文本/划分与任何辅助处理或评估继续分别保留。
- 详细实时进展以 [STATUS.md](STATUS.md) 为准。GRAB Subject 2 与 OmniRetarget 三个运动归档已上传并验证；Nymeria 按用户要求暂停，完整 Full-Mix 尚不可用。
- 29→27 的受限根部和关节拟合已在多样本做数值检查，足部与其他关键点误差仍存在取舍；尚未通过真实接触、碰撞、动力学和来源独立性验收，不等于论文的人体到机器人重定向。
- 条件扩散已有数据技术检查、受保护训练入口和训练循环原型；预训练人体模型上的 CPU 合成检查包含短期保存/重载，但未开始机器人模型正式训练。入口目前失败关闭，必须实现并接入真实质量验收才能启动；引导与初始化近似待明确。
- Isaac Lab PPO 的参考独立观测仍待环境接入及完整任务、奖励与评估验证；没有开始正式 PPO。启动训练前重新检查 GPU、存储、数据许可和已有任务，不停止其他进程、不覆盖已有检查点。

## 后续执行规则

优先把可核验的最小人体来源子集贯通到机器人扩散训练，而不是持续扩展单样本旁证；以 STATUS 中“扩散训练启动缺口与执行优先级”为当前执行顺序。保留来源、许可、时间和质量门禁，数据不合格或验收未实现时不启动正式训练。完整 Full-Mix、Nymeria、Isaac PPO 和逐条简单 PD 稳定跟踪不是最小扩散实验的统一前提；任何子集实验都必须标注来源和适用范围。

按上述阶段推进，每个阶段产出配置、代码、日志/指标、验收状态与可重现入口。优先处理可独立验证的实现，遇到数据授权或资源缺口保留阻塞原因；不将 smoke test、代理指标或初步实验当作论文复现成功。资源可用后才开始相应训练；不得重复启动已有运行或覆盖既有检查点。
