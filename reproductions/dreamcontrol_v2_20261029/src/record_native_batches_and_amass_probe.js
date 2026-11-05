const cp=require('child_process'),fs=require('fs'),path=require('path');
const L=path.resolve(__dirname,'..');
const py=`import json,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');O=R/'stage3_pick29_20261103/reference_scale500_20261104';W=R/'reactive29_pairing_work_20261104'
paths={'library_progress':O/'library_progress.json','native_controller':O/'native_drain_controller_v1/status.json',
 'native_controller_launch':O/'native_drain_controller_v1_launch.json','registry':O/'native_pipeline_registry.json',
 'uncovered_source_headers':W/'uncovered_source_header_probe_v1.json'}
out={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())} for k,p in paths.items() if p.exists()}
plan=json.loads((O/'plan.json').read_text());plan['tasks'][0]['new_admitted']=out['library_progress']['data']['native_admitted']
plan['tasks'][0]['status']='sampling_postoptimization_and_guarded_native_admission'
plan['current_processes']['native_admission_controller_wrapper_pid']=1411802
(O/'plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(out))
`;
const data=JSON.parse(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python -'],{input:py,encoding:'utf8',maxBuffer:5e6}));
for(const[k,v]of Object.entries(data)){const folder=path.join(L,k==='uncovered_source_headers'?'reports/reactive29_pairing_parallel_20261104':'reports/stage3_reference500_20261104');fs.writeFileSync(path.join(folder,k+'.json'),JSON.stringify(v.data,null,2)+'\n');}
const note=`## 2026-11-04：Pick批量原生场景验收已接通；未覆盖AMASS源头信息已分流

- 新原生批次脚本stage3_pick500_native_batch_v2.py按最多32条已完成且运动学通过的生成候选运行PhysX120帧注入回放，逐条核原文件/特征/目标位置/生成噪声绑定及原生报告，避免重新求解。当前累计原生接纳${data.library_progress.data.native_admitted}/500，实际进度以reference_scale500_20261104/library_progress.json为准。已保存每批inputs/child/exit/summary/资源日志和含全部已接纳轨迹的partial_reference_bank.json，校验机器人状态内容非重复；计数不等于PD跟踪/抓取成功。
- 自动验收控制器包装PID1411802、实际PID以active_native_controller.json为准，目录native_drain_controller_v1。先查status/日志/active指针，不手动重复启动native_batch或控制器。它串行批量验收，等待新后优化结果；达到500时等待独立最终清单/训练入口检查，或初始来源耗尽时报告补采样缺口，不直接开训。源/优化器/进程路由native_pipeline_registry.json；后续追加新seed补采样时以原子方式登记新producer_dirs、optimizer_record_paths、optimizer_terminal_dirs、active_jobs，不改正在运行源码。原采样PID1402368、后优化PID1405357保持。
- 新原生批次保护器只计本项目对应进程及子进程，实测合计GPU峰值5940183040字节、RSS约6961971200字节，低于12GiB/32GiB；磁盘8GiB+456MiB。若保护触发只停止新原生批次自身，保留证据，不干预他人/旧采样优化。几何/碰撞是生成参考适用检查，不额外筛源GRAB/Omni。
- AMASS差集251名义ID涉及184个现存候选SMPL-X165列文件。只读小字段与NPY头读取（不解压完整pose、不更改源）显示全部声明120Hz；按固定HumanML源码9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23的整数步长及头部裁剪/区间规则，108条在界、143条超范围，不能据此直接启动全量重定向或擅自更改FPS。详见uncovered_source_header_probe_v1.json；后续先查超范围来源/时钟与逐文件映射，再选未覆盖且可用的小样做GMR+JAX/PyRoki29DoF，不用旧7状态滑窗低效算法重新跑40天。已完成11424条发布名义60→20Hz配对集不重复。

`;
for(const name of ['PLAN.md','STATUS.md']){const f=path.join(L,name);let x=fs.readFileSync(f,'utf8');if(!x.includes('## 2026-11-04：Pick批量原生场景验收已接通')){let i=x.indexOf('\n');fs.writeFileSync(f,x.slice(0,i+1)+'\n'+note+x.slice(i+1));}}
const guide=`# 当前生成参考轨迹：29DoF G1 Pick基线

使用在3161条GRAB+Omni机器人运动/文字上训练完成的10000步条件扩散先验；新AMASS数据尚未用于此检查点。参考由任务文字、关节位置/时间目标和不同DDPM噪声生成，263维输出经JAX/PyRoki29DoF后优化、明确微小限位投影和独立FK/网格检查，再进行逐帧原生PhysX几何/碰撞回放。

当前文字为“Pick up the object from the table.”，物体XY在当前桌面中心目标周围+-0.06m变化，固定桌面/物体参数；这不是论文完整随机分布。右腕、根部和间隔膝部点用于扩散提示；后优化加入站立、脚部、桌面距离和腕朝向等约束。

| 名义时间 | 参考的设计目标 |
|---|---|
|0–2秒|站立，右手靠近物体|
|约2秒|右手到接触目标并闭合|
|2–4秒|右腕抬起约0.16m|
|4–约6秒|保持抬起姿态|

当前每条120状态、20Hz，采样时刻0–5.95秒；统一上限10秒，并非每条都10秒。后优化状态文件包含q(120,29)、root_pos(120,3)、root_quat(120,4,wxyz)、keypoints(120,22,3)、right_wrist_quat(120,4)、hand_closed(120,2)、joint_names、fps和sample_times，以及物体初始位姿、文字和生成来源元数据。二值手命令成组控制14手指物理关节；身体自由度29。

263D是扩散运动特征，不是263关节，也不能无损还原全部29关节；RL参考单独保存完整29状态。参考文件中的物体位姿用于匹配场景初始条件；后续物体是否被抓起由接触仿真和策略决定。

后续每任务500参考共同用于一次该任务PPO；每个并行环境在重置时选择参考、设置机器人与匹配物体位置，20Hz参考在50Hz控制时插值。参考用于初始化/跟踪奖励，Actor只接收102维实时本体/物体状态，输出29身体控制与双手二值命令。实际任务成功率另在训练后评估。

当前入库计数以[参考库进度](stage3_reference500_20261104/library_progress.json)及服务器同名清单为准。场景验收是运动学/几何核查，最终抓取成功率需PPO或SuperMimic的动态仿真评估；不做真机部署。
`;
fs.writeFileSync(path.join(L,'reports/STAGE3_GENERATED_REFERENCE_TRAJECTORIES.md'),guide);
console.log(JSON.stringify({library:data.library_progress.data,controller:data.native_controller?.data,headers:{sources:data.uncovered_source_headers.data.unique_sources,counts:data.uncovered_source_headers.data.status_counts}}));
