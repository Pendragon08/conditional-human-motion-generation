const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const L=path.resolve(__dirname,'..'),D=path.join(L,'reports/pick500_formal_start_20261104');fs.mkdirSync(D,{recursive:true});
const script=String.raw`import json,hashlib,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
active=json.loads((O/'active_pick500_run.json').read_text());run=Path(active['run_dir'])
paths={'active_run':O/'active_pick500_run.json','training_status':run/'status.json','controller_status':O/'completion_controller_v1/status.json',
 'final_library_audit':O/'final_library_audit.json','capacity':O/'ppo_capacity8912_v1/status.json','cpu_audit':O/'final_library_v1/cpu_audit.json',
 'reference_bank':O/'final_library_v1/reference_bank.json','replacement_completion':O/'replacement_queue128_v1/report.json',
 'archive':R/'checkpoint_archival_20261104/manifest.json','first_checkpoint_audit':O/'completion_controller_v1/first_checkpoint_audit.json'}
j={k:json.loads(p.read_text()) for k,p in paths.items()}
assert j['training_status']['status']=='ppo_running' and j['training_status']['policy_updates']>=1 and j['training_status']['reference_count']==500
assert j['final_library_audit']['status']=='passed' and len(j['reference_bank']['references'])==500
assert sha(paths['reference_bank'])==j['final_library_audit']['manifest_sha256']==j['training_status']['reference_manifest_sha256']
assert j['capacity']['status']=='capacity_completed' and j['capacity']['num_envs']==8912 and j['capacity']['steps']==128
assert j['first_checkpoint_audit']['passed'] and j['archive']['status']=='archived_verified_complete'
receipt={'status':'reference500_admitted_and_formal_PPO_updates_observed','reference_count':500,
 'manifest_sha256':sha(paths['reference_bank']),'manifest':str(paths['reference_bank']),
 'final_audit':str(paths['final_library_audit']),'final_audit_sha256':sha(paths['final_library_audit']),
 'first_actual_update_audit':str(paths['first_checkpoint_audit']),'first_actual_update_audit_sha256':sha(paths['first_checkpoint_audit']),
 'training_run':str(run),'observed_policy_updates':j['training_status']['policy_updates'],'total_updates':3000,
 'actual_training_pid':j['training_status']['pid'],'num_envs':8912,'body_dof':29,'reference_fps':20,'frames_per_reference':120,
 'hand_commands':2,'actor_inputs':102,'actor_reference_input':False,'server_free_bytes_observed':shutil.disk_usage('/data').free,
 'training_start_floor_bytes':j['training_status']['required_start_free_bytes'],'completion_is_not_Pick_success':True,
 'archive_manifest':str(paths['archive']),'archive_manifest_sha256':sha(paths['archive']),'archived_bytes':j['archive']['total_bytes'],
 'scope':'generated reference admission and actual PPO startup; training and final100-env evaluation still in progress/pending'}
output=O/'usable500_training_started_receipt.json';output.open('x').write(json.dumps(receipt,indent=2));j['start_receipt']=receipt
print(json.dumps(j))
`;
const j=JSON.parse(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
for(const[k,v]of Object.entries(j))fs.writeFileSync(path.join(D,k+'.json'),JSON.stringify(v,null,2));
const title='## 2026-11-04：Pick最终500条已验收，8912环境3000更新PPO已开始';
const note=title+'\n\n- 从507条原生接纳中按candidate ID确定性选恰500，manifest SHA01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572，O/final_library_v1/reference_bank.json。全628原始候选GPU噪声复算/源目标/时间/去重审计完毕，选中500逐文件状态/原生证明/FK/限位/目标重读通过；20Hz、每条120状态约6秒、29身体关节加双手二值。默认ReferenceBank500真实加载及8912环境128控制步通过，物体重置误差0、Actor私有参考变更差0；final_library_audit.json status passed。不是PD跟踪或抓取成功。无需再生成补样、重复最终库或容量。\n- 已真正启动新Pick PPO：'+j.training_status.pid+'，S/runs/pick500_8912_3000_v1，8912环境/总3000更新，当前快照'+j.training_status.policy_updates+'更新，实际以status/metrics为准。新pointer O/active_pick500_run.json，保护/评估接续控制器仍1427653、completion_controller_v1；六个训练/环境/评估/最终化入口源已绑定，不修改运行源码或重复启动。实读第一正式checkpoint model_0.pt SHA1820678bf4243f5ec87084d8ef98df7cbf3c3e48a67622e293a06533e2514d1d，iter0=1更新、真实优化器状态、有限权重、500库哈希/102实时Actor/31动作/29DoF均通过，完成收据usable500_training_started_receipt.json。训练与最终100环境评估尚未完成。\n- 为保持11GiB启动预算，58个本项目非活跃检查点已完整复制到本机并两遍SHA校验、服务器再核内容/inode/device及无打开FD后冷归档，释放2004750460字节。包含两个扩散中间点、六个SuperMimic接线/恢复诊断点、旧Pick v3的50个中间点；原始GRAB/Omni/人体源、当前扩散最终10000、旧Pick最终2999、当前新训练模型/日志/报告/参考均留服务器，两个受保护最终模型哈希前后未变。全部检查点内容仍完整保留，未丢弃结果。\n- 归档索引：R/checkpoint_archival_20261104/manifest.json；本机checkpoint_archive_20261104/manifest.json列每个旧路径→C/D本机文件/SHA/尺寸，C位于本项目checkpoint_archive_20261104，D位于D:/DreamControl_v2_checkpoint_archive_20261104。源路径附.pt.archived.json。需要旧中间/诊断权重时先用src/restore_archived_checkpoint_20261104.js按manifest相对路径校验恢复，不因文件归档而重新跑已通过见证，也不能把归档标记当权重。正式最终模型不需恢复。\n- 启动前实际/data约12.08GB，满足11GiB，不是降低预算；本任务GPU约8GB/RSS约16GB低于12GiB/32GiB。当前正式状态看leaf和controller.resources，controller顶层旧free_bytes/shortfall_bytes仍是此前等待空间的历史值，勿误判未开训；library_progress里的formal_training_started=false也是原生阶段旧快照，以新active_pointer/leaf/实测checkpoint为准，不改旧证明装完成。后续控制器核model_2999后真实100环境评估归档，预算完成与高度型成功率分别报告。其余四任务顺序与暂缓任务保持，SuperMimic额外同库对照不替代四项PPO。\n\n';
for(const name of ['PLAN.md','STATUS.md']){const p=path.join(L,name),s=fs.readFileSync(p,'utf8');if(!s.includes(title)){const n=s.indexOf('\n');fs.writeFileSync(p,s.slice(0,n+1)+'\n'+note+s.slice(n+1));}}
fs.writeFileSync(path.join(D,'启动说明.md'),'# Pick500训练已开始\n\n500条20Hz、29DoF生成参考完成最终文件/来源/时间/目标/FK/原生与真实加载验收，8,912环境128步容量检查通过。新PPO总3000更新，训练结束后自动100环境物理仿真评估；训练启动不等于抓取成功。\n\n为满足既定空间门槛，旧中间与诊断检查点已完整校验冷归档到本机，原始数据及最终模型留服务器。归档索引见本项目checkpoint_archive_20261104/manifest.json，可按SHA恢复任何归档点。\n');
console.log(JSON.stringify({reference_count:500,policy_updates:j.training_status.policy_updates,pid:j.training_status.pid,run:j.active_run.run_dir,archive_bytes:j.archive.total_bytes}));
