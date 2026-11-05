const cp=require('child_process'),fs=require('fs'),path=require('path');
const L=path.resolve(__dirname,'..');
const py=`import json,pickle,csv,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');W=R/'reactive29_pairing_work_20261104'
P=Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl')
H=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public')
with P.open('rb') as f:data=pickle.load(f)['data_dict']
all_ids=set((H/'HumanML3D/all.txt').read_text().splitlines());base={s.removeprefix('M') for s in all_ids}
covered=set(data);uncovered=sorted(base-covered)
with (H/'index.csv').open(newline='') as f:index={Path(r['new_name']).stem:r for r in csv.DictReader(f)}
coverage={'scope':'nominalID coverage only, no recording identity inference','humanml_ids_with_mirrors':len(all_ids),
 'humanml_base_ids':len(base),'available_reactive29_ids':len(covered),'covered_base_ids':len(base&covered),
 'uncovered_base_ids':uncovered,'uncovered_index_records':{k:index.get(k) for k in uncovered},
 'mirrored_augmentation_not_added':True,'current_action':'reuse11424 covered G1 sequences; classify uncovered raw sources before any new retargeting',
 'current_gpu_trajectory_generation_unchanged':True}
(W/'nominal_coverage.json').write_text(json.dumps(coverage,indent=2))
paths={'launch':W/'publisher60_20hz_full_v1_launch.json','build_status':W/'publisher60_20hz_full_v1/status.json',
 'build_report':W/'publisher60_20hz_full_v1/report.json','audit':W/'independent_audit_v1.json','audit_exit':W/'automatic_audit_exit.json',
 'coverage':W/'nominal_coverage.json','loader':W/'loader_v1/report.json','uncovered_triage':W/'uncovered_source_triage.json'}
out={k:{'data':json.loads(p.read_text()),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items() if p.exists()}
complete=('audit' in out and out['audit']['data']['status']=='paired_profile_file_audit_passed' and 'loader' in out and out['loader']['data']['status']=='combined_stats_and_three_split_loader_passed')
plan={'user_requested_parallel_AMASS_processing':True,'existing_G1_reuse_count':11424,
 'active_cpu_wrapper_pid':None if complete else 1408020,'build_phase':'nominal_profile_completed' if complete else 'converting_and_auditing','source_contract_git_blob':'04ebf854bf9c6deaac9623509c4b49c6794c7041',
 'profile':'publisher_nominal60_to20Hz_29DoF_weak_text','CPU_threads':1,'GPU_used':False,
 'new_dataset_root':str(W/'publisher60_20hz_full_v1'),'coverage_manifest':str(W/'nominal_coverage.json'),
 'new_training_started':False,'generation_PIDs_unchanged':[1402368,1405357],
 'next_actions':['classify only uncovered HumanML sourceIDs before further GMR/PyRoki work'] if complete else ['complete and audit11424 profile','classify only uncovered HumanML sourceIDs before further GMR/PyRoki work'],
 'raw_data_and_other_projects_read_only':True}
(W/'parallel_plan.json').write_text(json.dumps(plan,indent=2))
out['plan']={'data':plan,'path':str(W/'parallel_plan.json'),'sha256':hashlib.sha256((W/'parallel_plan.json').read_bytes()).hexdigest()}
if complete:
 receipt={'status':'nominal_publisher60_motion_text_profile_completed','trajectories':11424,'published_captions':34132,
  'body_dof':29,'target_fps':20,'point_count':22,'features':263,'maximum_seconds':10,
  'source_fps_basis':'publisher60Hz declaration, not independent physical-clock verification',
  'manifest_path':str(W/'publisher60_20hz_full_v1/manifest.json'),'manifest_sha256':out['audit']['data']['manifest_sha256'],
  'build_report_sha256':out['build_report']['sha256'],'independent_audit_sha256':out['audit']['sha256'],
  'loader_report_sha256':out['loader']['sha256'],'combined_stats_path':str(W/'loader_v1/combined_stats.npz'),
  'combined_stats_sha256':out['loader']['data']['combined_stats_sha256'],'actual_recording_clock_verified':False,
  'raw_and_legacy_models_preserved':True,'new_model_training_started':False}
 p=W/'completion_receipt.json'
 if not p.exists():p.open('x').write(json.dumps(receipt,indent=2))
 out['completion']={'data':json.loads(p.read_text()),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(out))
`;
const d=JSON.parse(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python -'],{input:py,encoding:'utf8',maxBuffer:5e6}));
const folder=path.join(L,'reports/reactive29_pairing_parallel_20261104');fs.mkdirSync(folder,{recursive:true});
for(const [name,item]of Object.entries(d))fs.writeFileSync(path.join(folder,name+'.json'),JSON.stringify(item.data,null,2)+'\n');
fs.writeFileSync(path.join(folder,'evidence_locations.json'),JSON.stringify(Object.fromEntries(Object.entries(d).map(([k,v])=>[k,{path:v.path,sha256:v.sha256}])),null,2)+'\n');
const note=`## 2026-11-04最新要求：生成参考继续，同时并行处理AMASS/HumanML3D

- 用户明确参考生成保持，AMASS/HumanML3D文字—29DoF G1参考数据并行推进。沿用用户此前指定的ReactiveBFM公开11424条，不重复对已覆盖内容人体重定向；未覆盖名义ID另列coverage，后续仅对确实需要的原始源确认差集/时间后再GMR+JAX/PyRoki。当前500任务参考采样PID1402368及后优化PID1405357保持，已训练扩散先验不更改。
- 已核作者当前数据README Git blob04ebf854bf9c6deaac9623509c4b49c6794c7041：根xyz/xyzw+29关节、Z-up、所有g1_36dim声明60Hz。CPU低优先级一线程包装PID1408020独立构建publisher_nominal60_to20Hz_29DoF_weak_text，目录reactive29_pairing_work_20261104/publisher60_20hz_full_v1；最多10秒、22点/263D、完整29状态、发布原文全部保留，并记录HumanML原文/来源成员哈希及文字差异。60Hz仅发布说明的名义时间依据，不冒称逐条真实录制时钟；历史20Hz敏感性试验仍保留。未新增人工语义/物理筛选，不启动新扩散训练，不改变当前3161数据/模型。
- 全11424条各自转换后读回，按确切索引源路径进行test>val>train分组、保留官方与发布方切分字段、仅训练侧重算统计；不能把路径/ID当真实录制身份。输出预算1GiB、磁盘8GiB+456MiB底线，CUDA隐藏不占GPU。publisher60_20hz_full_v1/status.json、同名.log和launch收据为准，运行不重复；完成后watch_reactive29_pairing_audit_20261104.py自动独立核所有文件/来源/文字/20Hz网格/统计及32条同编码器复算，检查independent_audit_v1.json和automatic_audit_exit.json，失败保留实际前缀新目录续。
- 本轮论文核对：v2附录V-E/TableIX明确共用运动跟踪奖励并加任务项；Pick物体高度>0.95m，Deep Squat & Pick>0.65m，Drawer位移>=0.05m等；Deep Squat/BimanualPick额外约束根平面线/角速度。v2称RL沿用DreamControl，原公开README以--task单独训练/独立任务检查点，支持按任务训练技能策略；v2没有公开统一8任务Actor联合训练方案。共享PPO算法/MLP结构不代表共享同一策略权重，也不是500条各训一次。

`;
for(const name of ['PLAN.md','STATUS.md']){const p=path.join(L,name);let x=fs.readFileSync(p,'utf8');if(!x.includes('## 2026-11-04最新要求：生成参考继续，同时并行处理AMASS/HumanML3D')){let i=x.indexOf('\n');fs.writeFileSync(p,x.slice(0,i+1)+'\n'+note+x.slice(i+1));}}
if(d.completion){const n=`## 2026-11-04：ReactiveBFM11424条名义60→20Hz配对集已完成及独立技术验收

- 复用11424条现成29DoF轨迹与34132句原文，全部完成20Hz/最长10秒/22点263D及完整29状态转换，0失败；665378548字节运动文件，CPU一线程约305.5秒。每条文件/来源运动/文字绑定/20Hz网格/四元数/形状/统计重读通过，32条同采样器/FK/编码器复算最大差0；确切索引源组跨划分重叠0，精确特征跨划分重复0。按源路径分组保留全部轨迹，训练8146/验证738/测试2540，保留作者与HumanML原划分字段；仅训练侧264224特征帧重算统计。
- 增补时空条件统计与真正CPU数据加载器，三个划分分别4条实际数据加载/有效帧掩码/完整29状态/文字/263D/22点条件归一化通过。combined_stats_sha256为7f2dceda1e94c8649d1cf7aab48c8a384163dae6bab07db824fba8e4e9b692c1；主清单SHA256 8ee660fcc10c01a0f42225a091cc94f9df9bc9b37f337b51831f5f1d33f25538。数据目录reactive29_pairing_work_20261104/publisher60_20hz_full_v1，统计loader_v1/combined_stats.npz，最终收据completion_receipt.json。原构建report中的independent_file_audit=pending为原始时点，实际后续独立审计及加载已通过；不回头重跑或覆盖原报告。
- 这是明确采用发布方60Hz声明转换的名义时间版本，不把历史20Hz敏感性差异掩盖或声称逐条真时钟已核；采用发布原文弱监督，294条文字非逐字相同仍原样保留且标注，未新增逐条语义/物理筛选。未合并或重训正在生成参考所用模型，500任务参考两条GPU流水线保持。此11k既有覆盖无需再重定向。
- HumanML非镜像14614名义ID中尚有3190不在此发布集：1191属于单列HumanAct12非AMASS源、1748需现存归档/源解析、251在服务器找到名义原始候选路径；只是可用性/名字线索，不证明身份或时间，差集和逐条来源uncovered_source_triage.json保留。后续优先对确实未覆盖且来源/时间可用者做GMR+JAX/PyRoki，不重复已处理11k、不启动旧GRAB源重定向。

`;for(const name of ['PLAN.md','STATUS.md']){const p=path.join(L,name);let x=fs.readFileSync(p,'utf8');if(!x.includes('## 2026-11-04：ReactiveBFM11424条名义60→20Hz配对集已完成')){const i=x.indexOf('\n');fs.writeFileSync(p,x.slice(0,i+1)+'\n'+n+x.slice(i+1));}}}
console.log(JSON.stringify({build:d.build_status?.data,audit:d.audit?.data?.status,coverage:{available:d.coverage.data.available_reactive29_ids,uncovered:d.coverage.data.uncovered_base_ids.length},local:folder}));
