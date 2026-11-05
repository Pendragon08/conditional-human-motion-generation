const fs=require('fs'),path=require('path'),cp=require('child_process');
const L=path.resolve(__dirname,'..');
const tasks=['pick','punch','deep_squat','deep_squat_pick','drawer_open','pour','vertical_wipe','step_punch'];
const plan={schema:1,user_requirement:'500 distinct generated references per task for PPO,29DoF',
 source:'explicit user instruction2026-11-04',target_per_task:500,target_all_eight_tasks:4000,
 current_phase:'pick_reference_generation_and_admission',formal_training_enabled:false,
 sequence:['build500Pick references','new500-reference Pick PPO baseline','same500-reference Pick SuperMimic comparison','remaining tasks each500references then training/evaluation'],
 current_pick_legacy:{references:2,ppo_updates:3000,evaluated_environments:100,successes:1,preserve_all_results:true},
 admission:['distinct DDPM noise provenance and generated feature content',
 '20Hz at most200states,29bodyjoints plus binary hands',
 'task target/time and object pose bound per reference',
 'postoptimization all29DoF, limits/feet/root/wrist/scene checks',
 'native scene replay appropriate for each reference object pose',
 'final manifest hash binding, exact content duplicate audit and loader check'],
 notes:['500 means admitted usable references, not500raw candidates or500copies',
 'resample failed candidates with new seeds; preserve rejection reports',
 'do not restart formalPPO/SuperMimic before applicable reference count500',
 'do not reretarget or re-audit sourceGRAB/Omni dataset',
 '500 references cannot by itself prove the cause or correction of low success'],
 pick_sampling:{checkpoint_sha256:'109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9',
    fps:20,frames:120,seed_start:20261104,batch_size:2,pilot_candidates:4,
    object_xy_uniform_range_m:[-.06,.06],grasp_time_s:2,lift_time_s:4,
    pose_scope:'current tabletop baseline, not full paper randomization',
    need_new_scene_adapter_for_per_reference_object_pose:true},
 tasks:tasks.map(task=>({task,target_references:500,new_admitted:0,status:task==='pick'?'preparing_batched_sampling':'pending_task_adaptation'}))};
const out=path.join(L,'reports/stage3_reference500_20261104');fs.mkdirSync(out,{recursive:true});
fs.writeFileSync(path.join(out,'plan.json'),JSON.stringify(plan,null,2)+'\n');
const note=`## 2026-11-04最新用户要求：每项技能500条生成参考后再进行PPO

- 用户明确每个验证子任务需500条生成动作用于PPO。执行范围仍为8项技能，目标共4000条通过任务适用检查的不同参考；500原始候选不等于500已接纳参考，不复制文件凑数。保留既有两条参考的3000更新PPO及1/100评估，另建500参考Pick基线，再在同一500参考上做SuperMimic对照。先扩充Pick，不在两条旧参考上启动正式SuperMimic；后续任务先各自500条再训练。现有2条只属小规模基线，不能说论文规模复现已完成，也不能将1%全部归因参考条数。
- 已实现隔离分批采样入口stage3_sample_pick29_batch.py：复用最终10000步扩散先验，每条独立初始噪声seed/hash及生成特征内容hash，20Hz/120状态，按当前桌面XY+-0.06m变化接触目标；每批落盘清单与进度，受磁盘/5%GPU保护。先4条真实采样验证，再按实测扩大；后优化与原生场景回放后计入500接纳数，失败补采样。扩充对象位置必须同步后优化目标、原生场景和训练重置，不能沿用旧固定物体位置使参考失配。
- 500参考计划及当前计数见本地reports/stage3_reference500_20261104/plan.json和服务器stage3_pick29_20261103/reference_scale500_20261104/plan.json。新500参考训练与旧2参考结果严格分开目录/清单，不覆盖检查点。29DoF、31策略动作、102实时无参考Actor、20Hz参考、50Hz控制及每次Pick3000×24真实交互预算保持；继续执行共享GPU/磁盘保护、只做仿真、Nymeria暂停。11GiB仅正式训练启动门槛，不阻碍受限参考采样构建。

`;
for(const name of ['PLAN.md','STATUS.md']){
 const file=path.join(L,name);let x=fs.readFileSync(file,'utf8');
 if(!x.includes('## 2026-11-04最新用户要求：每项技能500条')){let p=x.indexOf('\n');fs.writeFileSync(file,x.slice(0,p+1)+'\n'+note+x.slice(p+1));}
}
const py=`import json,os\nfrom pathlib import Path\nR=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')\nS=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103'\nplan=json.loads(${JSON.stringify(JSON.stringify(plan))})\nf=S/'reference_scale500_20261104';f.mkdir(exist_ok=True)\n(f/'plan.json').write_text(json.dumps(plan,indent=2)+'\\n')\nstatus=json.loads((D/'adaptation_status.json').read_text())\nstatus.update(phase='pick_reference500_expansion_before_further_formal_training',reference_policy=str(f/'plan.json'),reference_target_per_task=500,formal_supermimic_started=False,training_ready=False)\nstatus['online_loop_author_passed']=(D/'reports/online_author_v1/report.json').exists()\nstatus['online_loop_doc_passed']=(D/'reports/online_doc_v1/report.json').exists()\n(D/'adaptation_status.json').write_text(json.dumps(status,indent=2)+'\\n')\nfor name in ['sequence.json','progress.json']:\n p=Q/name;backup=Q/(name+'.before_reference500_20261104')\n if not backup.exists():backup.write_bytes(p.read_bytes())\n data=json.loads(p.read_text());data['reference_policy_20261104']=str(f/'plan.json');data['next_phase']='pick_reference500_expansion'\n if name=='progress.json':data['status']='reference_expansion_requested_preserve_completed_2ref_baseline'\n else:data['minimum_generated_references_per_task']=500\n tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)\nprint(json.dumps({'recorded':str(f),'target_per_task':500,'total_target':4000,'formal_training_started':False}))\n`;
console.log(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8'}));
