const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const L=path.resolve(__dirname,'..');
const script=String.raw`import json,time
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104';Q=R/'stage3_skill_sequence_20261103'
def atomic(p,j):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2));t.replace(p)
progress=json.loads((O/'library_progress.json').read_text())
plan=json.loads((O/'plan.json').read_text())
plan['latest_user_instruction']='2026-11-04: eight skills each500 admitted references; task-specific PPO; start new Pick PPO immediately after its500-reference library is verified, without waiting for other tasks; provide generated-reference videos'
plan['tasks'][0]['new_admitted']=progress['native_admitted']
plan['ppo_policy']='one independent skill policy per task, sharing that task\'s500 references; not one PPO per reference'
plan['pick_ppo_total_updates']=3000
plan['pick_start_dependency']='Pick500 final manifest, loader/environment/PPO checks and existing resource protections; other seven libraries are not dependencies'
plan['supermimic_comparison']='after new Pick PPO evaluation, reuse the identical500-reference bank; not an additional ninth skill or500 new references'
atomic(O/'plan.json',plan)
sequence=json.loads((Q/'sequence.json').read_text())
sequence['policy'].update(primary_skill_training_backend='task-specific PPO',preferred_backend_after_pick_evaluation='same-library Pick SuperMimic comparison; keep task-specific PPO as the requested primary skill training',wait_for_all4000_before_pick_ppo=False)
sequence['generated_reference_target_per_skill']=500;sequence['generated_reference_target_all_skills']=4000
for item in sequence['tasks']:
 if item['id']=='pick_supermimic':
  item['reference_reuse_task']='pick';item['new_reference_target']=0
  item['after_completion']='continue remaining seven skill-specific PPO training/evaluation'
 else:
  item['generated_reference_target']=500;item['primary_training_backend']='PPO';item['independent_skill_policy']=True
  if item['id']=='pick':
   item['legacy_two_reference_baseline']={k:item[k] for k in ['run_dir','reference_bank','reference_bank_sha256','iteration_target','active_run_metadata'] if k in item}
   item['implementation']='legacy2-reference PPO completed and evaluated; new500-reference library building'
   item['new_library_progress']=str(O/'library_progress.json');item['new_ppo_iteration_target']=3000
sequence['latest_user_instruction']=plan['latest_user_instruction']
atomic(Q/'sequence.json',sequence)
print(json.dumps({'plan':plan,'sequence':sequence,'progress':progress},ensure_ascii=False))
`;
const j=JSON.parse(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
const dest=path.join(L,'reports/stage3_reference500_20261104');fs.mkdirSync(dest,{recursive:true});
for(const [n,v] of Object.entries(j))fs.writeFileSync(path.join(dest,'task_plan_'+n+'.json'),JSON.stringify(v,null,2));
const tasks=[['Pick','物体抓取'],['Punch','出拳'],['Deep Squat','深蹲'],['Deep Squat & Pick','深蹲抓取'],['Drawer Open','开抽屉'],['Pour','倾倒'],['Vertical Wipe','竖直擦拭'],['Step & Punch','迈步出拳']];
const guide='# 任务生成与策略训练计划\n\n每个任务500条通过任务场景适用检查、去重和加载验证的29DoF参考，8项共4000条。参考统一20Hz、最长10秒，保留29身体关节及双手开合状态；当前Pick为120状态、约6秒。500是最终入库数量，失败候选不计，必要时用新seed补采样。\n\n| 任务 | 中文名称 | 参考目标 | PPO策略 |\n|---|---|---:|---|\n'+tasks.map(x=>`| ${x[0]} | ${x[1]} | 500 | 独立训练一套 |`).join('\n')+'\n\n执行顺序：Pick500完整参考库→新Pick PPO总3000更新→100环境物理仿真评估→同500库SuperMimic对照→其余任务。Pick不等待全部4000条；一个任务的500条共同参与其策略训练，不是500次独立PPO。SuperMimic复用Pick库，不计第九个任务。其余七个任务的29DoF场景和生成条件尚待适配，不能将计划数量说成已完成。\n\n视频为真实生成并通过场景检查的Pick参考在标定Isaac机器人上的姿态回放。它不等于训练后的PPO执行，也不作为物体抓取成功证据；对象保持初始位置，不伪造被抓起的物体运动。\n';
fs.writeFileSync(path.join(L,'reports/STAGE3_TASK500_AND_VIDEOS.md'),guide);
const title='## 2026-11-04最新确认：八任务各500；Pick库完成后优先独立PPO，提供参考视频';
const note=title+'\n\n- 用户再次明确每任务独立PPO：Pick、Punch、Deep Squat、Deep Squat & Pick、Drawer Open、Pour、Vertical Wipe、Step & Punch各500条最终接纳参考，共4000。当前Pick已接纳'+j.progress.native_admitted+'条，动态进度以服务器library_progress.json为准；其余七项尚待任务适配与生成。SuperMimic同库对照不计第九项，不另生成500。\n- Pick500最终清单、加载/环境/PPO检查与既定资源保护满足后，优先在新目录训练Pick总3000更新，再100环境评估，不等其余3500条。旧两参考PPO结果/检查点全部保留；其余任务以按任务PPO为主，SuperMimic保留为已授权同库对照，不把原生迭代数与PPO更新数直接等同。\n- 报告视频单独受限回放作业report_video_v1，先查launch/guard/log，不重复启动。只从已接纳库选3条不同目标位置，使用标定29DoF带手Isaac资产，20Hz双视图，明确参考姿态回放非策略执行/抓取成功；对象固定初始位置，不伪造携物轨迹。\n\n';
for(const name of ['PLAN.md','STATUS.md']){const f=path.join(L,name),old=fs.readFileSync(f,'utf8');if(!old.includes(title)){const i=old.indexOf('\n');fs.writeFileSync(f,old.slice(0,i+1)+'\n'+note+old.slice(i+1));}}
console.log(JSON.stringify({admitted:j.progress.native_admitted,total_target:4000,guide:path.join(L,'reports/STAGE3_TASK500_AND_VIDEOS.md')}));
