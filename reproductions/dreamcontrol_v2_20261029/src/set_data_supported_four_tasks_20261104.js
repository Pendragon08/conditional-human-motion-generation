const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const L=path.resolve(__dirname,'..');
const script=String.raw`import json,fcntl,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');Q=R/'stage3_skill_sequence_20261103';O=R/'stage3_pick29_20261103/reference_scale500_20261104'
def atomic(p,j):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2));t.replace(p)
def backup(p,name):
 q=p.parent/name
 if not q.exists():q.open('x').write(p.read_text())
lock=(Q/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
active=json.loads((Q/'active_controller.json').read_text());proc=Path('/proc')/str(active['pid'])/'stat'
assert not proc.exists() or proc.read_text().split()[2]=='Z','old skill controller still active'
p=Q/'sequence.json';backup(p,'sequence_before_supported_four_20261104.json');j=json.loads(p.read_text())
old={r['id']:r for r in j['tasks']};deferred=[r for r in j['tasks'] if r['id'] not in ['pick','pour','pick_supermimic']]
for r in deferred:r['execution_status']='deferred_by_user_missing_current_data_support'
pick=old['pick'];pick.update(status='building500_reference_library',generated_reference_target=500,primary_training_backend='PPO',iteration_target=3000)
pour=old['pour'];pour.update(status='task_adapter_and_generated_references_pending',generated_reference_target=500,primary_training_backend='PPO')
box={'id':'bimanual_box_lift','name':'Bimanual Box Lift & Place','label_zh':'双手搬箱','implementation':'29DoF two-wrist box scene, reference generation and task reward pending',
 'status':'task_adapter_and_generated_references_pending','generated_reference_target':500,'primary_training_backend':'PPO','independent_skill_policy':True,
 'training_support':'OmniRetarget/OMOMO Lift/move/put-down largebox descriptions','requires':['two-wrist target and grasp timing','real two-hand box contact scene','task-specific lift/move/place success metrics'],
 'no_pick_relabel':'Do not rename the single-right-wrist Pick scene/reference library as bimanual box lifting'}
push={'id':'box_push_pull','name':'Box Push & Pull','label_zh':'推拉箱','implementation':'29DoF box push-release-pull scene, reference generation and task reward pending',
 'status':'task_adapter_and_generated_references_pending','generated_reference_target':500,'primary_training_backend':'PPO','independent_skill_policy':True,
 'training_support':'OmniRetarget/OMOMO Push/release/pull largebox descriptions','task_unit':'one complete push-and-pull task,500 references and one PPO skill policy',
 'requires':['matched box pose/displacement targets','push/pull hand contact schedule','task-specific actual box displacement and return metrics'],
 'no_pick_relabel':'Do not reuse single-hand lift targets and call them push/pull'}
j.update(version=5,tasks=[pick,pour,box,push],deferred_tasks=deferred,active_task_order=['pick','pour','bimanual_box_lift','box_push_pull'],
 generated_reference_target_per_skill=500,generated_reference_target_all_skills=2000,
 scope='User2026-11-04: current data-supported four-task29DoF simulation baseline; unsupported paper tasks deferred',
 latest_user_instruction='Pick,Pour,bimanual box lifting,box push-pull each500 references, task-specific PPO. Finish Pick500 and train its policy first. Defer unsupported paper tasks.')
comparison=old.get('pick_supermimic')
if comparison:
 comparison.update(kind='separate_same_Pick500_backend_comparison',counted_as_skill=False,new_reference_target=0,reference_reuse_task='pick',after_completion='continue active data-supported four-task PPO plan')
 j['comparisons']=[comparison]
j['policy'].update(primary_skill_training_backend='PPO',train_tasks_sequentially=True,wait_for_all2000_before_pick_ppo=False,
 deferred_missing_data_paper_tasks=True,active_reference_target_total=2000,
 task_distribution_gate='Use present GRAB/Omni motion support, validate task-specific generation and scene before scaling; no implicit support for deferred paper tasks',
 current_model_task_scope='Pick first;Pour,bimanual_box_lift,box_push_pull follow, each with its own scene/reference/reward/success criteria',
 preferred_backend_after_pick_evaluation='preserve separately authorized same-library SuperMimic comparison; each active skill primary training remains PPO')
atomic(p,j)
pp=Q/'progress.json';backup(pp,'progress_before_supported_four_20261104.json');pr=json.loads(pp.read_text())
baseline=next(r for r in pr['tasks'] if r['id']=='pick');snapshot=json.loads((O/'library_progress.json').read_text())
pr.update(current_index=0,current_task='pick',status='building_data_supported_Pick500_before_new_PPO',
 tasks=[{'id':'pick','status':'reference_library_building','admitted_references':snapshot['native_admitted'],'target_references':500,'new_PPO_started':False},
        {'id':'pour','status':'pending_task_adaptation','target_references':500},
        {'id':'bimanual_box_lift','status':'pending_task_adaptation','target_references':500},
        {'id':'box_push_pull','status':'pending_task_adaptation','target_references':500}],
 legacy_two_reference_Pick_baseline=baseline,deferred_tasks=[r['id'] for r in deferred],
 next_action='Finish Pick500 with independent source/target/native/hash/loader checks; new Pick PPO3000 updates/8912env and100env evaluation first, not all2000 dependency',
 controller_pid=None,old_controller_status='normally exited, do not restart the old8-skill queue',
 reference_progress_source=str(O/'library_progress.json'),target_references_all_active_tasks=2000,
 next_phase='pick_reference500_expansion')
atomic(pp,pr)
planp=O/'plan.json';backup(planp,'plan_before_supported_four_20261104.json');plan=json.loads(planp.read_text())
plan.update(target_all_active_tasks=2000,active_tasks=['pick','pour','bimanual_box_lift','box_push_pull'],
 target_all_eight_tasks=4000,eight_task_target_status='historical broader paper scope, unsupported tasks deferred by user',
 current_phase='pick_reference_generation_and_admission',
 sequence=['Pick500 -> newPick PPO3000 ->100env evaluation','separate samePick500 SuperMimic comparison retained','Pour500 -> task PPO/evaluation','BimanualBoxLift500 -> task PPO/evaluation','BoxPushPull500 -> task PPO/evaluation'],
 latest_user_instruction=j['latest_user_instruction'])
plan['tasks']=[{'task':r['id'],'target_references':500,'new_admitted':snapshot['native_admitted'] if r['id']=='pick' else 0,
 'status':'sampling_postoptimization_and_guarded_native_admission' if r['id']=='pick' else 'pending_task_adaptation'} for r in j['tasks']]
atomic(planp,plan)
print(json.dumps({'sequence':j,'progress':pr,'plan':plan,'snapshot':snapshot},ensure_ascii=False))
`;
const j=JSON.parse(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
const D=path.join(L,'reports/data_supported_four_20261104');fs.mkdirSync(D,{recursive:true});for(const[k,v]of Object.entries(j))fs.writeFileSync(path.join(D,k+'.json'),JSON.stringify(v,null,2));
const note='## 2026-11-04最新授权：近期改为四项数据支持任务，每项500，Pick优先\n\n- 当前执行任务改为Pick、Pour、双手搬箱、推拉箱，各500条最终接纳参考，合计2000；推拉箱作为完整一项任务500条、一套PPO，不拆成各500的两项。G1保持29DoF，20Hz/最长10秒；每任务独立PPO和100环境物理仿真评估。先完成Pick500并训练其总3000更新，不等待其他1500。旧八任务/4000为历史扩大范围：Punch、Step&Punch、DeepSquat、DeepSquat&Pick、DrawerOpen、VerticalWipe按用户要求暂缓，保留原计划和所有结果。\n- Q/sequence.json、progress.json与O/plan.json已在旧控制器退出且取得队列锁后更新，旧版本以before_supported_four_20261104命名备份。实际Pick接纳以O/library_progress.json为准，当前快照'+j.snapshot.native_admitted+'/500；未重复或中断原采样/求解/原生批次。SuperMimic已授权同库对照单列，不计第五个技能/新增500，也不替代四项PPO。\n- Pour、双手搬箱、推拉箱仍需各自29DoF场景、双手/腕姿态条件、物体接触与真实成功标准、小批生成/仿真/PPO验证，之后再各扩到500。不得只改Pick名称或prompt就冒充搬箱推拉；使用当前固定扩散先验及对应数据动作支持，不把论文未实现任务标ready。\n\n';
for(const name of ['PLAN.md','STATUS.md']){const p=path.join(L,name),s=fs.readFileSync(p,'utf8');if(!s.includes(note.split('\n')[0])){let i=s.indexOf('\n');fs.writeFileSync(p,s.slice(0,i+1)+'\n'+note+s.slice(i+1));}}
const md='# 当前数据支持的四任务计划\n\n| 顺序 | 任务 | 参考目标 | 策略 |\n|---|---|---:|---|\n| 1 | Pick物体抓取 | 500 | 独立PPO，先训练 |\n| 2 | Pour倾倒 | 500 | 独立PPO |\n| 3 | 双手搬箱 | 500 | 独立PPO |\n| 4 | 推拉箱 | 500 | 一项完整推拉任务，独立PPO |\n\n近期合计2000条最终接纳参考。G1身体29自由度，双手二值开合；参考20Hz，最长10秒。先完成Pick500并训练总3000更新、100环境评估，不等待其余三项。其他任务尚待独立场景和生成验证，未宣称就绪。缺少数据支持的论文任务暂缓；原八项计划和全部既有结果保留。\n';
fs.writeFileSync(path.join(D,'任务计划.md'),md);console.log(JSON.stringify({active_tasks:4,total_references:2000,pick_admitted:j.snapshot.native_admitted,deferred:j.sequence.deferred_tasks.map(r=>r.id)}));
