from pathlib import Path
import argparse,json,hashlib
p=argparse.ArgumentParser()
p.add_argument('--phygile-upstream',type=Path,required=True)
p.add_argument('--reactive-upstream',type=Path,required=True)
p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1])
a=p.parse_args();PUP=a.phygile_upstream;RUP=a.reactive_upstream;BASE=a.output_root
for d in ['sources','configs']:(BASE/d).mkdir(parents=True,exist_ok=True)
def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,obj):Path(path).write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

import yaml,copy,shutil

root=PUP/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg'
audit=[]
for phase in ['a','b']:
    source=root/f'g1_training_moe_phase_{phase}.yaml'
    original=yaml.safe_load(source.read_text(encoding='utf-8'))
    cfg=copy.deepcopy(original)
    target=BASE/'sources'/source.name
    target.write_bytes(source.read_bytes())
    changes=[]
    def put(parts,value,evidence):
        current=cfg
        for p in parts[:-1]:current=current.setdefault(p,{})
        before=current.get(parts[-1]);current[parts[-1]]=value
        changes.append({'key':'.'.join(parts),'official_example':before,'paper_profile':value,'evidence':evidence})
    put(['policy','moe_config','expert_hidden_dims'],[2048,1024,512],'PhyGile v3 PDF p11 section 7a')
    for key,value in [('file_freeze_min_attempts',20000),('file_freeze_iters',4000),('file_freeze_check_interval',500)]:
        put(['commands','motion',key],value,'PhyGile v3 PDF p11 Table IX')
    for key,value in [('num_steps_per_env',24),('max_iterations',200000),('save_interval',1000)]:
        put(['runner',key],value,'PhyGile v3 PDF p11 Table XI; per-profile cap, stage allocation unspecified')
    put(['commands','motion','preload_device'],'cuda:0','Execution placement only; replace with allocated device')
    put(['commands','motion','curriculum_csv_path'],'REQUIRED_VERIFIED_DIFFICULTY_CSV','Author DeepSeek difficulty labels unavailable; no fabricated labels')
    put(['commands','motion','curriculum_mocap_dir'],'REQUIRED_VERIFIED_50HZ_MOTION_DIR','Full public corpus and private 3h unavailable')
    if phase=='a':
        put(['commands','motion','curriculum_resume_level_override'],1,'PhyGile v3 PDF p4: start Stage I at level1, not example continuation level7')
    (BASE/'configs'/f'phygile_paper_phase_{phase}.yaml').write_bytes(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True).encode('utf-8'))
    audit.append({'phase':phase,'upstream_sha256':sha(source.read_bytes()),'changes':changes})

dump(BASE/'configs/phygile_parameter_audit.json',{
 'status':'CONFIGURED_NOT_FULLY_TRAINED','official_commit':'d48621620f522f5c901ef378d57db8df92d2888d',
 'paper':'https://arxiv.org/pdf/2603.19305v3','changes':audit,
 'launch_arguments':{'num_envs':8192,'headless':True,'seed':42},
 'seed_source':'reproduction choice; not specified by paper',
 'environment':{'IsaacSim':'4.5.0','IsaacLab':'2.1.0','dt':.005,'decimation':4,'episode_seconds':10,'spacing_m':2.5},
 'must_verify_resolved_environment_before_launch':True,
 'full_run_blockers':['Full45h corpus includes unavailable private3h','Author difficulty CSV unavailable','Previously trained expert512x256 checkpoints incompatible with paper2048x1024x512 experts','8192 environments not yet resource-validated on currently shared GPUs'],
 'generator_missing_author_details':['optimizer and learning-rate schedule','training batch size and steps','exact codec frame/difference conventions','denoising schedule','ASFO rarity cap and probability alpha','closed-loop MPJPE tolerance and PPO update budget','robot-space generation evaluator weights/splits'],
 'do_not_infer':'Unspecified settings must remain reconstruction choices, never paper facts'})

dump(BASE/'configs/reactive_paper_vs_code.json',{
 'paper':'https://arxiv.org/pdf/2606.30362','official_commit':'5a79d03cab824eb5917271ba720294e8a1b9d374',
 'preserve_existing_training':True,
 'planner_paper_table4':{'optimizer':'Adam','batch_size':128,'learning_rate':1e-4,'iterations':2500,'environments':32,'n_steps':512,'n_epochs':4,'buffer_size':30,'value_coefficient':.8,'entropy_coefficient':.01,'gamma':.99,'GAE':.99,'ppo_clip':.2},
 'paper_table4_inconsistency':'The main text and public planner use supervised flow matching. These PPO-style table fields do not map to the published planner trainer; not silently used as planner PPO instructions.',
 'official_training_recipe':{'global_batch':1024,'teacher_forcing_updates':400000,'total_updates':1000000,'cross_prob_text_only':0.0,'cross_prob_streaming':0.1,'model_type':'flow'},
 'running_training':{'ranks':2,'batch_per_rank':512,'global_batch':1024,'learning_rate':1e-4,'teacher_forcing_updates':400000,'total_updates':1000000,'precision':'FP32 repair','context_frames':25,'prediction_frames':50},
 'paper_deployment':{'context_frames':20,'prediction_frames':40,'policy_hz':50,'pd_hz':200,'buffer_trigger_frames':10,'asynchronous':True},
 'controller_paper_table5':{'actor_learning_rate':2e-5,'critic_learning_rate':1e-3,'num_envs':8192,'minibatches':16,'epochs':2,'rollout_steps':64,'value_coefficient':1.,'entropy_coefficient':.005,'gamma':.99,'GAE':.95,'ppo_clip':.2,'sampling_discount':.999},
 'controller_reproduction':'Reuse frozen official ScaleBFM M model22200; no downstream RL fine-tuning, as paper Appendix B1. Table5 describes upstream BFM pretraining, not a required new ReactiveBFM PPO stage.',
 'unresolved':['Paper20/40 versus released/running25/50 window','Paper37.14h includes PhysHSI-Reach not in installed3components','Paper1000-pair100STYLE benchmark/evaluator not publicly installed','TensorRT latency not reproduced; shared A6000 native/Inductor compatibility run'],
 'completion_requirements':['training final valid checkpoint','scheduled forcing phase genuinely executed','fresh generated motion reaches controller in asynchronous loop','paper failure metrics from actual simulator states','5 or more fixed seeds and specified held-out split','video provenance and source/checkpoint hashes']})
for source in [PUP/'LICENSE',RUP/'LICENSE']:
 if source.exists():shutil.copy2(source,BASE/'sources'/('phygile_LICENSE' if source.parent==PUP else 'reactivebfm_LICENSE'))
dump(BASE/'sources/upstream_pins.json',{'phygile':{'url':'https://github.com/Baojch/phygile_tracking','commit':'d48621620f522f5c901ef378d57db8df92d2888d','released_scope':'tracking'},'reactivebfm':{'url':'https://github.com/zjwzcx/ReactiveBFM','commit':'5a79d03cab824eb5917271ba720294e8a1b9d374','released_scope':'planner training and deployment with frozen ScaleBFM'}})
print(json.dumps({'configs':len(list((BASE/'configs').glob('*'))),'sha256':{p.name:sha(p.read_bytes()) for p in (BASE/'configs').glob('*')}}))
