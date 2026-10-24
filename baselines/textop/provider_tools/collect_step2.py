"""Save authorized Step2 specifications and current blocked prerequisite status."""
import hashlib,html,importlib.util,json,pathlib,shutil,sys
from session import REPO
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
module_path=BASE/'scripts/check_step2_ready.py'
spec=importlib.util.spec_from_file_location('step2_check',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=json.loads((BASE/'configs/paper-step2.json').read_text(encoding='utf-8'))
dest=BASE/'results/step2_20261025';dest.mkdir(parents=True,exist_ok=True)
missing=module.audit(plan,None)
raw_progress=json.loads((BASE/'results/data_preparation_20261025/latest.json').read_text(encoding='utf-8'))
progress_rejection=module.audit(plan,raw_progress)
unbacked={'dataset':{'dof':29,'feature_dim':69,'fps':50.0,'physical_filter_passed':True,'pairing_validated':True,'train_val_source_disjoint':True,'normalization_fit_split':'train','missing_public_sources':0,'train_sequences':1,'val_sequences':1},'runtime':{k:True for k in ['vae69_forward_backward_passed','ldm69_conditioning_and_gradients_passed','fk29_geometry_loss_passed','fresh_document_witness_passed','paper_code_differences_audited']}}
fake_rejection=module.audit({**plan,'launch_enabled':True},unbacked)
assert not missing['ready'] and not progress_rejection['ready'] and not fake_rejection['ready']
status={'authorized_next_step':'VAE -> LDM -> autoregressive generation after Step1 acceptance','training_launched':False,'current_gate':missing,'negative_checks':{'preprocessing_progress_cannot_stand_in_for_acceptance':not progress_rejection['ready'],'unsupported_true_flags_cannot_stand_in_for_evidence':not fake_rejection['ready']},'check_script_sha256':hashlib.sha256(module_path.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((BASE/'configs/paper-step2.json').read_bytes()).hexdigest(),'numerical69D_validation_executed':False}
(dest/'readiness.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
source=BASE/'upstream/TextOpRobotMDAR/robotmdar'
audit={'pinned_source':'ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22','source_files':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['dtype/motion.py','config/data/babel.yaml','config/skeleton/g1.yaml','config/vae/def.yaml','config/denoiser/def.yaml','config/diffusion/def.yaml','train/manager.py']},'released_dof':23,'released_feature_dim':57,'released_qpos_dim':30,'paper_dof':29,'paper_feature_dim':69,'paper_qpos_dim':36,'ldm_rollout_cap_adaptation_pending':True,'sampling_variance_audit_pending':True,'adapted69D_training_ready':False}
(dest/'source_alignment.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
LOCAL.mkdir(parents=True,exist_ok=True)
for src,name in [(BASE/'reports/step2_motion_generator_20261025.md','Step2运动生成器训练安排_20261025.md'),(BASE/'configs/paper-step2.json','paper-step2.json'),(dest/'readiness.json','step2_readiness.json')]:shutil.copy2(src,LOCAL/name)
report=(BASE/'reports/step2_motion_generator_20261025.md').read_text(encoding='utf-8')
page='<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp Step2训练安排</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1000px;margin:36px auto;padding:24px;background:#f6f8fa}pre{white-space:pre-wrap;background:white;border:1px solid #d0d7de;border-radius:8px;padding:24px;font:15px/1.9 system-ui}a{color:#0969da}.status{background:#fff4ce;padding:18px}</style><h1>TextOp Step2：运动生成器训练</h1><p class="status">已记录授权，等待Step1数据验收与69D实现验证；当前未启动训练。</p><p><a href="数据准备核查报告_20261025.html">查看数据准备进度</a> · <a href="https://arxiv.org/pdf/2602.07439#page=14">原文方法和附录</a></p><pre>'+html.escape(report)+'</pre></html>'
(LOCAL/'Step2运动生成器训练安排_20261025.html').write_text(page,encoding='utf-8')
print(json.dumps({'status':plan['status'],'training_launched':False,'negative_checks':status['negative_checks']},ensure_ascii=True))
