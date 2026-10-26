"""Accept the bounded engineering chain; keep semantic/paper quality unaccepted."""
import base64,hashlib,json,pathlib,shutil
import numpy as np
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];T=B.parents[3]/'textop_reproduction_20261024'
# Locate task explicitly rather than relying on a different depth convention.
T=pathlib.Path.cwd()/'work/textop_reproduction_20261024'
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
docpath=T/'online_document_v1/SUMMARY.json';doc=json.loads(docpath.read_text(encoding='utf-8'))
assert doc['all_passed'] and doc['check_count']==doc['passed_count']==430
assert doc['outer_exit_code']==doc['generator_exit_code']==doc['sdk_exit_code']==0
assert doc['deadline_misses_recomputed']==0 and not doc['physical_failure'] and not doc['stream_underruns']
author=json.loads((B/'results/online_quick_20261026/online_author_v11/summary.json').read_text(encoding='utf-8'))
assert author['private_backup_verified'] and author['all_deadlines_passed']
private=D/'datasets/generated_reference_private_20261026/generated_author_v1/generated.private.npz'
with np.load(private,allow_pickle=False) as z:
    q=z['qpos_29'];emb=z['text_embeddings']
    review={'status':'QUICK_MODEL_QUALITY_REVIEWED_NOT_PAPER_ACCEPTED','input_sha256':hashlib.sha256(private.read_bytes()).hexdigest(),
            'walk_phase_root_displacement_xyz_m':(q[129,:3]-q[1,:3]).tolist(),
            'wave_phase_root_displacement_xyz_m':(q[-1,:3]-q[129,:3]).tolist(),
            'text_embedding_max_abs_difference':float(np.max(np.abs(emb[0]-emb[1]))),
            'joint_dynamic_std_mean_rad':float(np.mean(np.std(q[:,7:],axis=0))),
            'fixed_sampling_huber_before':0.31026655808091164,'fixed_sampling_huber_after':0.6075362637639046,
            'semantic_quality_accepted':False,'paper_quality_accepted':False,
            'conclusion':'The bounded functional pipeline is exercised, but quick motion/text quality is insufficiently established. Fixed sampled error worsened; the two command phases do not establish correct semantics. Preserve negative evidence. Full research training is the next quality-improvement experiment, not certification that quality already passed.',
            'causal_undertraining_claimed':False,'full_generator_data_gate_bypassed':False}
payload={'independent_summary':base64.b64encode(docpath.read_bytes()).decode(),'quality_review':review}
code='''import base64,json,pathlib,hashlib,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');payload=PAYLOAD
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
sp=r/'configs/full-teacher-resume-spec.json';s=json.loads(sp.read_text());assert not s['resume_enabled']
paths=['receipts/quick_functional_acceptance_20261026_v1.json','receipts/online_independent_document_v1_summary.json','receipts/quick_model_quality_review_20261026_v1.json']
assert all(not (r/p).exists() for p in paths)
docraw=base64.b64decode(payload['independent_summary']);d=json.loads(docraw);assert d['all_passed'] and d['passed_count']==430
evidence=[]
def bind(p):
 p=pathlib.Path(p);p=p if p.is_absolute() else r/p
 evidence.append({'path':str(p.relative_to(r)),'sha256':sha(p)})
for run in ['online_author_v11','online_document_v1']:
 o=r/'online_private'/run;j=json.loads((o/'sdk/receipt.json').read_text());x=json.loads((r/'receipts'/('online_'+run+'_outer_exit.json')).read_text())
 assert x['actual_returncode']==0 and j['actual_50hz_deadlines_passed'] and j['deadline_misses']==0
 assert j['actual_control_calls']==257 and not j['physical_failure'] and j['stream_underruns']==0 and j['switched_target_consumed']
 assert j['writers_closed'] and j['output_writers_closed']
 for p in [o/'pipeline.json',o/'sdk/receipt.json',o/'generator_receipt.json',r/'receipts'/('online_'+run+'_outer_exit.json')]:bind(p)
assert sha(r/'online_private/online_document_v1/sdk/receipt.json')=='03dfec6a0974f5252ba1146fe0538e3b051c415167bab93a56ed79d19504aacb'
g=json.loads((r/'configs/quick-generated-reference-spec.json').read_text())
for key,steps in [('vae',2000),('ldm',3000)]:
 m=g['trained_models'][key];j=json.loads(pathlib.Path(m['receipt']).read_text())
 assert sha(m['receipt'])==m['receipt_sha256'] and j['optimizer_updates']==steps and j['output_writers_closed']
 assert sha(m['exit'])==m['exit_sha256'];bind(m['receipt']);bind(m['exit'])
assert sha(s['checkpoint']['path'])==s['checkpoint']['sha256']
stop=r/'receipts/quick5000_controlled_stop.json';assert json.loads(stop.read_text())['status']=='USER_AUTHORIZED_QUICK_5000_EARLY_STOP_COMPLETE'
for p in [stop,r/'receipts/quick5000_backup_verified.json',r/'receipts/offline_chain_acceptance_20261026_v1.json']:bind(p)
(r/paths[1]).write_bytes(docraw);(r/paths[2]).write_text(json.dumps(payload['quality_review'],indent=2)+'\\n');bind(paths[1]);bind(paths[2])
accept={'status':'QUICK_BOUNDED_FUNCTIONAL_CHAIN_ACCEPTED_QUALITY_RESEARCH_PENDING',
 'functional_chain_accepted':True,'quality_review_completed':True,'semantic_quality_accepted':False,'paper_quality_accepted':False,'paper_complete':False,
 'unbounded_hard_realtime_guarantee':False,'final_reference_endpoint_verified':False,'deployable_final_wbc_complete':False,
 'online_runs':['online_author_v11','online_document_v1'],'independent_audit_checks':430,
 'full_training_research_continuation_authorized':True,'authorization':'User: continue completing quick overall validation, then start full training; previous automatic continuation authorization retained.',
 'next_phase':'Resume unchanged 8192-environment paper teacher from verified5000/100000 to40000/800000; then full available-data filtering/corpus gate before paper VAE/LDM.',
 'full_generator_training_started':False,'quality_review':payload['quality_review'],'evidence':evidence}
(r/paths[0]).write_text(json.dumps(accept,indent=2)+'\\n');bind(paths[0])
online=json.loads((r/'configs/quick-online-spec.json').read_text())
for n in ['scripts/run_tracker.sh','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py','scripts/bounded_headless_close.py']:
 s['frozen_bindings'][n]=online['bindings'][n]
for n,h in s['frozen_bindings'].items():assert sha(r/n)==h,n
s['evidence']=evidence;s['resume_enabled']=True
sp.write_text(json.dumps(s,indent=2)+'\\n')
p=subprocess.run(['python3','scripts/supervise_full_teacher_resume.py','--preflight-only'],cwd=r,capture_output=True,text=True)
proof={'actual_preflight_returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'spec_sha256':sha(sp),'acceptance_sha256':sha(r/paths[0]),'sdk_dispatches':0}
with (r/'receipts/full_teacher_resume_preflight_author_v1.json').open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps({'proof':proof,'spec':base64.b64encode(sp.read_bytes()).decode(),'acceptance':accept,'acceptance_bytes':base64.b64encode((r/paths[0]).read_bytes()).decode()}))
'''.replace('PAYLOAD',repr(payload))
j=json.loads(remote(code,timeout=90));o=B/'results/full_teacher_resume_20261026'
(o/'preflight_author_v1.json').write_text(json.dumps(j['proof'],indent=2),encoding='utf-8')
(B/'configs/full-teacher-resume-spec.json').write_bytes(base64.b64decode(j['spec']))
(B/'results/online_quick_20261026/functional_acceptance_v1.json').write_bytes(base64.b64decode(j['acceptance_bytes']))
(B/'results/online_quick_20261026/quality_review_v1.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
for folder in ['online_document_v1','analytic_fk_document_v1']:
    target=B/'results/online_quick_20261026'/folder;target.mkdir(exist_ok=True)
    for name in ['REPORT.md','SUMMARY.json']:shutil.copyfile(T/folder/name,target/name)
print(json.dumps(j['proof']));assert j['proof']['actual_preflight_returncode']==0
