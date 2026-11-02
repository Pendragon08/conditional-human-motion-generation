"""New disabled actual-pass VAE entry; reuses unchanged numerical runtime only."""
import pathlib,hashlib,json,ast
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,s):
 p=B/n;assert not p.exists();p.write_text(s,encoding='utf-8');ast.parse(s) if p.suffix=='.py' else None
s=(B/'scripts/run_generator8278_vae.py').read_text(encoding='utf-8')
s=s.replace('Separate pre-filter8278 VAE200k; paper-sized model/schedule, explicitly different data filtering.','Fresh VAE200k from actual strict15000 pass corpus; historical EMA route differs.')
s=s.replace('generator8278-vae-spec.json','filtered15000-vae-spec.json').replace('textop.generator8278_vae.v1','textop.filtered15000_vae.v1')
s=s.replace('receipts/generator8278_data_acceptance_v1.json','receipts/filtered15000_data_acceptance_v1.json')
s=s.replace("accepted['prefilter_training_data_accepted'] is True and accepted['paper_filtered_dataset_accepted'] is False","accepted['actual_strict_filtered_training_data_accepted'] is True and accepted['paper_two_stage_route_reproduced'] is False\nfrom filtered15000_inputs import load_selection\ncs=json.loads((ROOT/'configs/filtered15000-corpus-spec.json').read_text());load_selection(cs)\nassert accepted['strict_audit_receipt_sha256']==cs['strict_audit_receipt_sha256']==spec['strict_audit_receipt_sha256']")
s=s.replace('generator8278_20261027/vae_runs','filtered15000_20261027/vae_runs').replace("'generator8278_vae_'","'filtered15000_vae_'")
s=s.replace('GENERATOR8278_VAE','FILTERED15000_VAE').replace('PREFILTER8278_PAPER_SCHEDULE_NOT_FILTERED_PAPER_REPRODUCTION','DIRECT_STRICT15000_FILTERED_PAPER_SCHEDULE')
s=s.replace("'formal_training':False,'paper_schedule':True,'prefilter_data':True","'formal_training':True,'paper_schedule':True,'prefilter_data':False,'actual_strict_input':True,'paper_two_stage_route_reproduced':False")
s=s.replace("budget=check(dest,ROOT,policy,remaining_saves=5)","budget=check(dest,pathlib.Path(spec['checkpoint_allowed_root']),policy,remaining_saves=1 if a.mode=='witness' else 5)")
s=s.replace('check(dest,ROOT,policy,remaining_saves=remaining)',"check(dest,pathlib.Path(spec['checkpoint_allowed_root']),policy,remaining_saves=remaining)")
s=s.replace('check(dest,ROOT,policy,remaining_saves=1)',"check(dest,pathlib.Path(spec['checkpoint_allowed_root']),policy,remaining_saves=1)")
# generator8278_vae_runtime contains only the unchanged numerical model/clock;
# old corpus acceptance, entry, saved optimizer/RNG and checkpoints are not used.
put('scripts/run_filtered15000_vae.py',s)
s=(B/'scripts/supervise_generator8278_vae.py').read_text(encoding='utf-8')
s=s.replace('generator8278','filtered15000').replace('GENERATOR8278_VAE','FILTERED15000_VAE')
put('scripts/supervise_filtered15000_vae.py',s)
s=json.loads((B/'configs/generator8278-vae-spec.json').read_text(encoding='utf-8'))
s.update(schema='textop.filtered15000_vae.v1',witness_authorized=False,train_authorized=False,witnesses=[],
 corpus_dir='/dev/shm/ouyanghao_textop_20261025/filtered15000_actual_pass_v1',data_acceptance_sha256=None,corpus_receipt_sha256=None,strict_audit_receipt_sha256=None,
 checkpoint_dir='/data/code/ouyanghao/oyh/textop-reproduction-20261024/filtered15000_20261027/vae/checkpoints',
 checkpoint_allowed_root='/data/code/ouyanghao/oyh/textop-reproduction-20261024',
 entrypoint_sha256=sha(B/'scripts/run_filtered15000_vae.py'),supervisor_sha256=sha(B/'scripts/supervise_filtered15000_vae.py'),
 scope='New actual-strict-pass corpus only; fresh VAE200k initialization and full state. User bypassed historical EMA/40k selection. No prefilter8100 resume.',
 checkpoint_backup_scope='All real versions retained; select actual persistent owned capacity before enabling any witness/train. No data/checkpoint deletion.')
s['bound_sources']['scripts/filtered15000_inputs.py']=sha(B/'scripts/filtered15000_inputs.py')
put('configs/filtered15000-vae-spec.json',json.dumps(s,indent=2)+'\n')
print(json.dumps(dict(status='NEW_DISABLED_ACTUAL_PASS_VAE_ENTRY_PREPARED',actual_model_loads=0,actual_training_runs=0)))
