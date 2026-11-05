"""CPU-only source-based native budget calculation; no training or GPU work."""
import ast,hashlib,json,math,pathlib
from fractions import Fraction
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
N=R/'stage3_supermimic29_20261103/native_universal_20261104';O=N/'formal2000_preparation_v1'
source=N/'source/supermimic/train.py';expected='b94569605bd3ff96a49d4903bb791bb54bc4695a03f1914c8447c0ec7dd1e6bd'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(source)==expected and not O.exists()
tree=ast.parse(source.read_text());defaults={}
for node in ast.walk(tree):
 if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='add_argument':continue
 if not node.args or not isinstance(node.args[0],ast.Constant):continue
 default=next((x.value for x in node.keywords if x.arg=='default'),None)
 if default is not None:
  try:defaults[node.args[0].value]=ast.literal_eval(default)
  except (ValueError,TypeError):pass
assert defaults['--utd']==.4 and defaults['--wm_only_iters']==500 and defaults['--warmup_steps']==64
assert defaults['--wm_batches']==10 and defaults['--critic_epochs']==5 and defaults['--critic_minibatches']==4
text=source.read_text();assert 'step_acc += 1.0 / args.utd' in text and 'sim_steps += args.num_envs' in text
assert 'if torch.isfinite(gn):\n        opt_pi.step()' in text
target=3000*24;envs=8912;utd=Fraction('0.4');iterations=28774;warmup=65
accumulator=0.;native_collected=0;two_step=0;three_step=0
for _ in range(iterations):
 accumulator+=1/.4;n=int(accumulator);accumulator-=n;native_collected+=n
 two_step+=n==2;three_step+=n==3
assert native_collected==math.floor(iterations/utd)==71935
assert warmup+native_collected==target==72000 and target*envs==641664000
wm_only=500;attempts=iterations-wm_only
checkpoints=iterations//2000+bool(iterations%2000)
large=218548257;small=71782661
report={'status':'source_budget_plan_prepared_not_training_ready','formal_training_started':False,'source_train':str(source),'source_train_sha256':sha(source),
 'reference_library_required':{'tasks':['pick','pour','bimanual_box_lift','box_push_pull'],'per_task':500,'total':2000,
  'native_motion_dir':str(R/'stage3_supported_references_20261104/mixed2000_final_v1/motion_csv20'),
  'data_completion_receipt':str(R/'stage3_supported_references_20261104/mixed2000_final_v1/completion_receipt.json')},
 'single_launch_budget':{'num_envs':envs,'native_iterations':iterations,'native_utd':.4,'initial_real_vector_steps':warmup,
  'training_loop_real_vector_steps':native_collected,'total_real_vector_steps':target,'total_real_environment_transitions':target*envs,
  'collection_iterations_of_two_steps':two_step,'collection_iterations_of_three_steps':three_step,
  'world_model_only_native_iterations':wm_only,'nominal_world_model_Adam_steps':iterations*10,
  'policy_update_attempts':attempts,'maximum_policy_Adam_steps':attempts,'nominal_critic_Adam_steps_if_no_early_stop':attempts*20,
  'imagined_policy_transitions_excluded_from_real_budget':attempts*32*2048,
  'actual_optimizer_steps_must_be_read_from_Adam_state':True,
  'native_upstream_policy_nonfinite_gradient_rule':'upstream skips opt_pi.step when grad norm is not finite; count actual Adam steps independently and do not report every iteration as a successful policy update'},
 'planned_parameters':{'buffer_steps':128,'wm_batch':2048,'pi_batch':2048,'horizon':32,'goal_frames':10,'history_frames':10,
  'wm_batches':10,'wm_only_iters':500,'amp':'bf16','TF32':True,'utd':.4,'num_envs':envs,'warmup_steps':65},
 'differences_from_pinned_defaults':{'buffer_steps':[defaults['--buffer_steps'],128],'pi_batch':[defaults['--pi_batch'],2048],
  'warmup_steps':[defaults['--warmup_steps'],65],'iterations':[defaults['--iterations'],iterations]},
 'checkpoint_budget_example_not_authorization':{'ckpt_every':2000,'number_of_distinct_small_snapshots':checkpoints,
  'observed_large_snapshot_bytes':large,'observed_small_snapshot_bytes':small,
  'minimum_output_bytes_with_one_extra_full_snapshot_and_margin':checkpoints*small+3*large+128*1024**2,
  'source':'actual completed native benchmark/resume sizes; large snapshots do not include replay/RNG',
  'early_full_optimizer_snapshot_and_resume_layout':'must be implemented before launch; preserve all numbered snapshots and complete checkpoint evidence'},
 'resume_accounting':{'upstream_restores_replay':False,'upstream_restores_RNG':False,'step_acc_restarts_at_zero':True,
  'rule':'each recovered run recollects a real replay warmup and resets fractional collection accumulator; recompute total remaining iterations from actual logged sim_steps, never add an uncounted warmup',
  'example_one_even_iteration_boundary_restart_with65_new_warmup_steps':{'final_native_iteration':iterations-26,
   'real_vector_steps':2*65+int((iterations-26)/.4)},
  'diagnostic_interactions':'separate ledger, excluded from the new formal72000 vector-step budget'},
 'pending_before_launch':[
  'real four-task2000 completion receipt and manifest hashes, not preview8 or previous500 capacity',
  'actual2000 source library CPU MotionLib report (completion follower1491803 supplies this)',
  'bounded actual8912 native simulation with new2000 library to measure added-library GPU/RSS capacity; do not repeat prior500 library witness',
  'exact output/checkpoint capacity including early recoverable native model/optimizer snapshot and temporary writes',
  'formal guard for per-job12GiB GPU/32GiB RSS, aggregate24/64GiB, >=11GiB startup and remaining checkpoints, runtime8GiB+356MiB',
  'unmodified full native trainer dispatch, counter evidence and fail-preserving checkpoint/finite-metric monitoring',
  'read-only all2000 native body-tracking evaluation plus per-task aggregation; separate hand/object physical task evaluation'
 ],
 'scope':{'policy':'one shared reference-conditioned body29 tracker,10 proprioceptive history frames and10 future reference frames',
  'native_actions':29,'hands_objects':'retained in source/sidecar; explicit execution bridge and task evaluation remain separate from native body-tracking success',
  'PPO':'unchanged autonomous Pick3000 budget, currently running; other tasks remain independently trained',
  'fairness':'same real-transition budget only, not identical actor observations, objectives, physics or hand/object interfaces'},
 'gate_snapshot':json.loads((N/'all_four_reference_gate.json').read_text())}
assert report['resume_accounting']['example_one_even_iteration_boundary_restart_with65_new_warmup_steps']['real_vector_steps']==72000
O.mkdir(exist_ok=False);(O/'budget_source_audit.json').open('x').write(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({k:report[k] for k in ['status','formal_training_started','single_launch_budget','differences_from_pinned_defaults']},indent=2))
