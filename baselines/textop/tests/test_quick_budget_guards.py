"""CPU-only safety checks: no signals, SDK work, or synthetic training claims."""
import copy
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from quick_budget_guards import RUN, OUT, TRAINER, PLAN_SHA, validate_progress, validate_group, validate_backup

plan = dict(quick_target_iterations=5000, paper_target_iterations=40000)
progress = dict(status='RUNNING_FORMAL_FILTER_TRAINING', mode='formal', num_envs=8192,
                plan_sha256=PLAN_SHA, completed_learning_iterations=5000, optimizer_updates=100000)
launch = dict(child_pid=43210, command=['bash', TRAINER, '--output', OUT, '--mode', 'formal'])
member = dict(pid=43211, uid=1013, group=43210, session=43210, python_executable=True,
              argv=['python', TRAINER, '--output', OUT, '--mode', 'formal'])
checkpoint = dict(bytes=90, sha256='a'*64)
backup = dict(status='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED', completed=5000,
              audit=dict(completed_learning_iterations=5000, actual_optimizer_updates=100000,
                         full_checkpoint=dict(path=OUT+'/completed_5000.pt', **checkpoint)),
              private_full_two_location_sha_verified=True, inference_three_location_sha_verified=True,
              original_server_checkpoint_retained=True, data_deleted=False, full_checkpoint_publicly_uploaded=False)
checks = []

def rejects(name, call):
    try:
        call()
    except AssertionError:
        checks.append(dict(case=name, expected_rejection=True))
    else:
        raise AssertionError(name)

assert validate_progress(progress, plan) is True
assert validate_group(launch, [member]) == (43211,43210)
validate_backup(backup, checkpoint)
checks.append(dict(case='synthetic_complete_ready_identity_only', signals_sent=0))
assert validate_progress(dict(progress, completed_learning_iterations=2000, optimizer_updates=40000), plan) is False
checks.append(dict(case='not_yet_at_target_no_stop', passed=True))
rejects('other_user_refused', lambda:validate_group(launch, [dict(member,uid=1234)]))
rejects('other_group_refused', lambda:validate_group(launch, [dict(member,group=99999)]))
rejects('other_session_refused', lambda:validate_group(launch, [dict(member,session=99999)]))
rejects('wrong_output_refused', lambda:validate_group(launch, [dict(member,argv=['python',TRAINER,'--output','/data/other/run'])]))
rejects('duplicate_python_refused', lambda:validate_group(launch, [member,dict(member,pid=43212)]))
rejects('failed_training_refused', lambda:validate_progress(dict(progress,status='FAILED'),plan))
rejects('backup_not_verified_refused', lambda:validate_backup(dict(backup,private_full_two_location_sha_verified=False),checkpoint))
bad=copy.deepcopy(backup);bad['audit']['full_checkpoint']['sha256']='b'*64
rejects('different_checkpoint_refused',lambda:validate_backup(bad,checkpoint))
rejects('wrong_update_count_refused',lambda:validate_progress(dict(progress,optimizer_updates=90000),plan))
print(json.dumps(dict(status='QUICK_BUDGET_CPU_GUARDS_PASSED',checks=checks,checks_passed=len(checks),
                      signals_sent=0,sdk_runs=0,optimizer_updates=0,fixtures_are_synthetic=True)))
