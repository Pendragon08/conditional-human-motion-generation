"""Guards for the explicitly user-requested5k quick reproduction budget."""
ROOT = '/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RUN = 'filter_policy_paper69_solid_floor_20261026_v1'
TRAINER = ROOT + '/scripts/train_filter_policy.py'
OUT = ROOT + '/tracker_validation/' + RUN
PLAN_SHA = '153f8237a17928ffc3a28b7b00e366742a4ff0080e8fbb6bac87e57da8eb291d'


def validate_progress(r, plan):
    assert plan['quick_target_iterations'] == 5000 and plan['paper_target_iterations'] == 40000
    assert r['status'] == 'RUNNING_FORMAL_FILTER_TRAINING'
    assert r['mode'] == 'formal' and r['num_envs'] == 8192 and r['plan_sha256'] == PLAN_SHA
    assert r['optimizer_updates'] == r['completed_learning_iterations'] * 20
    return r['completed_learning_iterations'] >= 5000


def validate_group(launch, members):
    command = launch['command']
    assert TRAINER in command and command[command.index('--output') + 1] == OUT
    assert command[command.index('--mode') + 1] == 'formal'
    group = launch['child_pid']
    assert group > 1 and members
    assert all(m['uid'] == 1013 and m['group'] == group and m['session'] == group for m in members)
    candidates = [m for m in members if m['python_executable'] and TRAINER in m['argv']
                  and '--output' in m['argv'] and m['argv'][m['argv'].index('--output') + 1] == OUT]
    assert len(candidates) == 1, 'Exactly the target trainer Python process is required'
    return candidates[0]['pid'], group


def validate_backup(backup, checkpoint):
    assert backup['status'] == 'FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED'
    assert backup['completed'] == backup['audit']['completed_learning_iterations'] == 5000
    assert backup['audit']['actual_optimizer_updates'] == 100000
    assert backup['private_full_two_location_sha_verified'] is True
    assert backup['inference_three_location_sha_verified'] is True
    assert backup['original_server_checkpoint_retained'] is True and backup['data_deleted'] is False
    assert backup['full_checkpoint_publicly_uploaded'] is False
    proof = backup['audit']['full_checkpoint']
    assert proof['path'] == OUT + '/completed_5000.pt'
    assert proof['sha256'] == checkpoint['sha256'] and proof['bytes'] == checkpoint['bytes']
