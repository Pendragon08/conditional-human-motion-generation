"""Draft metadata contract for original TRAIN plus generated TRAIN PPO.

This does not read reference arrays, construct a simulator, or claim that the
generated FK library or a PPO job exists. Callers must authenticate the two
underlying manifests and their terminal evidence before building a receipt.
"""
import pathlib
import re

SHA = re.compile(r'^[0-9a-f]{64}$')
ROOT = pathlib.PurePosixPath('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
ORIGINAL = ROOT / 'step34_numeric_v1/user_step4_train_only_fk_library_v1'
GENERATED = ROOT / 'step34_20261029/user_augmentation2526_fk_library_v1'


def validate_policy(policy):
    if policy.get('schema') != 'textop.augmentation2526_mixed_ppo.v1':
        raise ValueError('mixed PPO schema required')
    expected = dict(training_data_mode='original_train_and_generated_train',
                    target_RL_iterations=5000, human_selected_target_iterations=5000,
                    checkpoint_interval=1000, num_envs=8192,
                    actor_observation_dim=431, critic_observation_dim=557,
                    future_reference_frames=5, action_dim=29, policy_hz=50,
                    method='one_stage_PPO', dagger_or_distillation=False,
                    fresh_actor_critic_and_Adam_initialization=True,
                    old_privileged_teacher_training_resumed=False)
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError('mixed PPO policy mismatch: ' + key)
    if policy.get('actor_hidden') != [2048, 1024, 512] or policy.get('critic_hidden') != [2048, 1024, 512]:
        raise ValueError('unexpected PPO architecture')
    if type(policy.get('training_authorized')) is not bool or type(policy.get('witnesses_authorized')) is not bool:
        raise ValueError('explicit Boolean authorization fields required')
    return policy


def validate_records(original, generated):
    if len(original) != 5551 or len(generated) != 2526:
        raise ValueError('both complete source sets are required')
    paths = set()
    for kind, records, parent in (('original_TRAIN', original, ORIGINAL),
                                  ('generated_TRAIN', generated, GENERATED)):
        for index, row in enumerate(records):
            path = pathlib.PurePosixPath(row['file'])
            if row.get('kind') != kind or row.get('source_split') != 'train' or path.parent != parent:
                raise ValueError('unexpected source split or reference directory')
            if path in paths or not isinstance(row.get('sha256'), str) or not SHA.fullmatch(row['sha256']):
                raise ValueError('duplicate reference or invalid SHA')
            if type(row.get('frames')) is not int or row['frames'] < 2:
                raise ValueError('invalid frame count')
            if kind == 'generated_TRAIN' and row.get('generated_index') != index:
                raise ValueError('generated reference order mismatch')
            paths.add(path)
    return dict(status='MIXED_REFERENCE_METADATA_VALID', original_clips=len(original),
                generated_clips=len(generated), total_clips=len(paths),
                original_val_motions=0, real_training_started=False)
