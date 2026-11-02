"""Draft pure join from independently audited interrupted TRAIN prefix to FK inputs.

Caller must independently authenticate all metadata, own the shared lock for
numerical operations and audit the original TRAIN library. This module neither
opens files nor executes FK; the returned plan is private metadata.
"""
import pathlib
import re

_SHA = re.compile(r'^[0-9a-f]{64}$')
_ROOT = pathlib.PurePosixPath('/data/code/ouyanghao/oyh/textop-reproduction-20261024/step34_20261029/user_augmentation5368_production_v1')


def plan_fk(audit, plan, source_joint_names):
    if audit.get('status') != 'TRAIN_AUGMENTATION_INTERRUPTED_PREFIX_2526_ARRAYS_AUDITED':
        raise ValueError('the complete producer audit is required')
    if audit.get('all_arrays_finite') is not True or audit.get('all_quaternions_unit') is not True:
        raise ValueError('numerical audit is incomplete')
    if len(plan['clips']) != 5368:
        raise ValueError('original generation plan is incomplete')
    clips, items = plan['clips'][:2526], audit['generated_train_items']
    if len(clips) != 2526 or len(items) != 2526 or audit['generated_clips'] != 2526:
        raise ValueError('wrong production clip count')
    if len(source_joint_names) != 5361:
        raise ValueError('incomplete TRAIN source name metadata')
    variants = {tuple(value) for value in source_joint_names.values()}
    if len(variants) != 1:
        raise ValueError('TRAIN joint ordering is not uniform')
    names = next(iter(variants))
    if len(names) != 29 or len(set(names)) != 29 or any(not isinstance(x, str) or not x for x in names):
        raise ValueError('invalid source joint names')
    refs, paths, total_intervals, total_frames, budget = [], set(), 0, 0, 0
    for index, (clip, item) in enumerate(zip(clips, items)):
        if type(clip['clip_index']) is not int or clip['clip_index'] != index or item['index'] != index:
            raise ValueError('missing, duplicated or reordered clip')
        if clip['source_split'] != 'train' or clip['initial_source'] not in source_joint_names:
            raise ValueError('non-TRAIN source')
        intervals = clip['target_control_intervals']
        frames = clip['target_pose_frames']
        if type(intervals) is not int or intervals < 1 or frames != intervals + 1:
            raise ValueError('invalid timing')
        if item['control_intervals'] != intervals or item['pose_frames'] != frames:
            raise ValueError('audit and source plan disagree')
        path = pathlib.PurePosixPath(item['path'])
        if path.parent != _ROOT or path.name != 'clip_' + str(index).zfill(5) + '.private.npz' or path in paths:
            raise ValueError('unexpected or duplicate production artifact')
        if not isinstance(item['sha256'], str) or not _SHA.fullmatch(item['sha256']):
            raise ValueError('invalid artifact SHA')
        qpos_sha = item['array_sha256']['qpos_29']
        if not isinstance(qpos_sha, str) or not _SHA.fullmatch(qpos_sha):
            raise ValueError('invalid pose-array SHA')
        if type(item['bytes']) is not int or not 0 < item['bytes'] <= frames * 152 + 4096:
            raise ValueError('invalid artifact length')
        paths.add(path)
        total_intervals += intervals; total_frames += frames; budget += frames * 2152 + 8192
        refs.append(dict(index=index, source_kind='generated_TRAIN', source_family=clip['initial_source'],
                         source_file=str(path), source_sha256=item['sha256'], source_bytes=item['bytes'],
                         qpos_sha256=qpos_sha, frames=frames,
                         control_intervals=intervals, fps=50, joint_names=list(names)))
    if total_intervals != 2666420 or total_frames != 2668946:
        raise ValueError('generated coverage differs from the authorized scope')
    if budget != 5764264784:
        raise ValueError('generated output bound differs from the checked prefix')
    return dict(schema='textop.augmentation2526_fk_plan.v1', inputs=refs,
                clips=len(refs), control_intervals=total_intervals, pose_frames=total_frames,
                output_npz_upper_bytes=budget, source_split='TRAIN',
                original_VAL_motions_used=0, fk_executed=False)
