"""Explicit original-TRAIN route after the user postponed data augmentation.

No model/SDK/array imports. Stage3 receipts are not relabelled as generated data.
"""
import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
SHM = pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
SCOPE_SHA = 'ac49c12afeedc8703cf2c3b8c8e19df4c25fc691c2220df9d927d32d3552cbb4'
TRAIN_SHA = 'f355de1c947487d750cbe3d59c51d78c8853a36315c659a3b08b4876df7273d2'
RESERVE = 2 * 1024**3
LOG = 128 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def owned(path):
    p = pathlib.Path(path)
    assert p.is_absolute() and p.resolve() == p and p.stat().st_uid == 1013
    assert any(p == r or r in p.parents for r in [ROOT, SHM]), str(p)
    assert not p.is_symlink() and all(not x.is_symlink() for x in p.parents)
    return p


def read(path):
    return json.loads(owned(path).read_bytes())


def atomic(path, value):
    path = pathlib.Path(path)
    tmp = path.with_suffix('.partial')
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with tmp.open('wb') as f:
        os.chmod(tmp, 0o600); f.write(raw); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def scope():
    p = owned(ROOT / 'step34_20261029/user_augmentation5368_scope_v1/augmentation_scope.private.json')
    assert sha(p) == SCOPE_SHA
    plan = read(p)
    rp = owned(ROOT / 'evaluator_tmr_20261029/tmr69d_training_scope_v2/receipt.json')
    assert sha(rp) == TRAIN_SHA
    rec = read(rp)
    assert rec['status'] == 'TMR69D_TRAINING_METADATA_SCOPE_BUILT' and rec['output_writers_closed']
    item = rec['private_artifact']; src = owned(item['path'])
    assert src.stat().st_size == item['bytes'] and sha(src) == item['sha256']
    data = read(src); sources = {x['canonical_source']: x for x in data['sources']}
    assert len(sources) == 5361 and all(x['original_split'] == 'train' and x['mirrored'] is False for x in sources.values())
    # Resolve the original name order from already accepted metadata, never infer
    # it from a tensor or from the Isaac ordering.
    names = {}
    for filename, digest in data['prepared_manifest_sha256'].items():
        mp = owned(filename); assert sha(mp) == digest
        for line in mp.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['canonical_source'] in sources:
                assert row['split'] == 'train' and row['dof'] == 29
                assert row['output_sha256'] == sources[row['canonical_source']]['output_sha256']
                assert len(row['joint_names']) == len(set(row['joint_names'])) == 29
                assert row['canonical_source'] not in names
                names[row['canonical_source']] = row['joint_names']
    assert set(names) == set(sources)
    clips = plan['original_RL_clips']
    assert len(clips) == 5551 and sum(x['control_intervals'] for x in clips) == 2835665
    coverage = {k: [] for k in sources}
    for x in clips:
        srcrow = sources[x['source_family']]
        assert x['source_split'] == 'train' and x['mirrored'] is False and x['generated'] is False
        assert x['source_path'] == srcrow['output_file'] and x['source_sha256'] == srcrow['output_sha256']
        first, last = x['first_pose_index'], x['last_pose_index_inclusive']
        assert type(first) is int and type(last) is int and 0 <= first < last < srcrow['frames50']
        assert last - first == x['control_intervals'] <= 1999 and x['fps'] == 50
        coverage[x['source_family']].append((first, last))
    for key, intervals in coverage.items():
        intervals.sort()
        assert intervals[0][0] == 0 and intervals[-1][1] == sources[key]['frames50'] - 1
        assert all(a[1] == b[0] for a, b in zip(intervals[:-1], intervals[1:]))
    return clips, sources, names, dict(scope_sha256=sha(p), train_receipt_sha256=sha(rp), train_artifact_sha256=sha(src))


def select(role):
    clips, sources, names, binding = scope()
    assert role in ['author', 'document', 'library']
    selected = clips if role == 'library' else clips[:1]
    return selected, sources, names, binding


def validate_policy(s):
    assert s['schema'] == 'textop.stage4_original_train_only.v1'
    assert s['training_data_mode'] == 'original_train_only' and s['stage3_status'] == 'POSTPONED_BY_USER'
    assert s['generated_training_clips'] == 0 and s['original_VAL_motions_in_training'] == 0
    assert s['actor_observation_dim'] == 431 and s['critic_observation_dim'] == 557
    assert s['future_reference_frames'] == 5 and s['action_dim'] == 29 and s['policy_hz'] == 50
    assert s['actor_hidden'] == s['critic_hidden'] == [2048, 1024, 512]
    assert s['method'] == 'one_stage_PPO' and s['dagger_or_distillation'] is False
    assert s['fresh_actor_critic_and_Adam_initialization'] is True and s['old_privileged_teacher_training_resumed'] is False
    assert type(s['training_authorized']) is bool and type(s['witnesses_authorized']) is bool
    if s['training_authorized']:
        assert s['accepted_original_train_library'] and s['real_RL_witnesses_accepted'] is True
    return s
