"""Metadata-only naming contract for a future TextOp -> SuperMimic adapter.

This module performs no file access, motion loading, model import, or training.
It does not establish that the two robot assets are kinematically equivalent.
"""

def reorder_indices(source_names, target_names, expected_count):
    """Return target-order indices, rejecting absent, duplicate or extra names."""
    if not isinstance(source_names, (list, tuple)) or not isinstance(target_names, (list, tuple)):
        raise ValueError('name sequences required')
    if len(source_names) != expected_count or len(target_names) != expected_count:
        raise ValueError('wrong count')
    for seq in (source_names, target_names):
        if any(not isinstance(n, str) or not n or n.strip() != n for n in seq):
            raise ValueError('invalid name')
        if len(set(seq)) != expected_count:
            raise ValueError('duplicate name')
    if set(source_names) != set(target_names):
        raise ValueError('source and target names differ')
    src_index = {name: i for i, name in enumerate(source_names)}
    return [src_index[name] for name in target_names]


def plan_names(source_joints, target_joints, source_bodies, target_bodies):
    if 'pelvis' not in source_bodies or 'pelvis' not in target_bodies:
        raise ValueError('pelvis missing')
    return {
        'joint_target_from_source': reorder_indices(source_joints, target_joints, 29),
        'body_target_from_source': reorder_indices(source_bodies, target_bodies, 30),
        'source_pelvis_index': source_bodies.index('pelvis'),
        'target_pelvis_index': target_bodies.index('pelvis'),
        'asset_equivalence_proven': False,
        'real_data_validation_complete': False,
    }
