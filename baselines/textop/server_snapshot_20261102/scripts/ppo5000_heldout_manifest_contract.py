"""Metadata-only contract for a future PPO5000 heldout generated-reference replay.

No model, simulation, file content or permission checks occur here. The caller
must separately bind the real accepted FK receipt, full manifest and files by
content hash, owner, path and closure before any physical evaluation.
"""
import re

_SHA = re.compile(r"^[0-9a-f]{64}$")


def validate(rows, seeds):
    """Reject missing, extra, duplicated or incompatible candidate metadata.

    Every one of 188 segment identities must occur exactly once for each of
    the five explicitly supplied seeds and both original generation modes.
    Failure is never a reason to remove a reference from this contract.
    """
    if not isinstance(rows, list) or len(rows) != 1880:
        raise ValueError('expected all 1880 original generated references')
    if (not isinstance(seeds, list) or len(seeds) != 5 or
            any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 5):
        raise ValueError('five distinct explicitly bound integer seeds required')
    seeds_set = set(seeds)
    modes = {'conditional_cfg5', 'unconditional'}
    seen = set()
    segments = set()
    paths = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('reference row must be an object')
        if row.get('status') != 'REAL_GENERATED_ISAAC_FK_EXPORTED':
            raise ValueError('unaccepted FK row')
        if row.get('source_kind') != 'REAL_LDM100000_GENERATED_SCORED_FUTURE':
            raise ValueError('wrong reference origin')
        if row.get('physical_filter_passed') is not False:
            raise ValueError('generated reference must not claim physical filtering')
        segment, seed, mode = row.get('segment_id'), row.get('seed'), row.get('mode')
        if not isinstance(segment, str) or not segment or type(seed) is not int:
            raise ValueError('invalid segment or seed')
        if seed not in seeds_set or mode not in modes:
            raise ValueError('unbound seed or mode')
        key = (segment, seed, mode)
        if key in seen:
            raise ValueError('duplicate segment/seed/mode')
        seen.add(key)
        segments.add(segment)
        path, digest = row.get('file'), row.get('sha256')
        if not isinstance(path, str) or not path.startswith('/') or path in paths:
            raise ValueError('invalid or duplicated absolute reference path')
        paths.add(path)
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            raise ValueError('invalid sha256')
        if type(row.get('bytes')) is not int or row['bytes'] <= 0:
            raise ValueError('invalid reference byte count')
        if type(row.get('frames')) is not int or row['frames'] < 2:
            raise ValueError('reference must contain at least one interval')
    if len(segments) != 188:
        raise ValueError('expected 188 segment identities')
    expected = {(segment, seed, mode) for segment in segments for seed in seeds_set for mode in modes}
    if seen != expected:
        raise ValueError('incomplete Cartesian candidate set')
    return {'reference_count': len(rows), 'segment_count': len(segments),
            'seed_count': len(seeds_set), 'mode_count': len(modes),
            'physical_outcomes_used_for_selection': False}
