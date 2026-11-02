"""Aggregate future new-actor rollout records; no model or SDK execution.

This is a bookkeeping component, not a source-binding or physics acceptance
checker. The caller must independently validate actual references, episodes,
model provenance, finite arrays, output closure, and strict test conditions.
"""
from ppo5000_heldout_manifest_contract import validate as validate_manifest


def aggregate(manifest, records, seeds):
    """Join all trials once; preserve terminations and incomplete timeouts.

    The exact finite candidate set is established by the metadata contract.
    External timeouts remain incomplete, not successful or physical failures.
    """
    validate_manifest(manifest, seeds)
    key = lambda r: (r['segment_id'], r['seed'], r['mode'])
    expected = {key(row): row for row in manifest}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError('every reference must have exactly one record')
    seen = set()
    groups = {(seed, mode): {'complete': 0, 'failed': 0, 'incomplete': 0}
              for seed in seeds for mode in ('conditional_cfg5', 'unconditional')}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('trial must be an object')
        try:
            identity = key(row)
        except KeyError as exc:
            raise ValueError('trial identity is missing') from exc
        if identity not in expected or identity in seen:
            raise ValueError('unknown or duplicate trial')
        seen.add(identity)
        trial = row.get('trial')
        if not isinstance(trial, dict):
            raise ValueError('trial evidence is missing')
        frames = expected[identity]['frames']
        steps = trial.get('control_steps')
        if (trial.get('repeat') != 0 or trial.get('start_frame') != 0 or
                trial.get('reference_frames') != frames or type(steps) is not int or
                not 1 <= steps <= frames - 1):
            raise ValueError('invalid frame coverage')
        failed, complete, reason = trial.get('failed'), trial.get('complete'), trial.get('end_reason')
        if type(failed) is not bool or type(complete) is not bool:
            raise ValueError('trial decisions must be boolean')
        if reason == 'motion_end' and complete and not failed and steps == frames - 1:
            outcome = 'complete'
        elif reason == 'termination' and failed and not complete:
            outcome = 'failed'
        elif reason == 'external_timeout' and not failed and not complete:
            outcome = 'incomplete'
        else:
            raise ValueError('inconsistent terminal outcome')
        groups[(identity[1], identity[2])][outcome] += 1
    if seen != set(expected):
        raise ValueError('reference coverage mismatch')
    per_group = []
    for seed in seeds:
        for mode in ('conditional_cfg5', 'unconditional'):
            counts = groups[(seed, mode)]
            denominator = sum(counts.values())
            if denominator != 188:
                raise ValueError('unexpected group coverage')
            per_group.append({'seed': seed, 'mode': mode, 'trials': denominator,
                              **counts, 'completion_fraction_all_trials': counts['complete']/denominator})
    totals = {name: sum(group[name] for group in per_group) for name in ('complete','failed','incomplete')}
    return {'trials': len(records), **totals, 'per_seed_and_mode': per_group,
            'all_trials_completed_or_failed': totals['incomplete'] == 0,
            'threshold_tuning_or_seed_selection': False}
