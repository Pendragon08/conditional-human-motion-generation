"""Read-only component screen on observed C1 construction/replay records.

This is neither a C2 three-call experiment nor a C3 acquisition/calibration run.
No rejected inputs, targets, scores or historical outputs are altered.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import time

from condition_feedback_guard_core_v29_20261012 import acceptable

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/paired_feedback_validation_v51_20261012'
OUT = ROOT / 'results/remaining_candidate_screen_20261012'


def main():
    started = time.monotonic()
    if OUT.exists():
        raise RuntimeError('Retain earlier attempt; no overwrite or automatic retry')
    bindings = {}
    def read(path):
        raw = path.read_bytes()
        bindings[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    versions = read(BASE / 'versions.json')
    summary = read(BASE / 'summary.json')
    if summary['status'] != 'COMPLETE_VALIDATION_QUALITY_FLOOR':
        raise RuntimeError('Unexpected parent state; inspect before diagnosing')
    if not versions['generator_frozen'] or not versions['tracker_frozen']:
        raise RuntimeError('Requires frozen model setting')
    seeds = (11, 12, 13)
    detailed = []
    replay_evidence = []
    reports = []
    for seed in seeds:
        paths = [BASE / 'raw' / ('val_shared_s%d_r0' % seed)]
        paths += [BASE / 'raw' / ('val_feedback_s%d_r%d' % (seed, k)) for k in (1, 2)]
        rows = [read(p / 'metrics.json') for p in paths]
        metas = [read(p / 'metadata.json') for p in paths]
        if any(len(r) != 72 for r in rows):
            raise RuntimeError('Incomplete denominator')
        if len({r['source_id'] for r in rows[0]}) != 24:
            raise RuntimeError('Incorrect source scope')
        for m in metas:
            if m['original_goal_sha256'] != metas[0]['original_goal_sha256']:
                raise RuntimeError('Original evaluation goals differ')
        reasons = Counter()
        chosen = Counter()
        joint_available = 0
        physical_available = 0
        for i in range(72):
            identity = (rows[0][i]['case_id'], rows[0][i]['source_id'])
            if any((r[i]['case_id'], r[i]['source_id']) != identity for r in rows):
                raise RuntimeError('Unpaired construction inputs')
            choice = 0
            decisions = []
            for k in (1, 2):
                accepted, reason = acceptable(rows[choice][i], rows[k][i])
                reasons[reason] += 1
                decisions.append(dict(candidate=k, accepted=accepted, reason=reason))
                if accepted:
                    choice = k
            chosen[str(choice)] += 1
            joint_available += int(any(r[i]['joint_pass'] for r in rows))
            physical_available += int(any(r[i]['physical_pass'] for r in rows))
            detailed.append(dict(seed=seed, case_id=identity[0], source_id=identity[1],
                                 observed_bank_choice=choice, decisions=decisions,
                                 selected_metrics=rows[choice][i]))
        replay_rows = []
        for arm in ('feedback', 'random'):
            p = BASE / 'raw' / ('val_%s_replay_s%d' % (arm, seed))
            rs = read(p / 'metrics.json'); meta = read(p / 'metadata.json')
            if len(rs) != 72:
                raise RuntimeError('Missing independent replay rows')
            if meta['original_goal_sha256'] != metas[0]['original_goal_sha256']:
                raise RuntimeError('Replay evaluation goal mismatch')
            if [(r['case_id'], r['source_id']) for r in rs] != [(r['case_id'], r['source_id']) for r in rows[0]]:
                raise RuntimeError('Replay source/order mismatch')
            replay_rows.append(rs)
            replay_evidence.append(dict(seed=seed, arm=arm, rows=72,
                joint_pass=sum(int(r['joint_pass']) for r in rs),
                physical_pass=sum(int(r['physical_pass']) for r in rs)))
        reports.append(dict(seed=seed, inputs=72, sources=24,
            construction_candidates_per_input=3, choices=dict(chosen),
            decisions=144, reasons=dict(reasons),
            cases_with_observed_joint_qualified_candidate=joint_available,
            cases_with_observed_physical_qualified_candidate=physical_available))
    for path in (Path(__file__), ROOT/'scripts/condition_feedback_guard_core_v29_20261012.py',
                 ROOT/'scripts/repair_budget_core_v38_20261012.py',
                 ROOT/'docs/condition_feedback_guard_simulation_v31_20261012.md',
                 ROOT/'docs/repair_budget_candidate_v38_20261012.md'):
        bindings[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(status='COMPLETE_RETROSPECTIVE_COMPONENT_SCREEN',
        scope='Internal validation only; observed C1 bank, not the preregistered C2 repeat-plus-update bank',
        upstream_split_warning='Internal val includes official HumanML3D train and test sources; not held-out upstream evaluation',
        candidate_efficacy_test=False, new_simulator_calls=0, new_generation_calls=0,
        controller_training_calls=0, gpu_seconds=0, cpu_seconds=time.monotonic()-started,
        models_frozen=True, targets_changed=False, thresholds_changed=False,
        source_inputs=24, seed_offset_inputs=216, reports=reports, choices=detailed,
        independent_replay_evidence=replay_evidence,
        c2_status='NOT_RUN_ORIGINAL_NONZERO_JOINT_PREREQUISITE_UNMET',
        c3_status='NOT_RUN_NO_OBSERVED_NONZERO_JOINT_REPAIR_LABELS',
        limits=[
            'Construction scores do not establish utility of an independently replayed selected reference.',
            'All-zero observed candidates bound selection gains only on this finite bank; unseen candidates may differ.',
            'No prospective acquisition schedule, paid-call savings, efficacy, robot transfer or downstream benefit is measured.',
            'Componentwise guard screens multiple metrics and can reject a useful trade-off; no new tolerance is tuned here.',
        ], input_hashes=bindings)
    OUT.mkdir()
    (OUT/'component_screen.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','seed_offset_inputs','reports','c2_status','c3_status')},indent=2),flush=True)


if __name__ == '__main__':
    main()
