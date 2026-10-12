"""Read-only breakdown of V43 fixed-reference metrics; no altered thresholds."""
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/reference_velocity_v43_20261012'
OUT = ROOT / 'results/reference_quality_v44_20261012'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    threshold_file = BASE / 'frozen_thresholds.json'
    thresholds = json.loads(threshold_file.read_text())['thresholds']
    witness = json.loads((BASE / 'DOCUMENT_WITNESS.json').read_text())
    assert witness['status'] == 'PASS'
    inputs = [threshold_file, BASE / 'DOCUMENT_WITNESS.json', BASE / 'versions.json']
    batches = {}
    for label in ('val_planned_s11', 'val_replay_s11'):
        path = BASE / 'public/raw' / label
        rows = json.loads((path / 'metrics.json').read_text())
        meta = json.loads((path / 'metadata.json').read_text())
        assert len(rows) == 72 and len({r['case_id'] for r in rows}) == 72
        assert len({r['source_id'] for r in rows}) == 24
        inputs.extend(path / name for name in ('metrics.json','metadata.json','receipt.json','lineage.json'))
        predicates = dict(
            completed=lambda r:r['completed'],
            position=lambda r:r['endpoint_error_m'] <= thresholds['goal_position_m'],
            heading=lambda r:r['heading_error_rad'] <= thresholds['heading_rad'],
            time=lambda r:r['endpoint_time_error_s'] <= thresholds['goal_time_s'] + 1e-6,
            slip_proxy=lambda r:r['surface_slip_proxy_fraction'] is not None and r['surface_slip_proxy_fraction'] <= thresholds['sliding_fraction'],
            penetration=lambda r:r['penetration_fraction'] <= thresholds['penetration_fraction'],
            continuity=lambda r:r['jerk_p95_m_s3'] is not None and r['jerk_p95_m_s3'] <= thresholds['continuity_jerk_p95_m_s3'],
            tracking=lambda r:r['tracking_pass'])
        all_flags = []
        for row in rows:
            flags = {name:bool(test(row)) for name,test in predicates.items()}
            assert (flags['position'] and flags['heading'] and flags['time']) == row['condition_pass']
            assert (flags['slip_proxy'] and flags['penetration'] and flags['continuity']) == row['physical_pass']
            assert all(flags.values()) == row['joint_pass']
            all_flags.append(dict(case_id=row['case_id'],source_id=row['source_id'],family=row['family'],offset_m=row['offset_m'],passed=flags))
        batches[label] = dict(attempts=len(rows),sources=24,
            pass_counts={key:sum(test(r) for r in rows) for key,test in predicates.items()},
            joint_pass=sum(r['joint_pass'] for r in rows),
            failures_by_family={family:{key:sum(not test(r) for r in rows if r['family']==family) for key,test in predicates.items()} for family in sorted({r['family'] for r in rows})},
            physical_failure_combinations={pattern:sum(','.join(k for k in ('slip_proxy','penetration','continuity') if not r['passed'][k])==pattern for r in all_flags) for pattern in sorted({','.join(k for k in ('slip_proxy','penetration','continuity') if not r['passed'][k]) for r in all_flags})},
            median_slip_proxy_fraction=statistics.median(r['surface_slip_proxy_fraction'] for r in rows if r['surface_slip_proxy_fraction'] is not None),
            cases=all_flags,
            reference_velocity_rule=meta['reference_velocity_rule'])
    assert batches['val_planned_s11']['cases']==batches['val_replay_s11']['cases']
    result=dict(status='COMPLETE_READ_ONLY_FAILURE_BREAKDOWN',batches=batches,
        thresholds_changed=False,historical_results_changed=False,simulation_calls=0,generation_calls=0,
        innovation_confirmed=False,overleaf_results_sync=False,
        interpretation='Counts explain the frozen conjunction; surface slip is a collision-surface proxy, not ground-truth contact-point slip. Equal replay excludes the previously observed interface discrepancy for this pair, not controller limitations or proxy misspecification. Original dataset reference with corrected velocity is the next calibration control.',
        source_sha256=sha(Path(__file__)),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    OUT.mkdir(exist_ok=True)
    path=OUT / ('diagnostic_%d.json'%time.time_ns())
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({label:{k:v for k,v in b.items() if k not in ('cases','reference_velocity_rule')} for label,b in batches.items()},indent=2))
    print('DIAGNOSTIC',path)

if __name__=='__main__':
    main()
