"""Post-hoc archive binding only; never rewrites preregistration or experiment scores."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contact_sampling_v5_20261011'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    paths = [Path(__file__), ROOT / 'scripts/analyze_contact_repeats_20261011.py']
    paths += [OUT / name for name in ('versions.json', 'frozen_thresholds.json',
        'sampling_summary.json', 'repeatability_summary.json',
        'EXPERIMENT_AUDIT.md', 'EXPERIMENT_AUDIT.json')]
    batches = [OUT / 'raw/repeat_a', OUT / 'raw/repeat_b',
        ROOT / 'results/condition_feedback_probe_20261011/raw/val_native_s11_r0',
        ROOT / 'results/condition_feedback_probe_v2_20261011/raw/val_native_s11_r0']
    paths += [batch / name for batch in batches for name in
        ('metadata.json', 'metrics.json', 'receipt.json', 'lineage.json', 'trajectories.npz')]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    destination = OUT / 'analysis_provenance_posthoc.json'
    if destination.exists():
        assert json.loads(destination.read_text())['sha256'] == hashes, 'Archive input changed'
    else:
        record = dict(recorded_unix_time=time.time(), role='post-hoc publication provenance',
            preregistered=False, original_versions_unchanged=True,
            scores_and_thresholds_unchanged=True, independent_audit_verdict_unchanged=True,
            caution='This archive binding does not retroactively register the analysis or resolve contact measurement validity.',
            sha256=hashes)
        destination.write_text(json.dumps(record, indent=2), encoding='utf8')
    print('POSTHOC_CONTACT_ANALYSIS_BOUND', len(hashes), destination, flush=True)
