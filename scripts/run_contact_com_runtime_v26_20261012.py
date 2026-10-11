"""Use collector's numerical runtime for unchanged V24 validation; no resimulation on check."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import run_reset_alignment_checks_v4 as runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contact_com_runtime_v26_20261012'
V24 = ROOT / 'results/contact_com_validation_v24_20261012'
DOC = ROOT / 'docs/contact_com_runtime_v26_20261012.md'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def frozen_outputs():
    names = ['versions.json', 'source_manifest.json', 'frozen_thresholds.json']
    for folder in ('requests', 'raw'):
        for path in (V24 / folder).rglob('*'):
            if path.is_file() and path.name in ('metadata.json', 'metrics.json', 'trajectories.npz',
                    'sensor_diagnostic.npz', 'native_contacts.json', 'receipt.json', 'lineage.json'):
                names.append(str(path.relative_to(V24)))
            elif folder == 'requests' and path.is_file():
                names.append(str(path.relative_to(V24)))
    return {name: runtime.sha(V24 / name) for name in names}


def main(check, inside):
    if not inside:
        env = runtime.environment()
        os.execve(str(runtime.PYTHON), [str(runtime.PYTHON), '-u', str(Path(__file__).resolve()),
                  '--check' if check else '--run', '--inside'], env)
    import numpy as np
    assert np.__version__ == '1.23.5'
    assert str(runtime.RUNTIME / 'envs/closd-r3/site-packages') in np.__file__
    assert shutil.disk_usage('/data').free >= 20 * 1024 ** 3
    OUT.mkdir(exist_ok=True)
    bindings = {str(Path(__file__).relative_to(ROOT)): runtime.sha(Path(__file__)),
                str(DOC.relative_to(ROOT)): runtime.sha(DOC)}
    if not (OUT / 'versions.json').exists():
        old = json.loads((V24 / 'state.json').read_text())
        assert old['status'] == 'FAILED_RETAINED'
        assert 'float64 roundoff' in old['error']
        for name in ('state.json', 'budget_ledger.json', 'versions.json'):
            shutil.copyfile(V24 / name, OUT / ('v24_before_runtime_repair_' + name))
        diagnostic = ROOT / 'results/metric_runtime_diagnostic_v25_20261012/diagnostic.json'
        d = json.loads(diagnostic.read_text())
        assert d['collector_overlay']['maximum_mean_slip_difference_m_s'] == 0
        save(OUT / 'versions.json', dict(files=bindings, v24_versions_sha256=runtime.sha(V24/'versions.json'),
             runtime_diagnostic_sha256=runtime.sha(diagnostic), numpy_version=np.__version__, numpy_path=np.__file__,
             qualification='Verifier runtime amendment only. V24 collector, requests, original goals, thresholds, models and saved outputs unchanged. Old failure snapshot retained; cached first batch reused, not resimulated.',
             innovation_efficacy=False, overleaf_results_sync=False))
    version = json.loads((OUT / 'versions.json').read_text())
    assert version['files'] == bindings
    assert version['v24_versions_sha256'] == runtime.sha(V24 / 'versions.json')
    if not check:
        witness = json.loads((OUT / 'runtime_check.json').read_text())
        assert witness['status'] == 'PASS' and witness['files'] == bindings
    before = frozen_outputs()
    ledger_before = json.loads((V24 / 'budget_ledger.json').read_text())
    start = time.monotonic()
    import run_contact_com_validation_v24_20261012 as queue
    try:
        queue.main(check)
        assert all(runtime.sha(V24 / name) == digest for name, digest in before.items())
        ledger_after = json.loads((V24 / 'budget_ledger.json').read_text())
        if check:
            assert ledger_before == ledger_after, 'Check must reuse completed 72, no new GPU call'
        save(OUT / ('runtime_check.json' if check else 'run_receipt.json'),
             dict(status='PASS', files=bindings, seconds=time.monotonic()-start,
                  numpy_version=np.__version__, numpy_path=np.__file__,
                  preserved_outputs=before, preserved_outputs_unchanged=True,
                  before_batch_count=len(ledger_before), after_batch_count=len(ledger_after),
                  cached_first_batch_not_resimulated=True, thresholds_unchanged=True,
                  innovation_efficacy=False, overleaf_results_sync=False))
        print('CONTACT_COM_RUNTIME_DOCUMENT_CHECK_PASS' if check else 'CONTACT_COM_RUNTIME_FULL_VALIDATION_COMPLETE', flush=True)
    except Exception as error:
        save(OUT / ('check_failure.json' if check else 'run_failure.json'), dict(error=repr(error), time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    main(args.check, args.inside)
