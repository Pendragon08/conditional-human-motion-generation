"""Freeze revised read-only analysis before any repair-arm execution."""
from pathlib import Path
import json
import hashlib
import time

root=Path(__file__).resolve().parents[1]
out=root/'results/support_reference_probe_v63_20261012'
assert not (out/'summary.json').exists()
assert all(not (out/'raw'/('val_'+arm+'_s11')).exists() for arm in ('unconstrained','endpoint_preserved'))
source=root/'scripts/analyze_support_reference_probe_v63_20261012.py'
record=dict(analysis_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),time=time.time(),
    repair_arms_executed=0,reason='Freeze independent-audit corrections before repair-arm outcomes',
    preceding_inline_ssh_invocation='Remote shell syntax error before Python execution; no simulation or GPU calls')
with (out/'analysis_plan_lock.json').open('x') as handle:json.dump(record,handle,indent=2)
print('SUPPORT_ANALYSIS_PLAN_FROZEN',record['analysis_source_sha256'])
