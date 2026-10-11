"""Independent repeat comparison of known box fixtures; no acceptance changes."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/'results/contact_force_fixture_20261011'
with np.load(root/'repeat_a/fixture_arrays.npz',allow_pickle=False) as a, np.load(root/'repeat_b/fixture_arrays.npz',allow_pickle=False) as b:
    comparisons={key:float(np.max(np.abs(a[key]-b[key]))) for key in a.files}
    per_fixture={key:np.max(np.abs(a[key]-b[key]),axis=(0,2)).tolist() for key in a.files}
    records=[json.loads((root/label/'summary.json').read_text()) for label in ('repeat_a','repeat_b')]
    report=dict(known_fixture_calibration_both_pass=all(x['status']=='PASS' for x in records),
        repeat_maximum_differences=comparisons,per_fixture_maximum_differences=per_fixture,
        net_contact_repeatable_at_1mN=comparisons['net_contact_force']<=.001,
        constraint_sensor_repeatable_at_1mN=comparisons['constraint_force_sensor']<=.001,
        state_repeatable=comparisons['state']==0,
        evaluation_scope='Actual box-plane simulation with known mass, frictionless slide and initial airborne state; not articulated humanoid contact ground truth.',
        method_efficacy_test=False,thresholds_changed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes={label:hashlib.sha256((root/label/'fixture_arrays.npz').read_bytes()).hexdigest() for label in ('repeat_a','repeat_b')})
(root/'repeat_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
