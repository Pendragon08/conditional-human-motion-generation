"""Test export-buffer clearing against unchanged box state and sensor evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contact_force_fixture_v2_20261011'
OLD = ROOT / 'results/contact_force_fixture_20261011'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

arrays = {}
for group, root in [('original', OLD), ('cleared', OUT)]:
    for repeat in ['repeat_a', 'repeat_b']:
        with np.load(root/repeat/'fixture_arrays.npz', allow_pickle=False) as z:
            arrays[group, repeat] = {k: z[k].copy() for k in z.files}
repeat_diff = {k: float(np.max(np.abs(arrays['cleared','repeat_a'][k]-arrays['cleared','repeat_b'][k]))) for k in arrays['cleared','repeat_a']}
control_diff = {repeat: {k: float(np.max(np.abs(arrays['cleared',repeat][k]-arrays['original',repeat][k]))) for k in ['state','constraint_force_sensor']} for repeat in ['repeat_a','repeat_b']}
checks = [json.loads((OUT/r/'summary.json').read_text()) for r in ['repeat_a','repeat_b']]
result = dict(known_physics_checks_both_pass=all(x['status']=='PASS' for x in checks),
    repeat_maximum_differences=repeat_diff, original_state_and_sensor_differences=control_diff,
    state_and_sensor_unchanged=all(v == 0 for row in control_diff.values() for v in row.values()),
    net_contact_repeatable_at_1mN=repeat_diff['net_contact_force'] <= .001,
    human_measurement_validated=False, method_efficacy_test=False,
    thresholds_changed=False, overleaf_results_sync=False,
    source_sha256=sha(Path(__file__)),
    input_hashes={group+'/'+repeat:sha(root/repeat/'fixture_arrays.npz') for group,root in [('original',OLD),('cleared',OUT)] for repeat in ['repeat_a','repeat_b']})
result['box_buffer_hypothesis_supported']=result['known_physics_checks_both_pass'] and result['state_and_sensor_unchanged'] and result['net_contact_repeatable_at_1mN']
(OUT/'repeat_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
