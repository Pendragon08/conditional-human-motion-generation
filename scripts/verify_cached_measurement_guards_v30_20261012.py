"""Read-only recheck of completed V24/V27 caches; no simulator or resampling."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/fixed_reference_com_v27_20261012'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    packages = [ROOT/'results/contact_com_validation_v24_20261012/raw'/('val_'+mode+'_s'+str(seed))
                for seed in (11,12,13) for mode in ('planned','native')]
    packages += [OUT/'raw'/('val_'+ref+'_replay_s11') for ref in ('raw','refined')]
    records = []
    for path in packages:
        record = dict(path=str(path.relative_to(ROOT)), checks={})
        try:
            meta = json.loads((path/'metadata.json').read_text())
            receipt = json.loads((path/'receipt.json').read_text())
            lineage = json.loads((path/'lineage.json').read_text())
            checks = record['checks']
            checks['receipt_complete_72_cases'] = receipt['status']=='complete' and receipt['cases']==72 and len(meta['cases'])==72
            checks['completion_fence'] = meta['completion_fence_calls']>=120 and 'fetch_results(True)' in meta['completion_fence_protocol']
            checks['passive_sensor_configuration'] = meta['passive_sensor_bodies']==['L_Ankle','L_Toe','R_Ankle','R_Toe'] and meta['policy_observation_mode']==1 and not meta['sensor_metrics_used_for_selection']
            checks['sensor_file_binding'] = sha(path/'sensor_diagnostic.npz')==meta['sensor_diagnostic_sha256']
            checks['native_contact_file_binding'] = sha(path/'native_contacts.json')==meta['native_contacts_sha256']
            checks['cpu_export_gpu_physics'] = not meta['sim_params']['use_gpu_pipeline'] and meta['sim_params']['physx_use_gpu']
            checks['bridge_device_configuration'] = all(r['native_device']=='cpu' and r['policy_device'].startswith('cuda') for r in meta['bridge_buffers'].values())
            differences = meta['manual_reset_vs_first_action_state_max_difference']
            checks['seven_reset_fields_match'] = meta['initial_reset_gate_pass'] and len(differences)==7 and all(v==0 for v in differences.values())
            checks['native_body_coms_recorded'] = len(meta['native_body_properties'])==72 and all(len(row)==24 and all(len(b['com'])==3 and all(math.isfinite(x) for x in b['com']) for b in row) for row in meta['native_body_properties'])
            actual_hashes = {name:sha(path/name) for name in ('metadata.json','metrics.json','trajectories.npz')}
            checks['bound_output_hashes'] = actual_hashes==lineage['outputs'] and actual_hashes['trajectories.npz']==receipt['trajectories_sha256']
            record['output_hashes'] = actual_hashes
            with np.load(path/'trajectories.npz',allow_pickle=False) as arrays:
                checks['same_120_frame_clock'] = np.array_equal(arrays['reference_frame'],np.arange(120))
                checks['expected_attempt_shape'] = arrays['valid'].shape==(120,72) and arrays['terminated'].shape==(72,)
                checks['all_recorded_numeric_arrays_finite'] = all(np.isfinite(arrays[key]).all() for key in arrays.files)
        except Exception as error:
            record['error'] = repr(error)
        record['pass'] = bool(not record.get('error') and record['checks'] and all(record['checks'].values()))
        records.append(record)
    result = dict(status='PASS' if len(records)==8 and all(r['pass'] for r in records) else 'FAIL',
                  entries=records, source_sha256=sha(Path(__file__)),
                  scope='Read-only current-cache guards; does not change historical verifier or runtime. Not contact truth or efficacy.',
                  simulation_calls=0, generation_calls=0, thresholds_changed=False,
                  innovation_confirmed=False, overleaf_results_sync=False)
    path = OUT/'cached_guard_recheck_v30_20261012.json'
    if path.exists():
        assert json.loads(path.read_text())==result, 'Preserve prior certificate; do not overwrite differing evidence'
    else:
        path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='entries'}))
    assert result['status']=='PASS', records


if __name__=='__main__':
    main()
