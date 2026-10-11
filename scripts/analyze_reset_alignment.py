"""Read-only quality/coordinate analysis of the separate reset repair fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(out):
    state=json.loads((out/'state.json').read_text())
    assert state['status']=='COLLECTION_COMPLETE_REQUIRES_INDEPENDENT_CHECK'
    records=[]; errors=[]; metadata={}; quality={}
    for seed in (11,12,13):
        for mode in ('planned','native'):
            label=mode+'_s'+str(seed); directory=out/label
            meta=json.loads((directory/'metadata.json').read_text())
            receipt=json.loads((directory/'receipt.json').read_text())
            assert receipt['cases']==12
            metadata[label]=meta
            values=meta['manual_reset_vs_first_action_state_max_difference']
            valid=(meta['initial_actual_reference_xy_max_m']<.001 and
                   all(v<1e-5 for v in values.values()) and
                   all(v==0 for v in meta['progress_before_first_action']) and
                   all(v==0 for v in meta['motion_start_times_before_first_action']) and
                   np.max(np.abs(meta['global_offset_before_first_action']))==0)
            if not valid: errors.append(label+': initial state differs after player reset')
            records.append(dict(label=label,cases=receipt['cases'],completed=receipt['completed'],
                joint_pass=receipt['joint_pass'],initial_xy_max_m=meta['initial_actual_reference_xy_max_m'],
                full_state_max_difference=values,full_reset_calls=meta['full_reset_calls_before_first_action'],
                top_level_seed=meta['top_level_seed'],learning_seed=meta['learning_seed'],
                task_config_seed=meta['task_config_seed'],generator_config_seed=meta['generator_config_seed'],
                warmup_diffusion_calls=meta['upstream_warmup_diffusion_calls'],
                total_diffusion_calls=meta['total_diffusion_calls_including_warmup'],
                receipt_sha256=hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256((directory/'metadata.json').read_bytes()).hexdigest(),
                metrics_sha256=hashlib.sha256((directory/'metrics.json').read_bytes()).hexdigest()))
    first=metadata['planned_s11']; anchor=np.array(first['original_goal'])
    cross=[]
    for label,meta in metadata.items():
        delta=float(np.max(np.abs(np.array(meta['original_goal'])-anchor)))
        heading=float(np.max(np.abs(np.array(meta['original_heading'])-np.array(first['original_heading']))))
        root_delta=float(np.max(np.abs(np.array(meta['actual_initial_root'])-np.array(first['actual_initial_root']))))
        if max(delta,heading,root_delta)>1e-5: errors.append(label+': conditions vary across generation seeds or modes')
        cross.append(dict(label=label,target_max_difference_m=delta,heading_max_difference_rad=heading,initial_root_max_difference_m=root_delta))
    offset_checks=[]
    for index in range(0,12,3):
        group=first['cases'][index:index+3]
        assert len({r['source_id'] for r in group})==1
        relative=anchor[index:index+3]-anchor[index+1]
        expected=np.array([[-.15,0,0],[0,0,0],[.15,0,0]])
        error=float(np.max(np.abs(relative-expected)))
        if error>1e-4: errors.append(group[0]['source_id']+': offsets not controlled in world coordinates')
        offset_checks.append(dict(source_id=group[0]['source_id'],max_absolute_offset_error_m=error))
    for mode in ('planned','native'):
        label='full_val_'+mode+'_s11'; directory=out/label
        meta=json.loads((directory/'metadata.json').read_text()); rows=json.loads((directory/'metrics.json').read_text())
        receipt=json.loads((directory/'receipt.json').read_text())
        assert len(rows)==72 and receipt['cases']==72
        if meta['initial_actual_reference_xy_max_m']>=.001: errors.append(label+': anchor mismatch')
        if any(v>=1e-5 for v in meta['manual_reset_vs_first_action_state_max_difference'].values()): errors.append(label+': initial full-state mismatch')
        quality[label]=dict(attempts=72,completed=sum(r['completed'] for r in rows),
            condition_pass=sum(r['condition_pass'] for r in rows),physical_pass=sum(r['physical_pass'] for r in rows),
            tracking_pass=sum(r['tracking_pass'] for r in rows),joint_pass=sum(r['joint_pass'] for r in rows),
            endpoint_error_mean_m=float(np.mean([r['endpoint_error_m'] for r in rows])),
            slip_proxy_median=float(np.median([r['surface_slip_proxy_fraction'] for r in rows if r['surface_slip_proxy_fraction'] is not None])),
            foot_asset_sha256=meta['foot_asset_sha256'],
            metrics_sha256=hashlib.sha256((directory/'metrics.json').read_bytes()).hexdigest(),
            receipt_sha256=hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest())
    floor=all(r['joint_pass']==0 for r in quality.values())
    ledger=json.loads((out/'budget_ledger.json').read_text())
    old=json.loads((out.parent/'physics_guided_20261011/budget_ledger.json').read_text())
    result=dict(study='reset-aligned independent measurement diagnosis',attempts=216,
        coordinate_gate_pass=not errors,coordinate_errors=errors,initial_state_checks=records,
        cross_seed_and_mode_checks=cross,offset_checks=offset_checks,full_validation_quality=quality,
        joint_gate_floor=floor,quality_thresholds_changed=False,
        repair_gpu_hours=sum(r['seconds'] for r in ledger)/3600,
        combined_charged_gpu_hours=sum(r['seconds'] for r in ledger+old)/3600,
        next_stage=('STOP_MEASUREMENT_INVALID' if errors else 'STOP_NONDISCRIMINATING_JOINT_GATE' if floor else 'VALID_COORDINATES_NEW_CONTROLLED_COMPARISON_REQUIRED'),
        qualification='No efficacy experiment or downstream training in this repair. Do not pool with the historical run. Full-body initialization matches the existing tracker protocol, not a teleport to dataset joint poses. Physical proxies and thresholds still require construct-valid calibration.')
    (out/'measurement_repair_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('RESET_ALIGNMENT_ANALYSIS',json.dumps({k:result[k] for k in ('attempts','coordinate_gate_pass','joint_gate_floor','next_stage','combined_charged_gpu_hours')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(Path(p.parse_args().output))
