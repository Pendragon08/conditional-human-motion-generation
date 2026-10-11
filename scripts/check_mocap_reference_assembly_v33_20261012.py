"""Compile the new diagnostic adapter without invoking a simulator or model."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
adapter=ROOT/'scripts/mocap_reference_collect_v33_20261012.py'
ending="exec(compile(source,str(Path(__file__)),'exec'),globals())"
code=adapter.read_text(encoding='utf8')
assert code.rstrip().endswith(ending)
namespace={'__file__':str(adapter),'__name__':'mocap_assembly_only'}
exec(compile(code.rsplit(ending,1)[0],str(adapter),'exec'),namespace)
source=namespace['source']
compile(source,str(adapter),'exec')
checks=dict(single_original_reference_assignment=source.count('replay_world=dataset_sim.detach().clone()')==1,
    generated_reference_not_used_in_replay="if a.mode=='replay':\n            selected=replay_world" in source,
    external_replay_archive_branch_removed="replay_world=torch.as_tensor(replay['reference']" not in source,
    original_targets_built_before_reference_assignment=source.index('original_goal=np.stack(')<source.index('replay_world=dataset_sim.detach().clone()'),
    no_correction_allowed="assert a.correction_file is None" in source,
    dataset_reference_explicitly_saved_and_bound="np.savez_compressed(out/'dataset_reference.npz'" in source and "metadata['dataset_reference_sha256']" in source,
    corrected_com_metric_retained='from physics_guided_metric_v23_20261012 import foot_geometry, heading, metrics' in source,
    dense_manifold_recording_disabled="metadata['dense_manifold_recording']=False" in source,
    body_completion_fence_retained='fetch_results' in source,
    original_reference_not_physical_ground_truth="metadata['mocap_is_not_robot_physical_ground_truth']=True" in source)
paths=('scripts/mocap_reference_collect_v33_20261012.py','scripts/run_mocap_reference_control_v33_20261012.py',
    'scripts/physics_guided_run_v33_20261012.sh','docs/mocap_reference_control_v33_20261012.md',
    'scripts/check_mocap_reference_assembly_v33_20261012.py')
result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
    evaluation_type='adapter assembly/compilation only',simulation_calls=0,
    simulation_efficacy_tested=False,innovation_confirmed=False,
    assembled_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
out=ROOT/'results/mocap_reference_assembly_v33_20261012';out.mkdir(exist_ok=True)
(out/'assembly_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2));assert all(checks.values()),checks
