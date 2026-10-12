"""Passive native manifolds on the corrected V43 reference interface.

Only enables a previously calibrated read-only exporter. No physics, generator,
policy, rewards, reference, thresholds or metrics change. Equality must be
checked against the execution-bound V51 replay before interpreting contacts.
"""
from pathlib import Path
previous=Path(__file__).with_name('reference_velocity_collect_v43_20261012.py')
adapter=previous.read_text()
ending="exec(compile(source,str(Path(__file__)),'exec'),globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending,1)[0],str(previous),'exec'),globals())
old="        store['pending']={\n"
assert source.count(old)==1
source=source.replace(old,"        read_contacts(task.gym,task.envs,store['steps'],store['native_contact_records'],store['native_contact_errors'],store['native_contact_dtypes'])\n"+old)
old="    metadata['dense_manifold_recording']=False\n"
assert source.count(old)==1
source=source.replace(old,"    metadata['dense_manifold_recording']=True\n")
old="    metadata['native_contact_scope']='Dense manifold diagnostics disabled after calibration; native body COM and frozen surface-proxy metrics retained'\n"
assert source.count(old)==1
source=source.replace(old,"    metadata['native_contact_scope']='V56 passive manifolds at the same completed physics step; simulator contact observations only, not real contact ground truth or replacement scores'\n")
exec(compile(source,str(Path(__file__)),'exec'),globals())
