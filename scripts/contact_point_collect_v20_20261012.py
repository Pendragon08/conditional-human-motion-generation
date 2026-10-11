"""Passive humanoid native manifolds and COM probe, frozen V11 runtime."""
from pathlib import Path

previous = Path(__file__).with_name('contact_cpu_export_collect_v11.py')
adapter = previous.read_text()
ending = "exec(compile(source,str(Path(__file__)),'exec'),globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending, 1)[0], str(previous), 'exec'), globals())

patches = [
    ('from physics_guided_core import foot_geometry, heading, metrics\n',
     'from physics_guided_core import foot_geometry, heading, metrics\n'
     'from native_contact_diagnostic import read_contacts, body_properties\n'),
    ('    sim_params=task.gym.get_sim_params(task.sim)\n',
     "    store['native_contact_records']=[]; store['native_contact_errors']=[]; store['native_contact_dtypes']=[]\n"
     "    metadata['native_body_properties']=body_properties(task.gym,task.envs,task.humanoid_handles)\n"
     "    assert len(metadata['native_body_properties'])==n\n"
     "    assert all(len(row)==b for row in metadata['native_body_properties'])\n"
     "    metadata['native_contact_scope']='Read-only manifolds at the same completed physics step as state snapshots; diagnostic only, not a replacement metric'\n"
     '    sim_params=task.gym.get_sim_params(task.sim)\n'),
    ("        store['pending']={\n",
     "        read_contacts(task.gym,task.envs,store['steps'],store['native_contact_records'],store['native_contact_errors'],store['native_contact_dtypes'])\n"
     "        store['pending']={\n"),
    ("    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))\n",
     "    native=dict(records=store['native_contact_records'],errors=store['native_contact_errors'],dtype_fields=store['native_contact_dtypes'],\n"
     "                thresholds_changed=False,metrics_replaced=False,innovation_confirmed=False)\n"
     "    (out/'native_contacts.json').write_text(json.dumps(native,indent=2))\n"
     "    metadata['native_contacts_sha256']=hashlib.sha256((out/'native_contacts.json').read_bytes()).hexdigest()\n"
     "    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))\n"),

]
for old, new in patches:
    assert source.count(old) == 1, 'Parent changed; review passive instrumentation'
    source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
