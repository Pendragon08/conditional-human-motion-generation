"""V24 metric/physics unchanged; retain COM but omit dense diagnostic manifolds."""
from pathlib import Path

previous=Path(__file__).with_name('contact_com_collect_v24_20261012.py')
adapter=previous.read_text()
ending="exec(compile(source,str(Path(__file__)),'exec'),globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending,1)[0],str(previous),'exec'),globals())
old="        read_contacts(task.gym,task.envs,store['steps'],store['native_contact_records'],store['native_contact_errors'],store['native_contact_dtypes'])\n"
assert source.count(old)==1
source=source.replace(old,'')
old="    metadata['native_contact_scope']='Read-only manifolds at the same completed physics step as state snapshots; diagnostic only, not a replacement metric'\n"
assert source.count(old)==1
source=source.replace(old,"    metadata['native_contact_scope']='Dense manifold diagnostics disabled after calibration; native body COM and frozen surface-proxy metrics retained'\n"
                         "    metadata['dense_manifold_recording']=False\n")
exec(compile(source,str(Path(__file__)),'exec'),globals())
