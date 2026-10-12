"""Prepared prospective adapter; historical V43 and results stay unchanged.

Requires --execution-versions explicitly. Not witnessed or ready for GPU runs.
"""
from pathlib import Path

previous = Path(__file__).with_name('reference_velocity_collect_v43_20261012.py')
adapter = previous.read_text()
ending = "exec(compile(source,str(Path(__file__)),'exec'),globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending, 1)[0], str(previous), 'exec'), globals())

old = "p.add_argument('--request-file')\n"
assert source.count(old) == 1
source = source.replace(old, old + "p.add_argument('--execution-versions', required=True)\n")
old = "replay=np.load(a.replay_file,allow_pickle=False) if a.replay_file else None\n"
new = old + """from child_input_guard_20261012 import verify as verify_loaded_inputs
assert a.request_file, 'Child binding requires an explicit frozen request'
child_request=json.loads(Path(a.request_file).read_text())
assert (child_request['mode'],child_request['seed'],child_request['round'])==(a.mode,a.seed,a.round)
child_versions=json.loads(Path(a.execution_versions).read_text())
assert {'scripts/child_input_guard_20261012.py','scripts/reference_child_binding_collect_20261012.py'}<=set(child_versions['files']), 'New guard and entry must be version-bound'
child_binding=verify_loaded_inputs(root,a.execution_versions,child_request,replay)
with (out/'loaded_input_binding.json').open('x',encoding='utf8') as handle:
    json.dump(child_binding,handle,indent=2)
"""
assert source.count(old) == 1
source = source.replace(old, new)

old = "        request=json.loads(Path(a.request_file).read_text())\n"
new = old + """        assert request==child_request, 'Execution request changed during run'
        verify_loaded_inputs(root,a.execution_versions,request,replay)
"""
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
