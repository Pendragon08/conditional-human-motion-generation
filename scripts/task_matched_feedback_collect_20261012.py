"""Prospective task-matched C1 epoch; frozen prior adapters remain untouched."""
import sys
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError('Optimized Python disables inherited integrity gates')
previous = Path(__file__).with_name('controller_state_collect_20261012.py')
adapter = previous.read_text()
ending = "exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending, 1)[0], str(previous), 'exec'), globals())

old = "cache={path:np.load(path,allow_pickle=True).item()['data_dict'] for path in {r['cache_file'] for r in cases}}\n"
new = '''# Hash the exact bytes supplied to np.load; external input hashes frozen
# before this epoch. Lazy NPZ replay binding is checked separately downstream.
import io
from child_input_guard_20261012 import sha as actual_input_sha
from physics_guided_com_lineage_v24_20261012 import object_hash
input_versions=json.loads(Path(a.execution_versions).read_text())
input_request=json.loads(Path(a.request_file).read_text())
dataset_loads={}
def bound_dataset_load(path,**kwargs):
    resolved=str(Path(path).resolve())
    expected=input_versions['external_dataset_files'][resolved]
    data=Path(path).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if digest!=expected:raise RuntimeError('Dataset bytes differ from preregistration')
    dataset_loads[resolved]=digest
    return np.load(io.BytesIO(data),**kwargs)
cache={path:bound_dataset_load(path,allow_pickle=True).item()['data_dict'] for path in {r['cache_file'] for r in cases}}
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = "mean=np.load(manifest['mean']); std=np.load(manifest['std'])"
assert source.count(old) == 1
source = source.replace(old, "mean=bound_dataset_load(manifest['mean']); std=bound_dataset_load(manifest['std'])")
old = "correction=json.loads(Path(a.correction_file).read_text()) if a.correction_file else None\n"
new = old + '''if correction is not None:
    if object_hash(correction)!=input_request.get('correction'):
        raise RuntimeError('Actually loaded correction differs from bound request')
    feedback_source=Path(correction['feedback_source'])
    for name,digest in input_request['feedback_parent'].items():
        if actual_input_sha(feedback_source/name)!=digest:raise RuntimeError('Feedback parent changed')
elif 'correction' in input_request:
    raise RuntimeError('Bound correction was not loaded')
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = "        assert loaded_weights==expected_weights, 'Published checkpoints were not actually loaded'\n"
new = old + '''        if dataset_loads!=input_versions['external_dataset_files']:
            raise RuntimeError('Expected dataset bytes were not all consumed')
        if correction is not None:
            if object_hash(json.loads(Path(a.correction_file).read_text()))!=input_request['correction']:
                raise RuntimeError('Correction changed during execution')
            for name,digest in input_request['feedback_parent'].items():
                if actual_input_sha(feedback_source/name)!=digest:raise RuntimeError('Feedback parent changed during execution')
        for path,digest in dataset_loads.items():
            if actual_input_sha(Path(path))!=digest:raise RuntimeError('Dataset file changed during execution')
        with (out/'generation_input_binding.json').open('x') as handle:
            json.dump(dict(exact_loaded_dataset_bytes=dataset_loads,
                loaded_correction_object_sha256=object_hash(correction) if correction is not None else None,
                request_sha256=actual_input_sha(a.request_file),versions_sha256=actual_input_sha(a.execution_versions),
                scope='prospective generation/replay input binding; not retroactive certification'),handle,indent=2)
'''
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
