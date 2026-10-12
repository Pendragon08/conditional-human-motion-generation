"""Prospective fixed-reference checkpoint-state diagnosis, same CLoSD algorithm."""
from pathlib import Path

previous = Path(__file__).with_name('reference_child_binding_collect_20261012.py')
adapter = previous.read_text()
ending = "exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending, 1)[0], str(previous), 'exec'), globals())
old = "p.add_argument('--execution-versions', required=True)\n"
assert source.count(old) == 1
source = source.replace(old, old + "p.add_argument('--controller-state', required=True, choices=['multitask','t2m','unfinetuned'])\n")
old = "'exp_name=CLoSD_multitask_finetune'"
assert source.count(old) == 1
source = source.replace(old, "'exp_name='+state_spec['exp_name']")
old = "store={}\n"
new = '''# Bind exact checkpoint bytes supplied to torch.load, not only path metadata.
import io
import subprocess
from child_input_guard_20261012 import sha as binding_sha
assert 'scripts/controller_state_collect_20261012.py' in child_versions['files']
state_spec=child_versions['controller_states'][a.controller_state]
assert child_request['controller_state']==a.controller_state
assert child_request['controller_weights_sha256']==state_spec['sha256']
upstream=str(entry.SOURCE)
actual_commit=subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()
assert actual_commit==child_versions['upstream_commit']
assert subprocess.check_output(['git','-C',upstream,'diff','--name-only','HEAD','--','closd'],text=True)==''
expected_weights={str((root/state_spec['path']).resolve()):state_spec['sha256'],
    str((root/child_versions['generator_path']).resolve()):child_versions['generator_weights_sha256']}
original_torch_load=torch.load
loaded_weights={}
def bound_torch_load(file,*args,**kwargs):
    if isinstance(file,(str,Path)):
        resolved=str(Path(file).resolve())
        if resolved in expected_weights:
            data=Path(file).read_bytes()
            digest=hashlib.sha256(data).hexdigest()
            assert digest==expected_weights[resolved], 'Checkpoint bytes changed'
            loaded_weights[resolved]=digest
            return original_torch_load(io.BytesIO(data),*args,**kwargs)
        if Path(file).name=='Humanoid.pth':
            raise RuntimeError('Unexpected tracking checkpoint')
    return original_torch_load(file,*args,**kwargs)
torch.load=bound_torch_load
store={}
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = "        verify_loaded_inputs(root,a.execution_versions,request,replay)\n"
new = old + '''        assert loaded_weights==expected_weights, 'Published checkpoints were not actually loaded'
        assert subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==actual_commit
        assert subprocess.check_output(['git','-C',upstream,'diff','--name-only','HEAD','--','closd'],text=True)==''
        with (out/'loaded_checkpoint_binding.json').open('x') as handle:
            json.dump(dict(controller_state=a.controller_state,exp_name=state_spec['exp_name'],
                actual_loaded_weight_bytes=loaded_weights,actual_upstream_commit=actual_commit,
                versions_sha256=binding_sha(a.execution_versions),request_sha256=binding_sha(a.request_file),
                same_algorithm_different_training_state=True),handle,indent=2)
'''
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
