"""Passive record of the exact reference velocity and raw task observation.

Extends calibrated V28 without changing its generator, controller or metrics.
New output is diagnostic only; licensed state arrays remain server-local.
"""
from pathlib import Path

previous = Path(__file__).with_name('contact_com_collect_v28_20261012.py')
adapter = previous.read_text()
ending = "exec(compile(source,str(Path(__file__)),'exec'),globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending, 1)[0], str(previous), 'exec'), globals())
old = '    generated={}\n'
new = '''    generated={}
    store['observation_trace']={key:[] for key in ('ref_position','ref_previous','ref_velocity','task_observation')}
    original_gen_state=task._get_state_from_gen_cache
    def traced_gen_state(motion_ids):
        result=original_gen_state(motion_ids)
        store['latest_reference_velocity']=result['body_vel'].detach().cpu().numpy().copy()
        store['latest_reference_previous']=task.prev_ref_body_pos.detach().cpu().numpy().copy()
        return result
    task._get_state_from_gen_cache=traced_gen_state
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = "        store['applied_ref']=task.ref_body_pos.detach().cpu().numpy().copy()\n"
new = old + '''        trace=store['observation_trace']
        trace['ref_position'].append(store['applied_ref'].copy())
        trace['ref_previous'].append(store['latest_reference_previous'].copy())
        trace['ref_velocity'].append(store['latest_reference_velocity'].copy())
        trace['task_observation'].append(task.obs_buf.detach().cpu().numpy().copy())
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = "    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))\n"
new = '''    observation_trace={key:np.stack(value) for key,value in store['observation_trace'].items()}
    assert all(value.shape[0]==store['steps'] and np.isfinite(value).all() for value in observation_trace.values())
    np.savez_compressed(out/'observation_trace.npz',**observation_trace)
    metadata['observation_trace_sha256']=hashlib.sha256((out/'observation_trace.npz').read_bytes()).hexdigest()
    metadata['observation_trace_scope']='Raw task obs before action; exact returned reference velocity and previous pose, not normalized policy observation'
''' + old
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
