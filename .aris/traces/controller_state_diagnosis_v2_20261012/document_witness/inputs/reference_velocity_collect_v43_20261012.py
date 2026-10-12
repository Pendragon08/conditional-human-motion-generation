"""Diagnostic amendment: velocity uses successive applied reference positions.

Not a novel generator/refinement method. Preserve historical V42 and V28.
"""
from pathlib import Path

previous=Path(__file__).with_name('reference_observation_collect_v42_20261012.py')
adapter=previous.read_text()
ending="exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending,1)[0],str(previous),'exec'),globals())
old='    original_gen_state=task._get_state_from_gen_cache\n'
new='''    legacy_gen_state=task._get_state_from_gen_cache
    canonical_reference={}
    def consistent_gen_state(motion_ids):
        result=legacy_gen_state(motion_ids)
        current=result['rg_pos']
        frame=store['provided_frame']
        if not canonical_reference:
            assert frame==0
            previous=current.clone()
        elif frame==canonical_reference['frame']:
            assert torch.equal(current,canonical_reference['current']), 'Repeated frame changed reference'
            previous=canonical_reference['previous']
        else:
            assert frame==canonical_reference['frame']+1, 'Reference clock skipped'
            previous=canonical_reference['current']
        # First frame has no earlier applied reference: use zero velocity.
        # Subsequent velocities use the very positions supplied to the tracker,
        # rather than the generator's reconstructed context predecessor.
        result['body_vel']=(current-previous)/task.s_dt
        task.prev_ref_body_pos=previous.clone()
        canonical_reference.update(frame=frame,current=current.clone(),previous=previous.clone())
        return result
    task._get_state_from_gen_cache=consistent_gen_state
    original_gen_state=task._get_state_from_gen_cache
'''
assert source.count(old)==1
source=source.replace(old,new)
old="    metadata['observation_trace_scope']='Raw task obs before action; exact returned reference velocity and previous pose, not normalized policy observation'\n"
new=old+"    metadata['reference_velocity_rule']='Successive applied canonical positions / s_dt; first frame zero; repeated query preserves previous; diagnostic amendment, not innovation'\n"
assert source.count(old)==1
source=source.replace(old,new)
exec(compile(source,str(Path(__file__)),'exec'),globals())
