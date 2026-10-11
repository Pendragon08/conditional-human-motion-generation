"""Replay original MoCap-derived HumanML references, no motion synthesis.

Only replaces the replay input in the frozen calibrated collector. This is a
control/measurement diagnostic, not another generation algorithm or innovation.
"""
from pathlib import Path

previous=Path(__file__).with_name('contact_com_collect_v28_20261012.py')
adapter=previous.read_text(encoding='utf8')
ending="exec(compile(source,str(Path(__file__)),'exec'),globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending,1)[0],str(previous),'exec'),globals())
start="    replay_world=None\n    if replay is not None:\n"
finish="    goal=torch.as_tensor(proposed_goal,device=task.device)\n"
begin=source.index(start);end=source.index(finish,begin)
replacement="""    assert a.mode=='replay' and a.replay_file is None
    assert a.correction_file is None, 'Original references and targets cannot be corrected in this diagnostic'
    replay_world=dataset_sim.detach().clone()
    assert replay_world.shape==(n,t,b,3)
    np.savez_compressed(out/'dataset_reference.npz',reference=replay_world.transpose(0,1).cpu().numpy(),
                        original_goal=original_goal)
"""
source=source[:begin]+replacement+source[end:]
old="    store.update(metadata=metadata,steps=0,finished=np.zeros(n,bool),terminated=np.zeros(n,bool),\n"
assert source.count(old)==1
extra="""    metadata['reference_origin']='Original licensed HumanML3D MoCap-derived data, converted by frozen representation handler; no generated continuation'
    metadata['dataset_reference_sha256']=hashlib.sha256((out/'dataset_reference.npz').read_bytes()).hexdigest()
    metadata['diagnostic_only']=True
    metadata['mocap_is_not_robot_physical_ground_truth']=True
"""
source=source.replace(old,extra+old)
exec(compile(source,str(Path(__file__)),'exec'),globals())
