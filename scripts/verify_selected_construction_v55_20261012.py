"""Read-only reconstruction from execution-bound candidate outputs.

Do not trust a mutable selected JSON for construction-complete labels. Rebuild
selection, all labels and selected reference from bound raw data; compare with
the motion hash used by independent replay. No historical certificate rewrite.
"""
import json
from pathlib import Path
import time
import numpy as np
from physics_guided_core import rank_candidate
from physics_guided_lineage import motion_hash
from verify_execution_lineage_v39_20261012 import read, sha, guard_batch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
OUT=ROOT/'results/selected_construction_binding_v55_20261012'

def main():
    assert not OUT.exists(), 'Preserve prior attempt; no blind rerun'
    assert np.__version__=='1.23.5'
    versions=read(BASE/'versions.json')
    for name in ('scripts/physics_guided_core.py','scripts/physics_guided_queue.py'):
        assert sha(ROOT/name)==versions['files'][name]
    config=read(ROOT/'configs/physics_guided_20261011.json')
    assert sha(ROOT/'configs/physics_guided_20261011.json')==versions['files']['configs/physics_guided_20261011.json']
    assert sha(BASE/'source_manifest.json')==versions['source_manifest_sha256']
    assert sha(BASE/'frozen_thresholds.json')==versions['thresholds_sha256']
    inputs={}; entries=[]
    for seed in (11,12,13):
        for arm in ('feedback','random'):
            paths=[BASE/'raw'/('val_shared_s%d_r0'%seed)]+[BASE/'raw'/('val_%s_s%d_r%d'%(arm,seed,r)) for r in (1,2)]
            packages=[]
            for path in paths:
                expected=read(BASE/'requests'/(path.name+'.json'))
                guard_batch(path,expected)
                assert expected['versions']==sha(BASE/'versions.json')
                with np.load(path/'trajectories.npz',allow_pickle=False) as z:
                    arrays={k:z[k] for k in z.files}
                packages.append((read(path/'metadata.json'),read(path/'metrics.json'),arrays))
                for name in ('metadata.json','metrics.json','trajectories.npz','lineage.json'):
                    inputs[str((path/name).relative_to(ROOT))]=sha(path/name)
            selection=BASE/'selected'/('%s_s%d.json'%(arm,seed))
            saved=read(selection); selected=BASE/'selected'/('%s_s%d.npz'%(arm,seed))
            replay=BASE/'raw'/('val_%s_replay_s%d'%(arm,seed))
            replay_request=read(BASE/'requests'/(replay.name+'.json'))
            guard_batch(replay,replay_request)
            assert motion_hash(selected)==replay_request['replay_motion']
            refs=np.zeros((config['episode_steps_30hz'],72,24,3),np.float32)
            rebuilt=[]; construction_complete=[]
            for i in range(72):
                k=min(range(3),key=lambda j:rank_candidate(packages[j][1][i],config['base_thresholds']))
                row=packages[k][1][i]; a=packages[k][2]
                chosen=a['actual'][a['valid'][:,i],i]
                assert len(chosen)>0
                length=min(len(chosen),len(refs))
                refs[:length,i]=chosen[:length];refs[length:,i]=chosen[length-1]
                rebuilt.append(dict(case_id=row['case_id'],candidate=k,construction_complete=row['completed'],path=str(paths[k]),metrics=row))
                construction_complete.append(bool(row['completed']))
            assert saved==dict(choices=rebuilt,source_type='physics_refined',original_goal_sha256=packages[0][0]['original_goal_sha256'],construction_candidates=3)
            with np.load(selected,allow_pickle=False) as z:
                assert np.array_equal(z['reference'],refs)
                assert np.array_equal(z['original_goal'],np.array(packages[0][0]['original_goal'],np.float32))
            replay_rows=read(replay/'metrics.json')
            entries.append(dict(arm=arm,seed=seed,cases=72,construction_complete=sum(construction_complete),
                end_to_end_joint=sum(c and bool(r['joint_pass']) for c,r in zip(construction_complete,replay_rows)),
                selected_motion_execution_hash=replay_request['replay_motion'],exact_reference_reconstruction=True,
                selection_labels_reconstructed_from_bound_raw=True))
            for path in (selection,selected,replay/'lineage.json',replay/'metrics.json'):
                inputs[str(path.relative_to(ROOT))]=sha(path)
    result=dict(status='CURRENT_SELECTION_EXACTLY_RECONSTRUCTED_FROM_BOUND_RAW',entries=entries,input_hashes=inputs,
        source_sha256=sha(Path(__file__)),generation_calls=0,simulation_calls=0,original_files_changed=False,
        warning='V54 selected JSON execution-binding hole remains historical; reconstruction independently checks current labels and selected motion. Not retrospective proof of an original selection-JSON execution hash.',
        evaluation_type='simulation_only; synthetic_proxy tracking',innovation_confirmed=False,time=time.time())
    OUT.mkdir();path=OUT/('certificate_%d.json'%time.time_ns());path.write_text(json.dumps(result,indent=2))
    print('SELECTED_CONSTRUCTION_RECONSTRUCTION_COMPLETE',path,flush=True)

if __name__=='__main__':main()
