"""Compare paired actual policy inputs, preserving all failures and raw units."""
import numpy as np
from verify_execution_lineage_v39_20261012 import sha, read


def compare(first, replay):
    ma,mb=read(first/'metadata.json'),read(replay/'metadata.json')
    assert ma['cases']==mb['cases'] and len(ma['cases'])==72
    assert ma['original_goal']==mb['original_goal']
    assert ma['initial_reset_gate_pass'] and mb['initial_reset_gate_pass']
    for path,meta in ((first,ma),(replay,mb)):
        assert sha(path/'observation_trace.npz')==meta['observation_trace_sha256']
    entries=[]
    with np.load(first/'observation_trace.npz',allow_pickle=False) as a,np.load(replay/'observation_trace.npz',allow_pickle=False) as b,\
         np.load(first/'trajectories.npz',allow_pickle=False) as ta,np.load(replay/'trajectories.npz',allow_pickle=False) as tb:
        keep=ta['valid'] & tb['valid']
        for frame in (0,1,29,58,59,60,61,90,119):
            valid=keep[frame]
            entries.append(dict(frame=frame,common_valid_inputs=int(valid.sum()),
                fields={name:dict(maximum_absolute_difference=float(np.max(np.abs(a[name][frame,valid]-b[name][frame,valid]))),
                                 equal=bool(np.array_equal(a[name][frame,valid],b[name][frame,valid])))
                        for name in a.files} if valid.any() else {}))
        assert np.array_equal(a['ref_position'],ta['reference'])
        assert np.array_equal(b['ref_position'],tb['reference'])
    return dict(status='COMPLETE_REFERENCE_OBSERVATION_DIAGNOSTIC',attempts=72,entries=entries,
        output_bindings={path.name:{name:sha(path/name) for name in
            ('metadata.json','metrics.json','trajectories.npz','observation_trace.npz','lineage.json')}
            for path in (first,replay)},source_sha256=sha(__import__('pathlib').Path(__file__)),
        thresholds_changed=False,targets_changed=False,historical_scores_changed=False,
        candidate_efficacy=False,innovation_confirmed=False,overleaf_results_sync=False,
        scope='Raw task observation and reference velocity comparison, not normalized policy input or causal proof')
