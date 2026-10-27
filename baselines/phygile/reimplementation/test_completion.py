"""Meaningful counterexamples for coordinate, rotation and promotion failures."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
from codec262 import G1Codec,rotvec_matrix,matrix_quat,quat_matrix,to6d,from6d
from curriculum_evaluator import PromotionPolicy,aggregate,sha


def rejects(f):
    try:f()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid input accepted')


def main():
    root=Path(__file__).resolve().parents[1]
    codec=G1Codec(root/'configs/g1_codec_skeleton.json')
    rng=np.random.default_rng(213)
    # Continuous roots include pitch/roll and a turn through pi; joint branches
    # cover both signs and origins with nonzero URDF rpy.
    n=80;t=np.arange(n)/50
    pos=np.stack([1+.15*t,-2+.03*t,.8+.01*np.sin(t)],-1)
    rot=np.stack([rotvec_matrix([0,0,2.8+.6*u])@rotvec_matrix([.2,.1+.05*np.sin(u),0]) for u in t])
    q=(codec.low+codec.high)/2+.15*np.sin(t[:,None]*np.arange(1,30))
    q=np.clip(q,codec.low+.02,codec.high-.02)
    native=np.concatenate([pos,matrix_quat(rot),np.zeros((n,6)),q,np.zeros((n,29))],-1)
    features,meta=codec.encode(native,50)
    rebuilt,_,_=codec.decode(features,meta)
    np.testing.assert_allclose(rebuilt[:,:3],native[:,:3],atol=3e-6)
    np.testing.assert_allclose(quat_matrix(rebuilt[:,3:7]),rot,atol=3e-6)
    np.testing.assert_allclose(rebuilt[:,13:42],q,atol=3e-6)
    initial=quat_matrix(meta['initial_root_quat'])
    assert abs(np.arctan2(initial[1,0],initial[0,0]))<1e-8
    assert abs(initial[2,0])>.01,'Pitch was accidentally erased'
    # Proper 6D SO(3), including near-pi rotations and invalid collinear vectors.
    rotations=rotvec_matrix(rng.normal(size=(100,3))*2)
    np.testing.assert_allclose(from6d(to6d(rotations)),rotations,atol=1e-12)
    rejects(lambda:from6d(np.zeros((3,6))))
    rejects(lambda:from6d(np.array([[1,0,0,2,0,0]])))
    bad=features.copy();bad[0,1]=np.nan
    rejects(lambda:codec.decode(bad,meta))
    rejects(lambda:codec.decode(features,{**meta,'skeleton_sha256':'wrong'}))
    rejects(lambda:codec.encode(native[:1],50))
    # Palm points must include their fixed URDF offsets, not wrist aliases.
    positions,_=codec.fk(native[:,:3],native[:,3:7],native[:,13:42])
    assert np.linalg.norm(positions['left_rubber_hand']-positions['left_wrist_yaw_link'],axis=-1).min()>.04
    # Scheduler guards: no low-error early-death promotion, timeout bypass or
    # cancellation of adjacent worsening. Exact zero errors remain legitimate.
    policy=PromotionPolicy(minimum_iterations=0,timeout_iterations=10)
    report={'valid':True,'mpjpe_m':.01,'mpjae_rad':.01,'minimum_file_success_fraction':0}
    assert not policy.decide(report,[],100)['promote']
    assert not policy.decide({**report,'valid':False},[],100)['promote']
    assert policy.decide({**report,'minimum_file_success_fraction':1},[],0)['promote']
    strict=PromotionPolicy(.01,.01,0,0,3,.03,0)
    high={'valid':True,'mpjpe_m':1,'mpjae_rad':1,'minimum_file_success_fraction':1}
    assert not strict.decide(high,[(1,1),(.9,.9),(1.1,1.1)],100)['promote']
    assert strict.decide(high,[(1,1),(1,1),(1,1)],100)['promote']
    disabled=PromotionPolicy(.01,.01,0,0,3,.03,0,False)
    assert not disabled.decide(high,[(1,1),(1,1),(1,1)],100)['promote']
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'motion';f.write_bytes(b'fixture, no performance evaluation')
        request={'files':[str(f)],'attempts_per_file':2,'checkpoint_sha256':'checkpoint','active_experts':1}
        record={'motion_file':str(f),'motion_sha256':sha(f),'checkpoint_sha256':'checkpoint',
          'episodes_completed':2,'num_envs':2,'episode_lengths':[1,1],'successes':0,'active_experts':1,
          'metrics':{'mpjpe_m_until_termination':0,'mpjae_rad_until_termination':0}}
        assert aggregate(request,[record])['valid']
        assert not aggregate(request,[])['valid']
        assert not aggregate(request,[{**record,'num_envs':1}])['valid']
        assert not aggregate(request,[{**record,'checkpoint_sha256':'different'}])['valid']
        assert not aggregate(request,[{**record,'active_experts':2}])['valid']
        rejects(lambda:aggregate(request,[record,record]))
    groups=['pose_roundtrip','canonical_pitch_roll','SO3_roundtrip','degenerate_6d',
        'nonfinite_features','skeleton_identity','sequence_length','palm_fixed_offset',
        'failed_episode_success_gate','invalid_evaluation_timeout','error_threshold_gate',
        'non_worsening_and_disabled_convergence','coverage_and_checkpoint_expert_identity']
    return {'status':'PASS','checks':len(groups),'check_groups':groups,'seed':213,
            'scope':'synthetic mathematical and counterexample checks; no learned-motion quality claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=main();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PHYGILE_COMPLETION_REGRESSION',json.dumps(result))
