"""A sparse execution-conditioning pilot, not a full CLoSD prefix transplant.

Same text, original future foot goal, model, denoising budget and noise seed.
Only past sparse root/ankle hints differ: planned versus physically executed.
HumanML 263D inpainting and improved control are NOT claimed here.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/feedback_world_model_20261006'
OUT = ROOT/'results/feedback_world_model_20261006'
IDS = [46, 47, 48]
HUMAN_FRAMES = np.arange(1, 13)


def residual_hint(human_reference, robot_reference, observed_robot, weights):
    """Robot arrays already use the declared human axis convention."""
    root_delta = observed_robot[0]-robot_reference[0]
    result = [human_reference[0]+root_delta]
    for j in (1,2):
        relative_delta = (observed_robot[j]-observed_robot[0])-(robot_reference[j]-robot_reference[0])
        result.append(human_reference[j]+root_delta+relative_delta@weights[j])
    return np.asarray(result)


def prepare():
    import mujoco
    from r3_sim2sim_mujoco import Transfer, sha
    bridge = json.loads((OUT/'feedback_bridge_v2.json').read_text())
    assert bridge['sparse_bridge_gate_pass']
    pairing = json.loads((ART/'sources/mapping.json').read_text())
    mapping = {r['reference']:r['source_sample'] for r in pairing['entries']}
    source = np.load(ART/'sources/Kick_official_results.npy',allow_pickle=True).item()
    points = source['motion'].transpose(0,3,1,2)
    samples = [mapping[i] for i in IDS]
    original = source['hint'][samples].copy()
    predicted, actual = original.copy(), original.copy()
    data = np.load(ART/'sources/sparse_bridge_v2.npz')
    R, weights = data['axis_rotation'], data['anchored_delta_weights']
    t = Transfer('Kick','cuda:0')
    bodies = [mujoco.mj_name2id(t.model,mujoco.mjtObj.mjOBJ_BODY,name)
              for name in ('pelvis','left_ankle_roll_link','right_ankle_roll_link')]
    meta = []
    for n,i in enumerate(IDS):
        trajectory = np.load(ART/'data'/f'Kick_{i:03d}_nominal.npz')
        sampled_steps = []
        for f in HUMAN_FRAMES:
            robot_frame = int(f+19)
            # Match the official library's frame index / length convention;
            # nearest 50Hz recorded state, max resampling error0.01s.
            k = round(robot_frame*float(t.ref.a['_motion_lengths'][i])/
                      (int(t.ref.a['_motion_num_frames'][i])-1)/.02)
            if k >= len(trajectory['states']):
                raise RuntimeError(f'Reference{i} fell before feedback context; do not replace it with a survivor')
            s = trajectory['states'][k]
            t.data.qpos[:3],t.data.qpos[3:7] = s[:3],s[3:7]
            t.data.qpos[t.qadr] = s[13:40]
            mujoco.mj_forward(t.model,t.data)
            observed = t.data.xpos[bodies].copy()@R
            reference = t.ref.state(i,k*.02)
            t.data.qpos[:3],t.data.qpos[3:7] = reference['pos'],reference['quat']
            t.data.qpos[t.qadr] = reference['joints']
            mujoco.mj_forward(t.model,t.data)
            reference_points = t.data.xpos[bodies].copy()@R
            human_reference = points[samples[n],f,[0,7,8]]
            # Residual feedback is anchored to the original human reference:
            # h_fb = h_ref + A(r_executed-r_reference). It does not require
            # asserting an inverse retargeting solution. Root translation
            # retains exact original units/axis direction, not a fitted scale.
            converted = residual_hint(human_reference, reference_points, observed, weights)
            predicted[n,f,[0,7,8]] = points[samples[n],f,[0,7,8]]
            actual[n,f,[0,7,8]] = np.asarray(converted)
            sampled_steps.append(k)
        # Joint11 is the original foot goal; feedback on ankle8 is past-only.
        assert np.array_equal(predicted[n,:,11], original[n,:,11])
        assert np.array_equal(actual[n,:,11], original[n,:,11])
        meta.append({'reference':i,'human_source_sample':samples[n],
                     'actual_sample_steps':sampled_steps,
                     'trajectory_sha256':sha(ART/'data'/f'Kick_{i:03d}_nominal.npz')})
    assert not np.array_equal(predicted,actual)
    shuffled = original.copy()
    for n in range(3):
        shuffled[n,HUMAN_FRAMES[:,None],[0,7,8]] = actual[(n+1)%3,HUMAN_FRAMES[:,None],[0,7,8]]
    assert np.array_equal(shuffled[:,:,11],original[:,:,11])
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) < 3
    # Algebraic identity/equivariance fixtures are separate from task trials.
    fixture_reference = np.array([[.2,1.,.4],[.1,.2,.3],[.3,.2,.5]])
    assert np.max(np.abs(residual_hint(fixture_reference,fixture_reference,fixture_reference,weights)-fixture_reference)) < 1e-12
    translation = np.array([.1,.2,.3])
    assert np.allclose((translation@R)@R.T,translation,atol=1e-12)
    assert np.allclose(residual_hint(fixture_reference,fixture_reference,fixture_reference+translation,weights),
                       fixture_reference+translation,atol=1e-12)
    np.savez_compressed(ART/'sources/feedback_generation_hints_residual_v3.npz',
                        predicted=predicted, actual=actual, shuffled=shuffled,original=original,
                        source_samples=samples, lengths=source['lengths'][samples])
    result = {'status':'PASS','type':'generator_input_interface_only','reference_ids':IDS,
              'samples':meta,'text_identical':True,'future_original_foot_goal_bitwise_identical':True,
              'context_joints':[0,7,8],'context_frames':HUMAN_FRAMES.tolist(),
              'contexts_differ':True,'no_extra_robot_ik':True,'full_humanml_prefix':False,
              'feedback_source':'actual_frozen_policy_mujoco_execution',
              'source_sha256':sha(ART/'sources/feedback_generation_hints_residual_v3.npz'),
              'calibration_was_exploratory':True,
              'feedback_form':'human_reference + root_translation_residual + calibrated_relative_ankle_residual',
              'zero_residual_identity_fixture':True,'translation_axis_equivariance_fixture':True,
              'control_benefit_evaluated':False}
    (OUT/'feedback_generation_inputs_v3.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FEEDBACK_GENERATION_INPUT_PASS',json.dumps(result),flush=True)


def generate(arm, remaining):
    import sample.generate as official
    from data_loaders.get_data import get_dataset_loader
    hints = np.load(ART/'sources/feedback_generation_hints_residual_v3.npz')
    raw = hints[arm]
    mean = np.load('dataset/humanml_spatial_norm/Mean_raw.npy').reshape(22,3)
    std = np.load('dataset/humanml_spatial_norm/Std_raw.npy').reshape(22,3)
    normalized = np.where(np.any(raw != 0,axis=-1,keepdims=True),(raw-mean)/std,0)
    texts = ['A person stands and kicks with his right leg.']*len(raw)

    def collate(n_frames,dataset,task):
        assert n_frames==196 and dataset=='humanml' and task=='kick'
        return texts,normalized.reshape(len(raw),196,-1).copy()

    def loader(**kwargs):
        kwargs['hml_mode']='text_only'
        return get_dataset_loader(**kwargs)

    official.collate_all=collate
    official.get_dataset_loader=loader
    sys.argv=[sys.argv[0],*remaining]
    start=time.time()
    official.main()
    (OUT/f'feedback_generation_{arm}_timing.json').write_text(json.dumps({
        'arm':arm,'wall_s':time.time()-start,'generated_samples':len(raw),
        'online_realtime_control_test':False,'full_humanml_prefix':False})+'\n')


def analyze():
    from r3_sim2sim_mujoco import sha
    source=np.load(ART/'sources/feedback_generation_hints_residual_v3.npz')
    provenance=json.loads((OUT/'feedback_generation_inputs_v3.json').read_text())
    assert provenance['source_sha256']==sha(ART/'sources/feedback_generation_hints_residual_v3.npz')
    weights=np.load(ART/'sources/sparse_bridge_v2.npz')['anchored_delta_weights']
    fixture=np.array([[.2,1.,.4],[.1,.2,.3],[.3,.2,.5]])
    translation=np.array([.1,.2,.3])
    assert np.allclose(residual_hint(fixture,fixture,fixture,weights),fixture,atol=1e-12)
    assert np.allclose(residual_hint(fixture,fixture,fixture+translation,weights),fixture+translation,atol=1e-12)
    fixed=source['original']
    rows={}
    motions={}
    for arm in ('predicted','actual','shuffled'):
        path=ART/'generation_v3'/arm/'results.npy'
        data=np.load(path,allow_pickle=True).item()
        assert data['motion'].shape==(3,22,3,196)
        expected=source[arm]
        assert np.allclose(data['hint'],expected,atol=1e-5)
        motion=data['motion'].transpose(0,3,1,2)
        motions[arm]=motion
        mask=np.linalg.norm(fixed[:,:,11],axis=-1)>1e-6
        context_error=[]
        goal_error=[]
        actual_context_error=[]
        for i in range(3):
            goal_error.append(float(np.linalg.norm(motion[i,mask[i],11]-fixed[i,mask[i],11],axis=-1).mean()))
            context_error.append(float(np.linalg.norm(motion[i,HUMAN_FRAMES][:,[0,7,8]]-
                expected[i,HUMAN_FRAMES][:,[0,7,8]],axis=-1).mean()))
            actual_context_error.append(float(np.linalg.norm(motion[i,HUMAN_FRAMES][:,[0,7,8]]-
                source['actual'][i,HUMAN_FRAMES][:,[0,7,8]],axis=-1).mean()))
        rows[arm]={'finite':bool(np.isfinite(motion).all()),'generated_samples':3,
                   'goal_errors_m':goal_error,'mean_original_goal_error_m':float(np.mean(goal_error)),
                   'context_errors_m':context_error,'mean_condition_error_m':float(np.mean(context_error)),
                   'errors_to_same_actual_context_m':actual_context_error,
                   'mean_error_to_same_actual_context_m':float(np.mean(actual_context_error)),
                   'results_sha256':sha(path),
                   'timing':json.loads((OUT/f'feedback_generation_{arm}_timing.json').read_text())}
    for arm in rows:
        rows[arm]['mean_generated_position_change_vs_planned_m']=float(np.linalg.norm(motions[arm]-motions['predicted'],axis=-1).mean())
    result={'status':'COMPLETE','same_seed':42,'same_denoising_budget':True,
            'same_text_and_future_original_goal':True,'past_hints_are_executed_vs_planned':True,
            'results':rows,'full_humanml_prefix_verified':False,
            'saved_hints_match_inputs':True,'zero_residual_identity_and_translation_fixtures':True,
            'context_accuracy_means_are_condition_fitting_not_physical_execution_accuracy':True,
            'actual_robot_execution_after_regeneration_evaluated':False,
            'online_feedback_control_evaluated':False,'full_innovation_validated':False,
            'scope':'Sparse conditioning response experiment, not a control-success ablation'}
    (OUT/'feedback_generation_interface_v3.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FEEDBACK_GENERATION_INTERFACE_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--stage',choices=['prepare','generate','analyze'],required=True)
    p.add_argument('--arm',choices=['predicted','actual','shuffled'])
    a,remaining=p.parse_known_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='analyze':analyze()
    else:generate(a.arm,remaining)
