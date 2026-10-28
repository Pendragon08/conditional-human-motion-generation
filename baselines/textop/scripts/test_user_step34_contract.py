"""Isolated numerical/count/authorization guards; no motion, model or SDK use."""
import copy
import hashlib
import json
import random
from augmentation5368_protocol import COUNT,TOTAL_INTERVALS,duration_allocation,bounded_partition,validate_plan
from user_step34_contract import validate,BASE_TERMS,PRIVILEGED_TERMS,parameter_count

def run():
    lengths=duration_allocation()
    assert len(lengths)==5368 and sum(lengths)==5666400
    assert set(lengths)=={1055,1056} and lengths==duration_allocation()
    for count in [3,4,5]:
        for total in [132,133]:
            for seed in range(10):
                assert sum(bounded_partition(total,count,random.Random(seed)))==total
    text='a synthetic guard fixture';tid=hashlib.sha256(text.encode()).hexdigest();sid='fixture_train'
    clips=[]
    for index,intervals in enumerate(lengths):
        calls=(intervals+8)//8;dur=bounded_partition(calls,3,random.Random(index));cursor=0;commands=[]
        # Distinct synthetic labels, never an actual BABEL or motion read.
        for number,size in enumerate(dur):
            label=text+str(number);key=hashlib.sha256(label.encode()).hexdigest()
            commands.append(dict(text_id=key,text=label,source_family=sid,start_feature=cursor*8,
                end_feature_exclusive=min(intervals,(cursor+size)*8),start_primitive=cursor,
                end_primitive_exclusive=cursor+size));cursor+=size
        clips.append(dict(clip_index=index,clip_id='fixture'+str(index),generation_seed=index,source_split='train',
            initial_source=sid,actual_generated=False,fps=50,target_control_intervals=intervals,target_pose_frames=intervals+1,
            generated_future_pose_features=intervals+1,primitive_calls=calls,generated_tail_features_excluded=calls*8-intervals-1,
            initial_reference_features=[0,1],seed_pose_indices=[0,1,2],commands=commands))
    text_ids={c['text_id'] for c in clips[0]['commands']}
    assert validate_plan(clips,{sid},text_ids)
    mutations=[lambda x:x.pop(),lambda x:x[0].update(source_split='val'),lambda x:x[0].update(actual_generated=True),
        lambda x:x[0].update(target_control_intervals=100),lambda x:x[0].update(target_pose_frames=1),
        lambda x:x[0].update(primitive_calls=1),lambda x:x[0].update(generated_tail_features_excluded=x[0]['generated_tail_features_excluded']+1),
        lambda x:x[0].update(initial_reference_features=[0,2]),lambda x:x[0].update(initial_source='VAL'),
        lambda x:x[0]['commands'][0].update(text='mismatch'),lambda x:x[0]['commands'][1].update(start_feature=3),
        lambda x:x[1].update(generation_seed=0)]
    for change in mutations:
        broken=copy.deepcopy(clips);change(broken)
        try:validate_plan(broken,{sid},text_ids)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('A broken augmentation plan was accepted')
    policy=dict(method='one_stage_PPO',dagger_or_distillation=False,actor_terms=[list(x) for x in BASE_TERMS],
        critic_terms=[list(x) for x in BASE_TERMS+PRIVILEGED_TERMS],actor_observation_dim=431,critic_observation_dim=557,
        future_reference_frames=5,dof=29,action_dim=29,actor_hidden=[2048,1024,512],critic_hidden=[2048,1024,512],
        activation='elu',simulation_hz=200,policy_hz=50,rollout_steps_per_env=24,num_envs=8192,learning_rate=.001,
        minibatches=4,ppo_epochs=5,clip_param=.2,fresh_actor_critic_and_Adam_initialization=True,
        old_privileged_teacher_training_resumed=False,training_authorized=False,accepted_mixed_dataset=None,
        real_RL_witnesses_accepted=False,new_RL_training_updates=0,pfail_conditional_regularization=False)
    assert validate(policy)
    changes=[dict(actor_observation_dim=557),dict(future_reference_frames=4),dict(policy_hz=200),
        dict(actor_hidden=[512,256,128]),dict(dagger_or_distillation=True),dict(training_authorized=True),
        dict(old_privileged_teacher_training_resumed=True),dict(pfail_conditional_regularization=True)]
    for change in changes:
        broken=copy.deepcopy(policy);broken.update(change)
        try:validate(broken)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('A broken one-stage PPO contract was accepted')
    return dict(status='USER_STEP34_ISOLATED_CONTRACT_VERIFIED',negative_cases=len(mutations)+len(changes),
        balanced_duration_exact=True,real_terminal_pose_required=True,model_loads=0,motion_arrays_read=0,
        actual_generated_clips=0,actual_RL_updates=0,parameters=parameter_count())

if __name__=='__main__':print(json.dumps(run()))
