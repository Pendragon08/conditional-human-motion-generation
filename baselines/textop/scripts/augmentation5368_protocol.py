"""Deterministic TRAIN-only text streams and explicit interval/duration ledger.

No tensors, model imports or motion arrays. 31.48h is a requested generation
target, not a completed result and not an assertion of the author's20s schedule.
"""
import hashlib
import json
import random
from decimal import Decimal

COUNT=5368
FPS=50
TOTAL_INTERVALS=int(Decimal('31.48')*3600*FPS)
POLICY_SEED=20261029


def duration_allocation(count=COUNT,total=TOTAL_INTERVALS,seed=POLICY_SEED):
    assert type(count) is int and type(total) is int and count>0 and total>=count
    base,extra=divmod(total,count)
    order=list(range(count));random.Random(seed).shuffle(order)
    lengths=[base]*count
    for index in order[:extra]:lengths[index]+=1
    assert sum(lengths)==total and max(lengths)-min(lengths)<=1
    return lengths


def bounded_partition(total,count,rng,low=12,high=50):
    assert low*count<=total<=high*count
    values=[];left=total
    for remaining in range(count,0,-1):
        lo=max(low,left-high*(remaining-1));hi=min(high,left-low*(remaining-1))
        value=rng.randint(lo,hi);values.append(value);left-=value
    rng.shuffle(values)
    assert left==0 and sum(values)==total and all(low<=x<=high for x in values)
    return values


def build_plan(data):
    sources=sorted(data['sources'],key=lambda x:x['canonical_source'])
    texts=data['texts']
    assert len(sources)==5361 and len({x['canonical_source'] for x in sources})==5361
    assert len(texts)==4043 and all(hashlib.sha256(value.encode()).hexdigest()==key for key,value in texts.items())
    assert all(x['original_split']=='train' and x['mirrored'] is False and x['frames50']>=2 for x in sources)
    seed_sources=[x for x in sources if x['frames50']>=3]
    assert seed_sources
    source_ids={x['canonical_source'] for x in sources}
    samples=sorted(data['samples'],key=lambda x:x['sample_id'])
    assert len(samples)==34068
    assert all(x['original_split']=='train' and x['mirrored'] is False and x['canonical_source'] in source_ids and x['text_id'] in texts for x in samples)
    lengths=duration_allocation();clips=[]
    for index,intervals in enumerate(lengths):
        seed=int.from_bytes(hashlib.sha256(('textop-user-aug5368-v1:'+str(index)).encode()).digest()[:8],'big')%(2**63-1)
        rng=random.Random(seed)
        source=seed_sources[rng.randrange(len(seed_sources))]
        # The accepted codec decodes one pose per feature, not N+1 poses.
        # Sample a genuine terminal pose as well as every scored interval.
        primitive_calls=(intervals+1+7)//8
        command_count=rng.randint(3,5)
        durations=bounded_partition(primitive_calls,command_count,rng)
        commands=[];cursor=0;previous=None
        for number,duration in enumerate(durations):
            choices=[samples[rng.randrange(len(samples))] for _ in range(16)]
            row=next((x for x in choices if x['text_id']!=previous),None)
            assert row is not None
            start=cursor*8;end=min(intervals,(cursor+duration)*8)
            assert end>start
            commands.append(dict(command_index=number,text_id=row['text_id'],text=texts[row['text_id']],
                source_sample_id=row['sample_id'],source_family=row['canonical_source'],
                start_feature=start,end_feature_exclusive=end,
                start_primitive=cursor,end_primitive_exclusive=cursor+duration))
            cursor+=duration;previous=row['text_id']
        clips.append(dict(clip_index=index,clip_id='augmentation_'+str(index).zfill(5),generation_seed=seed,
            source_split='train',initial_source=source['canonical_source'],initial_source_path=source['output_file'],
            initial_source_sha256=source['output_sha256'],seed_pose_indices=[0,1,2],
            initial_reference_features=[0,1],first_generated_feature=2,
            target_control_intervals=intervals,target_pose_frames=intervals+1,fps=FPS,
            native_future_frames=8,primitive_calls=primitive_calls,
            generated_future_pose_features=intervals+1,
            generated_tail_features_excluded=primitive_calls*8-(intervals+1),
            commands=commands,actual_generated=False))
    validate_plan(clips,source_ids,set(texts))
    poses=TOTAL_INTERVALS+COUNT
    real_clips=[]
    for source in sources:
        intervals=source['frames50']-1
        for first in range(0,intervals,1999):
            last=min(first+1999,intervals)
            real_clips.append(dict(source_family=source['canonical_source'],source_path=source['output_file'],
                source_sha256=source['output_sha256'],source_split='train',mirrored=False,
                first_pose_index=first,last_pose_index_inclusive=last,control_intervals=last-first,
                fps=50,generated=False))
    assert sum(x['control_intervals'] for x in real_clips)==sum(x['frames50']-1 for x in sources)
    assert all(0<x['control_intervals']<2000 for x in real_clips)
    return dict(schema='textop.user_aug5368_private_scope.v1',clips=clips,original_RL_clips=real_clips,
        summary=dict(planned_clips=COUNT,planned_control_intervals=TOTAL_INTERVALS,planned_pose_frames=poses,
            target_duration_hours='31.48',duration_definition='sum(T_minus_1)/50; seed history and padding tails excluded',
            per_clip_interval_counts={str(n):lengths.count(n) for n in sorted(set(lengths))},
            fps=50,train_family_count=5361,train_text_count=4043,annotation_chunk_sampling_population=34068,
            eligible_three_pose_initial_seed_families=len(seed_sources),
            original_RL_clip_count=len(real_clips),original_RL_control_intervals=sum(x['control_intervals'] for x in real_clips),
            original_maximum_clip_control_seconds=39.98,
            original_VAL_families_used=0,train_internal_holdout_is_still_original_TRAIN=True,
            fixed_seed=POLICY_SEED,minimum_command_native_calls=12,maximum_command_native_calls=50,
            labels_per_stream=[3,4,5],text_sampling='uniform accepted TRAIN annotation-chunk rows with unequal phrase frequencies retained',
            initial_seed_history='two past reference features from three TRAIN seed poses, then only generated history',
            author_exact_text_schedule_or_duration_claimed=False,paper_reported_clips=5368,paper_reported_hours='31.48',
            paper_reported_text_stream_seconds=20,
            paper_20s_times_count_hours=str(Decimal(COUNT)*20/3600),
            user_target_duration_overrides_unresolved_paper20s_total_mismatch=True,
            canonical_qpos36_contact2_float32_payload_bytes=poses*38*4,
            optional_generated69D_float32_payload_bytes=poses*69*4,
            derived_body30_joint29_float32_reference_payload_bytes=poses*(2*29+30*(3+4+3+3))*4,
            private_clip_container_header_allowance_bytes=COUNT*4096,
            actual_generated_clips=0,actual_generated_hours=0,actual_RL_updates=0))


def validate_plan(clips,train_sources,text_ids):
    assert len(clips)==COUNT and [x['clip_index'] for x in clips]==list(range(COUNT))
    assert len({x['clip_id'] for x in clips})==COUNT and len({x['generation_seed'] for x in clips})==COUNT
    assert sum(x['target_control_intervals'] for x in clips)==TOTAL_INTERVALS
    for clip in clips:
        assert clip['source_split']=='train' and clip['initial_source'] in train_sources
        assert clip['actual_generated'] is False and clip['fps']==50
        intervals=clip['target_control_intervals']
        assert clip['target_pose_frames']==intervals+1
        assert clip['generated_future_pose_features']==intervals+1
        assert clip['primitive_calls']==(intervals+1+7)//8
        assert clip['generated_tail_features_excluded']==clip['primitive_calls']*8-(intervals+1)
        assert clip['initial_reference_features']==[0,1] and clip['seed_pose_indices']==[0,1,2]
        previous_end=0;previous_primitive=0;prior_text=None
        assert 3<=len(clip['commands'])<=5
        for command in clip['commands']:
            assert command['text_id'] in text_ids and command['source_family'] in train_sources
            assert hashlib.sha256(command['text'].encode()).hexdigest()==command['text_id']
            assert command['text_id']!=prior_text
            assert command['start_feature']==previous_end==command['start_primitive']*8
            assert command['start_primitive']==previous_primitive
            assert command['end_feature_exclusive']==min(intervals,command['end_primitive_exclusive']*8)
            assert 12<=command['end_primitive_exclusive']-command['start_primitive']<=50
            previous_end=command['end_feature_exclusive'];previous_primitive=command['end_primitive_exclusive'];prior_text=command['text_id']
        assert previous_end==intervals and previous_primitive==clip['primitive_calls']
    return True
