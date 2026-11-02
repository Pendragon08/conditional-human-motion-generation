"""User-selected PPO5000 boundary; metadata guards and bound signals only."""
from vae100k_control_protocol import (read,sha,write_new,identity,verify_identity,open_bound_pidfd,send_once)

RUN='user_step4_train_only_ppo10000_train_v4'
CONTROL_RUN='user_ppo5000_control_v1'
REQUEST_SHA='715feef32953cc79491781753587aa0041dd154db12bfdaa8576d3f57de4aa82'
ENTRY_SHA='78d98f457e0a286263196ae523476895b71b56ae690a9c8ee889426805cf7bc8'
SUP_SHA='a15cf3e6eda8600f1cf95f439b79f4b263e93f7be3af287916b41e186238beaa'
RUNTIME_SHA='d6cec5d3c54bd30a8c2d9de98d855c759192b8e489056bf86c7068e62db8eecc'
LIB_SHA='375b16c3b8d919561e59cea8830ab071f2a128f39bf76363474cf0c37eb5ddde'
TARGET=5000

def boundary(rec):
 assert rec['run']==RUN and rec['role']=='train'
 assert rec['status']=='RUNNING_STAGE4_FRESH_ORIGINAL_TRAIN_PPO'
 assert rec['request_sha256']==REQUEST_SHA and rec['source_sha256']==ENTRY_SHA
 assert rec['training_data_mode']=='original_train_only' and rec['stage3_status']=='POSTPONED_BY_USER'
 assert rec['input_library_binding']['receipt_sha256']==LIB_SHA
 assert rec['original_train_families']==5361 and rec['actual_original_train_clips']==5551
 assert rec['original_VAL_motions_in_training']==rec['generated_training_clips']==0
 assert rec['diagnostic_optimizer_calls']==rec['old_teacher_training_updates']==rec['VAE_updates']==rec['LDM_updates']==0
 assert rec['output_writers_closed'] is False and rec['initial_optimizer_state_entries']==0
 n=rec['formal_completed_iterations'];steps=rec['formal_optimizer_updates']
 assert type(n) is int and 0<=n<=TARGET,'Target was missed; do not signal or silently overshoot'
 assert type(steps) is int and steps==n*20
 assert rec['optimizer_attempts_in_process']==rec['actual_optimizer_calls_in_process']==steps
 assert rec['checkpoints']==[] and 'emergency_last_good' not in rec
 return n==TARGET
