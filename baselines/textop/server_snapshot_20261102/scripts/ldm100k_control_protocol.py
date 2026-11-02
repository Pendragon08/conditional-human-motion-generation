"""Explicit LDM100k endpoint; shares the tested stdlib pidfd/file protocol."""
from vae100k_control_protocol import read,sha,write_new,verify_checkpoint,identity,verify_identity,open_bound_pidfd,send_once,RESTORE_KEYS
from vae100k_control_protocol import checkpoint_ready as base_checkpoint_ready

def checkpoint_ready(receipt,policy):
 assert receipt['vae_pretraining_attempts']==100000
 assert receipt['vae_checkpoint_sha256']==policy['vae_checkpoint_sha256']
 assert receipt['vae_inference_sha256']==policy['vae_inference_sha256']
 ck=base_checkpoint_ready(receipt,policy)
 if ck is not None:
  assert isinstance(receipt['full_sampling_history_calls'],int) and receipt['full_sampling_history_calls']>=0
  assert isinstance(receipt['selected_rollout_history_primitives'],int) and receipt['selected_rollout_history_primitives']>=0
 # History/semantic quality is audited AFTER stop; it cannot authorize training past the user's endpoint.
 return ck
