"""New disabled semantic-policy corruptions, no actual sources/latents/models."""
import copy
import json
import sys
from tmr69d_semantic_inputs import PROTOCOL,validate_semantic_policy,runtime_signature

p=dict(schema='textop.trained_robot_tmr69d_semantic.v1',metric_protocol=copy.deepcopy(PROTOCOL),
    witnesses_authorized=False,metrics_authorized=False,frozen_model_binding=None,witness_bindings=None,
    reserve_bytes=2*1024**3,log_allowance_bytes=128*1024**2,output_upper_bytes=8*1024**2,
    seeds=[20261028,20261029,20261030,20261031,20261101],modes=['conditional_cfg5','unconditional'])
validate_semantic_policy(p)
negative=[]
for name,key,value in [
 ('cosine_substitution','metric_protocol.distance','L2_normalized_cosine'),
 ('sampled_latent','metric_protocol.encoding_sample_mean',False),
 ('L2_normalized_FID','metric_protocol.per_sample_L2_normalization',True),
 ('drop_last28','metric_protocol.all188_queries_scored',False),
 ('biased_covariance','metric_protocol.covariance_ddof',0),
 ('pooled_duplicate_reference','metric_protocol.pooled_repeated_reference_across_seeds',True),
 ('optimistic_tie','metric_protocol.tied_rank','best_case'),
 ('best_seed_only','seeds',[20261028]),
 ('missing_mode','modes',['conditional_cfg5']),
 ('reduced_reserve','reserve_bytes',1024**3),
 ('premature_witness','witnesses_authorized',True),
 ('premature_metric','metrics_authorized',True),
 ('placeholder_binding_in_disabled_policy','frozen_model_binding',dict(synthetic=True)),
]:
 q=copy.deepcopy(p);target=q;parts=key.split('.')
 for part in parts[:-1]:target=target[part]
 target[parts[-1]]=value
 try:validate_semantic_policy(q)
 except (AssertionError,KeyError,TypeError):negative.append(name)
 else:raise AssertionError(name)
# Signature changes with frozen data, but not only the completed authorization transition.
w=copy.deepcopy(p);w.update(witnesses_authorized=True,frozen_model_binding=dict(synthetic_metadata_only=True))
sig=runtime_signature(w)
m=copy.deepcopy(w);m.update(witnesses_authorized=False,metrics_authorized=True,witness_bindings=dict(synthetic=True))
assert runtime_signature(m)==sig
m['frozen_model_binding']['synthetic_metadata_only']=False
assert runtime_signature(m)!=sig
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps(dict(status='TMR69D_SEMANTIC_DISABLED_POLICY_CONTRACT_VERIFIED',new_negative_cases=negative,
 authorization_only_signature_transition_verified=True,actual_arrays_read=0,actual_models_loaded=0,
 actual_gpu_contexts=0,actual_metric_jobs=0,actual_weights_or_policy_mutations=0)))
