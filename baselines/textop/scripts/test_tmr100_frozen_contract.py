"""New100-prefix fullstate metadata negatives; no tensors/checkpoints/models."""
import copy,json,sys
from tmr100_trained_acceptance import validate_fullstate_metadata,TRAIN_SPEC,TRAIN_SPEC_SHA,TRAIN_RUNTIME_SHA
spec=dict(accepted_input_signature_sha256='1'*64)
state=dict(optimizer_updates=101500,completed_epochs=100,pending_validation=False,
    request_path=str(TRAIN_SPEC),request_sha256=TRAIN_SPEC_SHA,runtime_binding_sha256=TRAIN_RUNTIME_SHA,
    input_signature_sha256='1'*64,sampler=dict(count=32462,cursor=32462,epoch=99,steps=101500),
    all_rng=dict(python=None,numpy=None,torch=None,cuda=[None]),next_batch_sha256='6'*64,
    model=dict(synthetic=True),optimizer=dict(state={0:dict(step=101500)}))
assert validate_fullstate_metadata(state,spec)['optimizer_updates']==101500
changes=[('optimizer_updates',101499),('optimizer_updates',507500),('completed_epochs',500),
    ('pending_validation',True),('request_sha256','2'*64),('runtime_binding_sha256','2'*64),
    ('input_signature_sha256','2'*64),('sampler.epoch',100),('sampler.cursor',32461),
    ('sampler.steps',101501),('all_rng.cuda',[]),('next_batch_sha256','z'*64)]
for key,value in changes:
    bad=copy.deepcopy(state);target=bad;parts=key.split('.')
    for part in parts[:-1]:target=target[part]
    target[parts[-1]]=value
    try:validate_fullstate_metadata(bad,spec)
    except (AssertionError,KeyError):pass
    else:raise AssertionError('Wrong100-prefix fullstate metadata accepted: '+key)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps(dict(status='TMR100_FROZEN_PREFIX_METADATA_CONTRACT_VERIFIED',isolated_negative_cases=len(changes),
    actual_production_checkpoint_reads=0,actual_model_loads=0,actual_GPU_contexts=0,actual_exports=0,actual_optimizer_updates=0)))
