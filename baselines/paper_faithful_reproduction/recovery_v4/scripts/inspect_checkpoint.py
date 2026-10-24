"""CPU deserialization/finite-state diagnostic for a real ready checkpoint pair.

This does not resume optimization, run physics, or measure learned motion quality.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import torch
from prepare_verified_resume import select_verified


def inspect(manifest):
    manifest=Path(manifest).resolve()
    selected,_=select_verified(manifest.parent)
    if Path(selected['manifest']).resolve()!=manifest:
        raise ValueError('Explicit manifest is not the latest verified pair')
    model=next(Path(x['file']) for x in selected['files'] if Path(x['file']).name.startswith('model'))
    optimizer=next(Path(x['file']) for x in selected['files'] if Path(x['file']).name.startswith('opt'))
    payload=torch.load(model,map_location='cpu',weights_only=True,mmap=True)
    if set(payload)!= {'model','model_avg'}:raise ValueError('Expected full-run model and EMA states')
    if payload['model'].keys()!=payload['model_avg'].keys():raise ValueError('Model/EMA state keys differ')
    summary={}
    for name,state in payload.items():
        tensors=[v for v in state.values() if isinstance(v,torch.Tensor)]
        if not tensors or len(tensors)!=len(state):raise ValueError('Unexpected model-state structure')
        for key,value in state.items():
            if value.shape!=payload['model'][key].shape:raise ValueError('EMA shape mismatch')
            if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite model state: '+key)
        summary[name]={'tensor_count':len(tensors),'state_tensor_elements':sum(v.numel() for v in tensors),'all_finite':True}
    opt=torch.load(optimizer,map_location='cpu',weights_only=True,mmap=True)
    states=list(opt['state'].values())
    if not states:raise ValueError('Missing optimizer states')
    steps=sorted({float(s['step']) for s in states})
    if steps!=[float(selected['optimizer_updates'])]:raise ValueError('Optimizer counters disagree with recorded actual updates')
    for state in states:
        for value in state.values():
            if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite optimizer state')
    return {'status':'CHECKPOINT_CPU_INSPECTION_PASSED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'official_step':selected['official_step'],'optimizer_updates':selected['optimizer_updates'],
        'manifest_sha256':selected['manifest_sha256'],'files':selected['files'],'model':summary,
        'optimizer_state_count':len(states),'optimizer_step_values':steps,'all_optimizer_tensors_finite':True,
        'checkpoint_restored_to_gpu':False,'optimizer_resumed':False,'physics_evaluated':False,
        'scope':'SHA256/size, CPU tensor deserialization, finite model/EMA/optimizer and optimizer-counter diagnostic only.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);result=inspect(a.manifest)
    with a.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
