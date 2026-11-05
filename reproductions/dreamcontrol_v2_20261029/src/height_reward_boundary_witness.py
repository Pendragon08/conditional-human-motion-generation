"""Compare isolated continuous ramp with old-Pick CPU witness, never train."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from height_reward_boundary_prototype import height_reward
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1)
old=Path(__file__).with_name('report.json');r=json.loads(old.read_text());z=torch.tensor(r['reward_heights']);outputs={}
for closed in (False,True):
    flag=torch.full_like(z,closed,dtype=torch.bool);v,success=height_reward(z,flag,height_threshold=.95,fall_threshold=.9)
    expected=torch.tensor(r['outputs'][str(closed).lower()]['reward']);mask=torch.arange(len(z))!=2
    assert torch.allclose(v[mask],expected[mask],atol=1e-6,rtol=0) and (not closed or v[2].item()==1.)
    assert success.tolist()==[bool(x) for x in r['outputs'][str(closed).lower()]['success_counter_increment']]
    outputs[str(closed).lower()]={'reward':v.tolist(),'success':success.tolist()}
for val in (float('nan'),float('inf')):
    try:height_reward(torch.tensor([val]),torch.tensor([True]),height_threshold=.95,fall_threshold=.9)
    except ValueError:pass
    else:raise AssertionError('non-finite height accepted')
try:height_reward(torch.tensor([1.]),torch.tensor([True]),height_threshold=.9,fall_threshold=.9)
except ValueError:pass
else:raise AssertionError('degenerate threshold accepted')
out={'old_witness_sha256':sha(old),'prototype_sha256':sha(Path(__file__).with_name('height_reward_boundary_prototype.py')),'script_sha256':sha(Path(__file__)),'outputs':outputs,'limitations':'CPU mock, hand-specified threshold rule and no geometry or PhysX; not wired to task or evidence of author intent. Caller must choose height coordinate frame.'}
with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
