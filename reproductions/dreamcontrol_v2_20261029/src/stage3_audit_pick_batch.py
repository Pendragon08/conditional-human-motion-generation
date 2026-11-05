"""Independent reread of actual generated candidates; no reference admission."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--expected-count',type=int,required=True)
a=p.parse_args()
sha=lambda x:hashlib.sha256(x).hexdigest()
began=time.monotonic()
report=json.loads((a.input/'report.json').read_text())
records=[json.loads(x) for x in (a.input/'records.jsonl').read_text().splitlines()]
assert report['status']=='sampling_completed_not_postoptimized'
assert report['records_sha256']==sha((a.input/'records.jsonl').read_bytes())
assert len(records)==a.expected_count==report['generated_candidates']
assert report['accepted_references']==0
assert len({r['candidate_id'] for r in records})==len(records)
assert len({r['feature_content_sha256'] for r in records})==len(records)
assert len({r['initial_noise_sha256'] for r in records})==len(records)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.01,device='cuda:0')
details=[]
for r in records:
    f=Path(r['path']).resolve(strict=True)
    assert f.is_relative_to(a.input.resolve()) and sha(f.read_bytes())==r['sha256']
    with np.load(f,allow_pickle=False) as z:
        features=z['features'];points=z['canonical_points'];targets=z['target_points'];mask=z['target_mask']
        m=json.loads(z['metadata_utf8'].tobytes())
    assert features.shape==(1,120,263) and points.shape==(1,120,22,3)
    assert targets.shape==(120,22,3) and mask.shape==(1,120,22)
    assert all(np.isfinite(x).all() for x in [features,points,targets])
    assert m['fps']==20 and m['frames']==120 and m['grasp_time_s']==2
    assert m['candidate_id']==r['candidate_id'] and not m['physical_acceptance']
    assert sha(features.tobytes())==m['feature_content_sha256']==r['feature_content_sha256']
    noise=torch.randn((1,263,1,120),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(m['initial_noise_seed']))
    assert sha(noise.cpu().numpy().tobytes())==m['initial_noise_sha256']==r['initial_noise_sha256']
    rotation=np.array(m['canonical_to_world_rotation']);translation=np.array(m['canonical_to_world_translation'])
    world=targets@rotation.T+translation
    obj=np.array(r['task_object_position']);assert np.allclose(obj,m['task_object_position'])
    assert np.allclose(world[40,21],obj+[0.,0.,.04],atol=2e-7)
    assert np.allclose(world[80,21],obj+[0.,0.,.20],atol=2e-7)
    assert np.all(np.abs(obj[:2]-[.35,-.2])<=.0600001) and obj[2]==.9
    expected=np.array([.35,-.2,.9])+np.r_[np.random.default_rng(m['initial_noise_seed']).uniform(-.06,.06,2),0.]
    assert np.allclose(obj,expected,atol=1e-12)
    error=np.linalg.norm(points[0,:,21]-targets[:,21],axis=-1)
    assert abs(float(error[40])-r['wrist_error_at_grasp_m'])<1e-6
    details.append({'candidate_id':r['candidate_id'],'file_sha256':r['sha256'],
        'noise_recomputed':True,'target_object_binding':True,'20Hz_120states':True,
        'wrist_error_before_postopt_m':float(error[40])})
result={'passed':True,'scope':'actual DDPM candidate integrity/time/target/seed binding; not postoptimization or reference admission',
    'candidates':len(records),'admitted_references':0,'records_sha256':report['records_sha256'],
    'details':details,'elapsed_seconds':time.monotonic()-began,'PPO_started':False}
a.output.parent.mkdir(exist_ok=True,parents=True)
with a.output.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
