"""Two-update CPU plumbing witness using explicitly synthetic tensors, not motion data."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from train import train,load_manifest
from model import ModelConfig,MotionDenoiser

torch.set_num_threads(4)
root=Path(__file__).resolve().parents[1]
out=root/'results/reimplementation_training_witness';out.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(193)
entries=[];arrays=[]
for i in range(3):
    values=rng.normal(size=(4+i,262)).astype(np.float32)
    values[:,43:217]=rng.uniform(-1,1,values[:,43:217].shape)
    values[:,256:]=rng.integers(0,2,values[:,256:].shape)
    file=out/f'synthetic_unit_{i}.npy';np.save(file,values);arrays.append(values)
    entries.append({'features':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                    'source_group':f'synthetic_group_{i}','texts':[f'Numerical unit input {i}.'],
                    'annotation_source':'Artificial text solely for interface testing, not a motion caption','tags':['unit']})
manifest={'format':'phygile262-explicit-v1','purpose':'unit_test','mapping_status':'synthetic_unit_test',
    'feature_semantics':'Artificial bounded arrays; rotations are not valid robot rotations. Not motion data.',
    'source_and_license':'Seeded numerical unit test, no external dataset',
    'splits':{'train':entries[:2],'test':entries[2:]}}
manifest_path=out/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
try:load_manifest(manifest_path,'research_training')
except ValueError:pass
else:raise AssertionError('Synthetic witness accepted as research data')
bad=json.loads(json.dumps(manifest));bad['splits']['test'][0]['source_group']='synthetic_group_0'
(out/'overlap_rejection_case.json').write_text(json.dumps(bad))
try:load_manifest(out/'overlap_rejection_case.json','unit_test')
except ValueError:pass
else:raise AssertionError('Cross-split source overlap accepted')
config={'purpose':'unit_test','dataset_manifest':'manifest.json','text_encoder_path':'../../cache/text/distilbert-base-uncased',
    'seed':29,'batch_size':2,'max_updates':2,'save_every':2,'max_sequence_frames':8,
    'model':{'width':16,'hidden':24,'layers':2,'heads':4,'experts':3,'dropout':0.},
    'schedule':{'type':'linear','start':.0001,'end':.02,'steps':8},
    'learning_rate':1e-4,'adam_betas':[.9,.999],'weight_decay':0.,'gradient_clip':1.,
    'condition_dropout':.1,'balance_weight':.01,
    'asfo':{'enabled':False,'rho_max':4,'count_policy':'keep_at_least_one','mirroring':False}}
config_path=out/'config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
record=train(config_path,out/'run',torch.device('cpu'))
assert record['purpose']=='unit_test' and record['status']=='COMPLETED' and record['completed_updates']==2
checkpoint=torch.load(out/'run/model_000002.pt',map_location='cpu',weights_only=False)
assert checkpoint['purpose']=='unit_test' and checkpoint['optimizer_updates']==2
assert set(int(x['step']) for x in checkpoint['optimizer_state_dict']['state'].values())=={2}
model=MotionDenoiser(ModelConfig(**config['model'])).eval();model.load_state_dict(checkpoint['model_state_dict'])
with torch.no_grad():prediction,_=model(torch.zeros(1,4,262),torch.tensor([1]),torch.zeros(1,2,768))
assert torch.isfinite(prediction).all()
expected=np.concatenate(arrays[:2])[:,0].mean()
assert abs(float(checkpoint['normalization_mean'][0])-expected)<1e-6
result={'status':'PASSED_TRAINING_PLUMBING_CHECK','purpose':'unit_test','optimizer_updates':2,
        'device':'cpu','real_motion_data_used':False,'motion_quality_evaluated':False,
        'full_size_generator_trained':False,'physics_loop_run':False,
        'checks':['synthetic inputs rejected as research data','cross-split source overlap rejected',
                  'training-only normalization','two finite optimizer updates','checkpoint reload'],
        'checkpoint_sha256':hashlib.sha256((out/'run/model_000002.pt').read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
