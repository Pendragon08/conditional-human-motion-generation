"""Real pretrained text encoding plus an untrained full-denoiser forward; no motion-quality claim."""
from pathlib import Path
import hashlib,json,time
import torch
import transformers
from text import FrozenDistilBert
from model import MotionDenoiser,ModelConfig

torch.set_num_threads(4);torch.manual_seed(73)
root=Path(__file__).resolve().parents[1];out=root/'results/reimplementation_text'
metadata=json.loads((out/'download.json').read_text());assert metadata['status']=='COMPLETED'
folder=root/'cache/text/distilbert-base-uncased'
for entry in metadata['files']:
    assert hashlib.sha256((folder/entry['name']).read_bytes()).hexdigest()==entry['sha256']
started=time.monotonic();encoder=FrozenDistilBert(folder).train()
assert not encoder.encoder.training and all(not p.requires_grad for p in encoder.parameters())
prompts=['A person stands still.','A person turns quickly and jumps forward.']
tokens,valid=encoder(prompts);again,again_valid=encoder(prompts)
torch.testing.assert_close(tokens,again,atol=0,rtol=0);assert torch.equal(valid,again_valid)
assert tokens.shape[0]==2 and tokens.shape[-1]==768 and torch.isfinite(tokens).all() and not tokens.requires_grad
single,single_valid=encoder([prompts[0]])
torch.testing.assert_close(tokens[0,:int(valid[0].sum())],single[0,single_valid[0]],atol=2e-5,rtol=2e-5)
assert not torch.allclose(tokens[0,0],tokens[1,0])
model=MotionDenoiser(ModelConfig()).eval()
with torch.no_grad():prediction,balance=model(torch.randn(2,4,262),torch.tensor([2,2]),tokens,valid)
assert prediction.shape==(2,4,262) and torch.isfinite(prediction).all() and torch.isfinite(balance)
result={'status':'PASSED_INTERFACE_CHECK','evaluation_type':'interface_check_only','model_repository':metadata['repository'],
    'model_revision':metadata['revision'],'paper_exact_checkpoint_confirmed':False,'prompts':prompts,
    'token_shape':list(tokens.shape),'valid_tokens':valid.sum(1).tolist(),'prediction_shape':list(prediction.shape),
    'encoder_frozen':True,'encoder_eval_under_train':True,'repeat_output_identical':True,'batch_padding_consistent':True,
    'full_denoiser_forward_finite':True,'motion_generator_trained':False,'motion_dataset_used':False,
    'motion_quality_evaluated':False,'physics_tested':False,'transformers_version':transformers.__version__,
    'torch_version':torch.__version__,'elapsed_seconds':time.monotonic()-started,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
