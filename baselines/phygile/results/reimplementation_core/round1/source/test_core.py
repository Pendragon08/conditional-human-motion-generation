"""Deterministic numerical/gradient checks; not trained-model performance tests."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from torch import nn
from model import ModelConfig,MotionDenoiser,TPMoE
from features import BLOCKS,BlockNormalizer,pack_blocks,asfo_multipliers
from diffusion import X0Diffusion

torch.set_num_threads(4);torch.manual_seed(73);np.random.seed(73)
checks=[];started=time.monotonic()
def passed(name):checks.append(name);print('PASS',name,flush=True)

# Eq. 8: independently form one mixed expert's matrices and compare nonlinear output.
c=ModelConfig(width=16,hidden=24,heads=4,layers=2,experts=3,dropout=0,token_chunk=1)
moe=TPMoE(c);x=torch.randn(2,4,16);tokens=torch.randn(2,1,768)
attn=torch.rand(2,4,1);tv=torch.ones(2,1,dtype=torch.bool);fv=torch.ones(2,4,dtype=torch.bool)
actual,bal=moe(x,tokens,attn,tv,fv)
w=moe.gate(tokens).softmax(-1)[:,0,:]
w1=sum(w[:,k,None,None]*moe.w1[k] for k in range(3));b1=sum(w[:,k,None]*moe.b1[k] for k in range(3))
w2=sum(w[:,k,None,None]*moe.w2[k] for k in range(3));b2=sum(w[:,k,None]*moe.b2[k] for k in range(3))
expected=torch.nn.functional.gelu(torch.bmm(x,w1)+b1[:,None,:])
expected=torch.bmm(expected,w2)+b2[:,None,:]
mask=torch.sigmoid(24*(attn-.25*attn.amax(1,keepdim=True)))
torch.testing.assert_close(actual,expected*mask,atol=2e-6,rtol=2e-6)
torch.testing.assert_close(bal,3*((w.mean(0)-1/3)**2).sum())
passed('TP-MoE mixes parameters before nonlinear activation; balance matches formula')

model=MotionDenoiser(c).eval();motion=torch.randn(2,7,262);text=torch.randn(2,5,768);steps=torch.tensor([2,5])
tv=torch.tensor([[1,1,1,0,0],[1,1,1,1,0]],dtype=torch.bool)
fv=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,0]],dtype=torch.bool)
y,balance=model(motion,steps,text,tv,fv)
motion2=motion.clone();motion2[~fv]=9999;text2=text.clone();text2[~tv]=-9999
y2,balance2=model(motion2,steps,text2,tv,fv)
torch.testing.assert_close(y,y2);torch.testing.assert_close(balance,balance2)
assert y.shape==(2,7,262) and not bool(y[~fv].any())
passed('Padding payload cannot affect valid frames or routing balance')

model.train();ddpm=X0Diffusion(model,torch.linspace(.0001,.02,8))
result=ddpm.training_loss(motion,text,tv,fv,steps=steps,noise=torch.randn_like(motion),condition_dropout=0)
result['loss'].backward()
assert torch.isfinite(result['loss'])
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert all(block.moe.w1.grad.abs().sum()>0 and block.moe.gate[-1].weight.grad.abs().sum()>0 for block in model.blocks)
passed('x0 loss has finite nonzero gradients into every expert pool and gate')

class TokenProbe(nn.Module):
    def forward(self,x,t,tokens,text_valid=None,frame_valid=None):
        return torch.ones_like(x)*tokens[:,0,0][:,None,None],x.new_zeros(())
probe=X0Diffusion(TokenProbe().eval(),torch.linspace(.001,.02,4))
positive=torch.zeros(2,1,768);positive[:,0,0]=2
negative=torch.zeros_like(positive);negative[:,0,0]=-1
torch.testing.assert_close(probe.guided_x0(motion,torch.tensor([0,1]),positive,scale=2.5),torch.full_like(motion,5))
torch.testing.assert_close(probe.guided_x0(motion,torch.tensor([0,1]),positive,scale=2.5,negative_tokens=negative),torch.full_like(motion,6.5))
passed('Standard and negative-prompt CFG match Eqs. 13–14')

model.eval();known=torch.randn_like(motion);mask=torch.zeros_like(motion,dtype=torch.bool);mask[:,:2]=True;mask[:,-1]=True
sample=ddpm.sample(motion.shape,text,known_values=known,known_mask=mask,generator=torch.Generator().manual_seed(91))
assert torch.isfinite(sample).all();torch.testing.assert_close(sample[mask],known[mask],atol=0,rtol=0)
passed('Chosen inpainting sampler preserves exact clean prefix and suffix')

blocks={name:np.random.randn(9,width).astype(np.float32) for name,width in BLOCKS}
blocks['body_rotations_6d']=np.random.uniform(-1,1,(9,174)).astype(np.float32)
blocks['foot_contacts']=np.random.randint(0,2,(9,4));blocks['hand_contacts']=np.random.randint(0,2,(9,2))
features=pack_blocks(blocks);norm=BlockNormalizer.fit([features]);normalized=norm.transform(features)
np.testing.assert_array_equal(normalized[:,43:217],features[:,43:217]);np.testing.assert_array_equal(normalized[:,256:],features[:,256:])
np.testing.assert_allclose(norm.inverse(normalized),features,rtol=1e-5,atol=1e-6)
passed('262D block normalization preserves rotation/contact channels and round trips')

m,meta=asfo_multipliers([['walk'],['walk'],['walk'],['walk'],['jump'],['run'],['run']],4)
assert meta['median']==2 and meta['tag_multipliers']=={'walk':1,'jump':2,'run':1}
assert m.tolist()==[1,1,1,1,2,1,1]
passed('ASFO frequency, median and sample maxima match Eqs. 10–11')

full=MotionDenoiser(ModelConfig())
assert len(full.blocks)==8 and full.config.width==512 and full.config.experts==12
parameter_count=sum(p.numel() for p in full.parameters())
passed('Full stated architecture constructs without pretrained weights')
root=Path(__file__).resolve().parents[1];out=root/'results/reimplementation_core';out.mkdir(parents=True,exist_ok=True)
report={'status':'PASSED_NUMERICAL_CHECKS','tests':checks,'torch_version':torch.__version__,'seed':73,
        'elapsed_seconds':time.monotonic()-started,'full_architecture_parameters':parameter_count,
        'trained_model':False,'text_encoder_loaded':False,'dataset_used':False,'physics_simulation_tested':False,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
