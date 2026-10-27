"""Paper-grounded, independently written PhyGile diffusion backbone; untrained."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    features:int=262
    text_width:int=768
    width:int=512
    hidden:int=1024
    heads:int=4
    layers:int=8
    experts:int=12
    dropout:float=0.1
    mask_gamma:float=24.0
    mask_beta:float=0.25
    token_chunk:int=4

def sinusoidal(indices,width):
    half=width//2
    freq=torch.exp(-math.log(10000.0)*torch.arange(half,device=indices.device,dtype=torch.float32)/max(half-1,1))
    phase=indices.float()[...,None]*freq
    value=torch.cat([phase.sin(),phase.cos()],-1)
    return F.pad(value,(0,width-value.shape[-1]))

class TPMoE(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.gate=nn.Sequential(nn.Linear(c.text_width,c.width),nn.SiLU(),nn.Linear(c.width,c.width),nn.SiLU(),nn.Linear(c.width,c.experts))
        self.w1=nn.Parameter(torch.empty(c.experts,c.width,c.hidden));self.b1=nn.Parameter(torch.zeros(c.experts,c.hidden))
        self.w2=nn.Parameter(torch.empty(c.experts,c.hidden,c.width));self.b2=nn.Parameter(torch.zeros(c.experts,c.width))
        for k in range(c.experts):
            nn.init.xavier_uniform_(self.w1[k]);nn.init.xavier_uniform_(self.w2[k])

    def forward(self,x,tokens,attention,text_valid,frame_valid):
        # x [B,T,D], attention [B,T,N]. Mix weights BEFORE applying GELU (Eq. 8).
        weights=self.gate(tokens).softmax(-1)
        count=text_valid.sum().clamp_min(1)
        mean=(weights*text_valid[...,None]).sum((0,1))/count
        balance=self.c.experts*((mean-1/self.c.experts)**2).sum()
        maxima=attention.masked_fill(~frame_valid[...,None],float('-inf')).amax(1,keepdim=True)
        mask=torch.sigmoid(self.c.mask_gamma*(attention-self.c.mask_beta*maxima))
        mask=mask*text_valid[:,None,:]*frame_valid[:,:,None]
        delta=torch.zeros_like(x)
        for start in range(0,tokens.shape[1],self.c.token_chunk):
            stop=start+self.c.token_chunk;w=weights[:,start:stop]
            w1=torch.einsum('bnk,kdh->bndh',w,self.w1);b1=torch.einsum('bnk,kh->bnh',w,self.b1)
            w2=torch.einsum('bnk,khd->bnhd',w,self.w2);b2=torch.einsum('bnk,kd->bnd',w,self.b2)
            hidden=F.gelu(torch.einsum('btd,bndh->bnth',x,w1)+b1[:,:,None,:])
            transformed=torch.einsum('bnth,bnhd->bntd',hidden,w2)+b2[:,:,None,:]
            delta=delta+(transformed*mask[:,:,start:stop].transpose(1,2)[...,None]).sum(1)
        return delta,balance

class DecoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.norms=nn.ModuleList([nn.LayerNorm(c.width,elementwise_affine=False) for _ in range(3)])
        self.ada=nn.Sequential(nn.SiLU(),nn.Linear(c.width,6*c.width))
        self.self_attention=nn.MultiheadAttention(c.width,c.heads,c.dropout,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(c.width,c.heads,c.dropout,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(c.width,c.hidden),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.hidden,c.width))
        self.moe=TPMoE(c);self.dropout=nn.Dropout(c.dropout)

    def forward(self,x,time,memory,tokens,text_valid,frame_valid):
        mod=self.ada(time).chunk(6,-1)
        def norm(i,value):return self.norms[i](value)*(1+mod[2*i][:,None,:])+mod[2*i+1][:,None,:]
        h=norm(0,x);x=x+self.dropout(self.self_attention(h,h,h,key_padding_mask=~frame_valid,need_weights=False)[0])
        memory_valid=torch.cat([torch.ones_like(text_valid[:,:1]),text_valid],1)
        h,attention=self.cross_attention(norm(1,x),memory,memory,key_padding_mask=~memory_valid,need_weights=True,average_attn_weights=True)
        x=x+self.dropout(h);x=x+self.dropout(self.ffn(norm(2,x)))
        update,balance=self.moe(x,tokens,attention[:,:,1:],text_valid,frame_valid)
        return (x+self.dropout(update))*frame_valid[...,None],balance

class MotionDenoiser(nn.Module):
    def __init__(self,c=None):
        super().__init__();self.config=c or ModelConfig();c=self.config
        self.motion_in=nn.Linear(c.features,c.width);self.text_in=nn.Linear(c.text_width,c.width)
        self.summary_query=nn.Parameter(torch.randn(1,1,c.width)/math.sqrt(c.width))
        self.summary_pool=nn.MultiheadAttention(c.width,c.heads,c.dropout,batch_first=True)
        self.time_mlp=nn.Sequential(nn.Linear(c.width,c.width),nn.SiLU(),nn.Linear(c.width,c.width))
        self.blocks=nn.ModuleList([DecoderBlock(c) for _ in range(c.layers)])
        self.final_norm=nn.LayerNorm(c.width);self.motion_out=nn.Linear(c.width,c.features)

    def forward(self,noisy,timesteps,tokens,text_valid=None,frame_valid=None):
        b,t,f=noisy.shape;n=tokens.shape[1]
        assert f==self.config.features and tokens.shape==(b,n,self.config.text_width)
        if text_valid is None:text_valid=torch.ones(b,n,dtype=torch.bool,device=noisy.device)
        if frame_valid is None:frame_valid=torch.ones(b,t,dtype=torch.bool,device=noisy.device)
        if not bool(text_valid.any(1).all()) or not bool(frame_valid.any(1).all()):raise ValueError('Every sequence needs a valid token and frame')
        # Erase padded payloads before projection as well as masking attention.
        tokens=tokens.masked_fill(~text_valid[...,None],0)
        memory=self.text_in(tokens)
        query=self.summary_query.expand(b,-1,-1)
        summary=self.summary_pool(query,memory,memory,key_padding_mask=~text_valid,need_weights=False)[0]
        memory=torch.cat([summary,memory],1)
        x=self.motion_in(noisy.masked_fill(~frame_valid[...,None],0))
        x=x+sinusoidal(torch.arange(t,device=x.device),self.config.width)[None,:,:].to(x.dtype)
        time=self.time_mlp(sinusoidal(timesteps,self.config.width).to(x.dtype))
        balances=[]
        for block in self.blocks:
            x,balance=block(x,time,memory,tokens,text_valid,frame_valid);balances.append(balance)
        return self.motion_out(self.final_norm(x))*frame_valid[...,None],torch.stack(balances).mean()
