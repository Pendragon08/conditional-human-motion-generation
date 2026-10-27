"""x0 DDPM primitives with an explicitly supplied (not paper-inferred) schedule."""
import torch
from torch import nn

class X0Diffusion(nn.Module):
    def __init__(self,model,betas):
        super().__init__();self.model=model
        betas=torch.as_tensor(betas,dtype=torch.float64)
        if betas.ndim!=1 or len(betas)<2 or not bool(((betas>0)&(betas<1)).all()):
            raise ValueError('Supply an explicit valid beta schedule; its author value is not established')
        alpha=1-betas;abar=alpha.cumprod(0);previous=torch.cat([torch.ones(1,dtype=abar.dtype),abar[:-1]])
        for name,value in {'betas':betas,'alpha_bar':abar,'posterior_variance':betas*(1-previous)/(1-abar),
                           'posterior_x0':betas*previous.sqrt()/(1-abar),
                           'posterior_xt':(1-previous)*alpha.sqrt()/(1-abar)}.items():
            self.register_buffer(name,value.float())

    @staticmethod
    def extract(values,steps,x):return values[steps].reshape(-1,*([1]*(x.ndim-1))).to(x.dtype)

    def q_sample(self,clean,steps,noise):
        abar=self.extract(self.alpha_bar,steps,clean)
        return abar.sqrt()*clean+(1-abar).sqrt()*noise

    def training_loss(self,clean,tokens,text_valid=None,frame_valid=None,steps=None,noise=None,
                      condition_dropout=0.1,balance_weight=0.01):
        batch=clean.shape[0]
        if steps is None:steps=torch.randint(len(self.betas),(batch,),device=clean.device)
        if noise is None:noise=torch.randn_like(clean)
        if not 0<=condition_dropout<=1:raise ValueError('Invalid conditioning dropout')
        drop=torch.rand(batch,device=tokens.device)<condition_dropout
        conditioned=tokens.masked_fill(drop[:,None,None],0)
        prediction,balance=self.model(self.q_sample(clean,steps,noise),steps,conditioned,text_valid,frame_valid)
        squared=(prediction-clean).square()
        if frame_valid is None:mse=squared.mean()
        else:mse=(squared*frame_valid[...,None]).sum()/(frame_valid.sum()*clean.shape[-1]).clamp_min(1)
        return {'loss':mse+balance_weight*balance,'x0_mse':mse,'load_balance':balance}

    def guided_x0(self,noisy,steps,tokens,text_valid=None,frame_valid=None,scale=2.5,
                  negative_tokens=None,negative_valid=None):
        positive,_=self.model(noisy,steps,tokens,text_valid,frame_valid)
        if scale==1:return positive
        baseline=torch.zeros_like(tokens) if negative_tokens is None else negative_tokens
        baseline_valid=text_valid if negative_tokens is None else negative_valid
        negative,_=self.model(noisy,steps,baseline,baseline_valid,frame_valid)
        return negative+scale*(positive-negative)

    @torch.no_grad()
    def sample(self,shape,tokens,text_valid=None,frame_valid=None,scale=2.5,
               negative_tokens=None,negative_valid=None,known_values=None,known_mask=None,generator=None):
        """Chosen reconstruction: re-noise/reinject known prefix/suffix at each step.

        This inpainting algorithm is an explicit implementation assumption; the
        paper specifies conditioning, but not a complete sampler implementation.
        """
        if self.model.training:raise ValueError('Call model.eval() before deterministic inference')
        if (known_values is None)!=(known_mask is None):raise ValueError('Boundary values and mask must be paired')
        if known_values is not None:
            if tuple(known_values.shape)!=tuple(shape) or tuple(known_mask.shape)!=tuple(shape) or known_mask.dtype!=torch.bool:
                raise ValueError('Known arrays must match the full sample shape; mask must be boolean')
        rand=lambda:torch.randn(shape,device=tokens.device,dtype=tokens.dtype,generator=generator)
        x=rand();known_noise=rand() if known_values is not None else None
        for index in reversed(range(len(self.betas))):
            steps=torch.full((shape[0],),index,device=tokens.device,dtype=torch.long)
            if known_values is not None:x=torch.where(known_mask,self.q_sample(known_values,steps,known_noise),x)
            prediction=self.guided_x0(x,steps,tokens,text_valid,frame_valid,scale,negative_tokens,negative_valid)
            if known_values is not None:prediction=torch.where(known_mask,known_values,prediction)
            mean=self.extract(self.posterior_x0,steps,x)*prediction+self.extract(self.posterior_xt,steps,x)*x
            x=mean if index==0 else mean+self.extract(self.posterior_variance,steps,x).sqrt()*rand()
        if known_values is not None:x=torch.where(known_mask,known_values,x)
        if frame_valid is not None:x=x*frame_valid[...,None]
        return x
