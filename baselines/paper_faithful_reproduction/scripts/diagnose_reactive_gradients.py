"""Read-only paired backend/activation-recomputation gradient diagnostic.

Uses actual training data and saved (non-EMA) model; no optimizer is constructed.
Only numerical summaries are public; data and intermediate tensors stay private.
"""
import argparse,contextlib,hashlib,json,math,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.utils.checkpoint import checkpoint

p=argparse.ArgumentParser()
for n in ['code-root','runtime-root','checkpoint','output']:p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--batch',type=int,default=8)
p.add_argument('--checkpoint-only',action='store_true')
p.add_argument('--shuffle',action='store_true')
p.add_argument('--operator-witness',action='store_true')
p.add_argument('--include-fp64-attention',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.code_root/'upstream'))
from reactivebfm.data.datasets import get_dataset_loader
from reactivebfm.utils.training.models import create_model_and_diffusion_smooth,load_saved_model
from reactivebfm.model.motion_planner.architectures.dit import DiTBlock

torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65 if a.batch>8 else .25,0)
torch.manual_seed(42);random.seed(42);np.random.seed(42)
args=SimpleNamespace(**json.loads((a.checkpoint.parent/'args.json').read_text()))
loader=get_dataset_loader(args.dataset,batch_size=a.batch,data_dir=str(a.runtime_root/'data'),
    hml_type=args.hml_type,unit_length=args.unit_length,fixed_len=args.context_len+args.n_primitives*args.pred_len,
    pred_len=args.pred_len,device='cpu',num_workers=0,shuffle=a.shuffle,deterministic=True,
    relative_root_xy=args.relative_root_xy,collate_mode='self_rollout')
motion,cond=next(iter(loader));motion=motion.cuda();lengths=cond['y']['lengths'].cuda()
model,diffusion=create_model_and_diffusion_smooth(args,loader)
load_saved_model(model,str(a.checkpoint),use_avg=False);model.cuda().train()
with torch.no_grad():text_embed=model.encode_text(cond['y']['text'])
original=DiTBlock.forward
def recompute(self,*x,**kw):return checkpoint(original,self,*x,use_reentrant=False,preserve_rng_state=True,**kw)
state={'checkpoint':str(a.checkpoint),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'batch':a.batch,'optimizer_updates':0,
 'torch':torch.__version__,'seed':42,'shuffle':a.shuffle,'results':[]}
reference=None
operator_rows=[];attention_calls=[0];active_backend=['default']
sdpa_original=torch.nn.functional.scaled_dot_product_attention
def trace_sdpa(q,k,v,**kw):
    if active_backend[0]=='math_fp64':
        double_kw=dict(kw)
        if double_kw.get('attn_mask') is not None and double_kw['attn_mask'].is_floating_point():
            double_kw['attn_mask']=double_kw['attn_mask'].double()
        with sdpa_kernel(SDPBackend.MATH):
            out=sdpa_original(q.double(),k.double(),v.double(),**double_kw).to(q.dtype)
    else:
        out=sdpa_original(q,k,v,**kw)
    call=attention_calls[0];attention_calls[0]+=1
    if a.operator_witness and call<2 and out.requires_grad:
        frozen=[x.detach().clone() for x in [q,k,v]]
        actual_backend=type(out.grad_fn).__name__
        def hook(dout):
            results={}
            operator_kw=dict(kw,dropout_p=0.0)
            previous_tf32=torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32=False
            with torch.enable_grad():
                for name,dtype in [('default_fp32',torch.float32),('math_fp32',torch.float32),('math_fp64',torch.float64)]:
                    inputs=[x.to(dtype).clone().requires_grad_() for x in frozen]
                    ctx=sdpa_kernel(SDPBackend.MATH) if name.startswith('math') else contextlib.nullcontext()
                    with ctx:
                        y=sdpa_original(*inputs,**operator_kw)
                        grads=torch.autograd.grad(y,inputs,dout.to(dtype))
                    results[name]=[g.detach().double() for g in grads]
            torch.backends.cuda.matmul.allow_tf32=previous_tf32
            ref=results['math_fp64'];row={'call':call,'shape':list(q.shape),'actual_backend':actual_backend,
                 'actual_dropout':kw.get('dropout_p',0.0),'witness_dropout':0.0,'previous_tf32':previous_tf32,
                 'input_stats':[{'kind':kind,'shape':list(x.shape),'abs_max':float(x.abs().max()),'norm':float(torch.linalg.vector_norm(x.double()))} for kind,x in zip(['q','k','v'],frozen)],'comparisons':{}}
            for name,grads in results.items():
                row['comparisons'][name]=[{'kind':kind,'relative_l2_error':float(torch.linalg.vector_norm(g-r)/(torch.linalg.vector_norm(r)+1e-30)),
                      'max_abs_error':float((g-r).abs().max()),'reference_norm':float(torch.linalg.vector_norm(r))} for kind,g,r in zip(['q','k','v'],grads,ref)]
            operator_rows.append(row)
            return dout
        out.register_hook(hook)
    return out
torch.nn.functional.scaled_dot_product_attention=trace_sdpa
variants=[('default',True),('math',True)] if a.checkpoint_only else [('default',False),('math',False),('math',True)]
if a.include_fp64_attention:variants.append(('math_fp64',True))
for backend,recomputation in variants:
    active_backend[0]=backend
    model.zero_grad(set_to_none=True);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    DiTBlock.forward=recompute if recomputation else original
    started=time.monotonic();losses=[]
    context=sdpa_kernel(SDPBackend.MATH) if backend=='math' else contextlib.nullcontext()
    with context:
        for k in range(args.n_primitives):
            start=k*args.pred_len
            valid=(lengths-start-args.context_len).clamp(0,args.pred_len)
            mask=torch.arange(args.pred_len,device='cuda')[None,None,None,:]<valid[:,None,None,None]
            y={'prefix':motion[...,start:start+args.context_len], 'mask':mask,'lengths':valid,'text_embed':text_embed}
            if args.training_rtc:
                y['rtc_delay']=torch.randint(0,args.rtc_max_delay+1,(a.batch,),device='cuda')
                y['rtc_prefix_noise_std']=args.rtc_prefix_noise_std
            x=motion[...,start+args.context_len:start+args.context_len+args.pred_len]
            terms=diffusion.training_losses(model,x,torch.zeros(a.batch,device='cuda',dtype=torch.long),model_kwargs={'y':y})
            loss=terms['loss'].mean()/args.n_primitives;losses.append(float(loss.detach()));loss.backward()
    gradients={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters() if v.grad is not None}
    row={'backend':backend,'activation_checkpointing':recomputation,'primitive_losses':losses,
       'loss':sum(losses),'gradient_norm_fp64':math.sqrt(sum(float(v.double().square().sum()) for v in gradients.values())),
       'all_gradients_finite':all(bool(torch.isfinite(v).all()) for v in gradients.values()),
       'seconds':time.monotonic()-started}
    if reference is None:reference=gradients
    else:
        err={n:float((v-reference[n]).abs().max()) for n,v in gradients.items()}
        row['vs_default_max_abs_gradient_difference']=max(err.values())
        row['vs_default_l2_gradient_difference']=math.sqrt(sum(float((v-reference[n]).double().square().sum()) for n,v in gradients.items()))
        row['largest_gradient_difference_tensor']=max(err,key=err.get)
    state['results'].append(row)
    (a.output/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(row),flush=True)
DiTBlock.forward=original
state['checkpoint_unchanged']=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==state['checkpoint_sha256']
state['operator_witness']=operator_rows
(a.output/'result.json').write_text(json.dumps(state,indent=2)+'\n')
