# Full neural batch512 zero-update diagnostic

Warm unchanged paper69 environment934d8221; ledger SERVER_NOTES.md.
configs/neural-batch512-spec.json was written before dispatch. Pinned official
VAE9L/LDM8L, FP32 batch512, history2/future8/fourprimitives, real cached CLIP
and candidate-only train statistics. GPU1/CPU4/Torch allocator65%. No packages,
shared assets or official sources edited. Fixed candidate2296 view receiptSHA
c4875fa612707507ae8de28c15ca4f1160cff5293467cb91c7801d43184d3b23.
This is unfiltered data and random VAE/denoiser weights, not accepted training.

Author neural512_author_v1 actually exited0 in34.28s. EightVAE and fourLDM
backwards using the unmodified official MVAEManager/DARManager loss bodies
were finite with nonzero gradients. Model parameter hashes were unchanged.
Two5stepCFG5 sampling segments produced finite69D values only. No optimizer
was constructed; no Adam/EMA state was actually allocated. Measured Torch
allocated peakVAE3429537280bytes/LDM1681969664bytes excludes those states,
checkpointing and final accepted corpus. It is NOT full training memory,
semantic quality, deployment latency or permission to train on candidates.

## Fresh document invocation once, unchanged

Run from PowerShell exactly once. The output name must not exist. Do not edit,
retry, inspect source to repair, install packages, rebuild the view, stop other
processes or launch actual training. Existing GMR/FK queues and57D SIGSTOP
remain intact.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_neural_batch512.py --run neural512_document_v1'
```

Require directexit0/TEXTOP_NEURAL_BATCH512_SUPERVISED_EXIT0. Inspect only public
receipts/logs, not private candidate PKLs/caches/index/texts/arrays:

- generator_validation/neural512_document_v1/receipt.json
- receipts/neural_batch512_neural512_document_v1_exit.json
- logs/neural_batch512_neural512_document_v1.log

All server paths above are relative to
/data/code/ouyanghao/oyh/textop-reproduction-20261024.
Require child_exit_code0,timed_outfalse, statusTEXTOP_NEURAL_BATCH512_WITNESS_PASSED,
batch512,primitives4,feature_dim69,vae_backward_passes8,ldm_backward_passes4;
all losses/gradientnorms finite and positive with nonzero finitegradienttensors;
vae_parameters_unchangedtrue,denoiser_parameters_unchangedtrue,
frozen_vae_has_no_gradientstrue,optimizer_constructedfalse,
optimizer_updates0,optimizer_and_ema_states_actually_allocatedfalse,
dataset_acceptedfalse,physical_filter_executedfalse,
full_generator_training_startedfalse,model_weights_trainedfalse,
output_writers_closedtrue. Sampling512x16frames,5steps,CFG5,two8frame segments.
Observe CPUaffinity4 and source/spec/view hashes; exact numeric losses or
memory need not be identical. No IsaacSDK is used and no SDK closefallback
should be asserted for this run.

SourceSHA26cbf30cf75afbdfdee97ed3bdf26de11f91d82b67bf35690b719844b237410d
SpecSHA2a91a7f6ec63732b90a666b52f615264ee1117b467e41f7a49485b53652896ca
BaseSpecSHA30bcaa928d4e7ed4cf7359d88dfe909334718957fffbd07bc98510467612d421
CodecSHAcb97b9a2dd603979ec512dddca07be88ca0c9144a82b1eb935fc28dab6a0486f
SamplerSHAcf5c1e3f27665e7fc4d4aa1ede6bcd1931e3563409ad766ff67a576cc0c7b60d

Save command/stdout, public numerical receipt and supervisor exit proof plus a
concise independent REPORT.md under
work/textop_reproduction_20261024/neural_batch512_doc_witness_v1.
No publication, private copies or messaging other chats. Report divergence and
retain failures. No optimizer/accepted-corpus/fulltrain readiness claim.
