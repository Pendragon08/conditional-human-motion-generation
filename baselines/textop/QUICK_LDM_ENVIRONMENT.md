## CURRENT2026-10-26 19:32 — explicit sampler device, training disabled

Authorv1 completed4 diagnostic AdamW updates and memory restore, then failed
in the five-step sampler: it inferred device from model.parameters(), but CFG
was a function wrapper. Actual childexit1 and all source/receipt preserved.
Only sampler device='cuda' made explicit; no model/loss/sampling math change.
Existing frozen-VAE cache is reused after SHAcheck, not duplicated. New author
ldm_kernel_author_v2 dispatched ONCE; verify actual exit before independent
ldm_kernel_document_v1, which has never run. Do not rerun failed authorv1.
The first local doc metadata update also failed on defaultGBK decoding AFTER
source deploy; corrected explicitly to UTF8, no numerical/config change.

# Separate quick LDM3000: declared before launch

Use the existing paper69@934d8221 warm Python/Torch1.13.1 environment in
SERVER_NOTES.md and configs/paper69-env-spec.json. No package installs, SDK or
shared code changes. This is a different profile from VAE: run_quick_ldm.py,
quick_ldm_runtime.py and supervise_quick_ldm.py, exact byte-bound in
configs/quick-ldm-runtime-spec.json. Old19 paused-training/control sources and
current VAE runtime are not modified. Shared GPU1/CPU0–3/Torch0.65 authorization
overrides the generic empty-GPU rule; actual GPU1 free>20GiB is required.

The profile cannot be staged before quick_vae_train_v1 genuinely closes with
exit0, both fixedVAL reconstruction gates passed, actual2000 fullcheckpoint
and exact model/Adam/RNG file-resume proof. Full VAE2000 is private server+D
SHA backed up before staging. No4-update diagnostic/old57D/random VAE allowed.
Corpus remains the accepted42 ORIGINALreference sources,31train/11val,
174annotationpairs/74CLIPtexts with TRAIN-only69D statistics and source/pose
separation. No executed-pose training targets or unassessed mirrors.

The fixed official denoiser is hidden512/FF1024/8layers/4heads, CLIP512,
history(2,69),latent(1,128),condmask0.1. AdamW1e-4/clipnorm1/EMAfalse,
Huber reconstruction/latent plus original geometry weights; frozen trained
VAE2000,5cosine diffusion predicting x0, uniform timestep sampler, original
one-noised-latent training/evaluation (use_full_sample=false), rolloutcap.8 and
original static-history behavior. Integer timestep compatibility is the
unchanged paper69 adapter. Quick stages1000/1000/1000, compressed LR/curriculum,
save1000/eval500×12 are explicit differences from paper300000/2000×10.
Validation is always0 optimizer updates, gradient-skips counted separately.

The HOME owned persistent destination is
/home/ouyanghao/textop-quick69-checkpoints-20261026/ldm.
The official DARManager creates a VAE cache `vae.pth` next to DAR checkpoints.
The two filesystems differ, so the budget includes ONE <=600MiB full VAE copy,
shared across author/document/train, not duplicated by run. Three regular
LDM fullstates plus one emergency are each bounded300MiB, with2GiB reserve and
512MiB logs. Source, ownership/no-symlink/ext4/free-space checks run before any
dispatch and every100steps/save; oversized file/failure is retained, never
deleted to hide a problem. Emergency only after closed4primitive+scheduledVAL
boundary; real1000/2000/3000 are immediately reloaded and recursively checked.

Parent author invocation ONCE from
ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024:

```text
python3 scripts/supervise_quick_ldm.py --mode witness --run ldm_kernel_author_v2
```

Only after actual author exit0/closedwriters/source/spec/corpus/VAE/receipt/
resource SHA evidence is available and common GPU lock is free, a fresh agent
must inspect this document and the fixed profile, then execute verbatim ONCE:

```text
python3 scripts/supervise_quick_ldm.py --mode witness --run ldm_kernel_document_v1
```

Use a durable wrapper and record its actual subprocess exit. Do not rerun the
author, rebuild data, copy cache again, install packages, change code/spec,
stop another worker or train. Preflight the already verified VAE receipt/exit
and checkpoint SHA, accepted corpus/cache/real source bytes, old19 frozen
files, shared GPU/RAM/disks/common lock and absence of this document run.
Preserve all source/spec pre/post bytes and original failures, if any. A missing
kernel is a failure to inspect, not permission to invent counts or launch twice.

This witness performs4 actual AdamW updates on real512×4primitive batches,
finite gradients and0 validation updates. It roundtrips the denoiser+Adam in
memory, directly compares ALL saved RNG states, checks exact nextbatch replay
and pending validation, verifies the VAE remains frozen/unchanged. It also runs
the real five-step CFG5 sampler with12 validation histories×4 autoregressive
primitives, no posterior future input for sampling. Those4 diagnostic updates
are discarded, not reused for LDM training. No persistent diagnostic LDM state
or new SDK/physics. The one shared frozen-VAE cache copy is an expected asset.
The sampling Huber is a custom diagnostic, not semantic or paper benchmark.

After both actual witnesses pass, parent may change ONLY train_authorized and
bind witness receipt/exit metadata, then launch ONCE:

```text
python3 scripts/supervise_quick_ldm.py --mode train --run quick_ldm_train_v1
```

Freeze all three LDM sources and active spec while training. Starting is not
completion: require final3000 receipt, actual exit0/resources/closedwriters,
all snapshots retained/backed up, genuine complete5-step samples with
history2→future8. Generated-reference physics execution and text-switch
closed-loop need separate real validation; reference-tracking movies do not
prove a text generator or final deployable WBC. User already authorized quick
automatic continuation, no repeated confirmation required.
