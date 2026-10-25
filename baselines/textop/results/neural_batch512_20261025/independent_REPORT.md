# Independent neural batch512 zero-update document witness

PASS for the documented numerical witness: all 42 recorded checks true, no observed divergence. This uses the unfiltered candidate2296 view and random model weights. It does not establish accepted data, semantic quality, full-state training memory or training readiness.

## One exact run and actual exit

Read NEURAL_BATCH512_ENVIRONMENT.md, SERVER_NOTES.md and configs/neural-batch512-spec.json. Executed its exact PowerShell SSH invocation once unchanged for neural512_document_v1. The command is retained in documented-command.ps1.

Local session30558 returned actual exit0 and TEXTOP_NEURAL_BATCH512_SUPERVISED_EXIT0. Supervisor childPID3582485, child_exit_code0, timed_out=false, elapsed34.19710926711559s. Numerical receipt elapsed33.45101875625551s and status TEXTOP_NEURAL_BATCH512_WITNESS_PASSED; the original log contains the same marker.

Only the three listed public receipt/log artifacts and requested hash outputs were collected. No implementation inspection, edits, retries, package changes, candidate rebuild, private data copying, publication, actual training or other process actions occurred. No actions were sent to GMR/FK queues or paused57D.

## Numerical requirements

batch_size512, primitives4, feature_dim69, documented history2/future8. Receipt names the unmodified pinned MVAEManager.calc_loss and DARManager.calc_loss bodies.

- VAE:8 actual backwards, four primitives under ground_truth history and four under detached_generated history. Loss range 0.41504889726638794–0.4217350482940674; gradientL2 range 1.614615559577942–1.666118860244751;247 finite nonzero-gradient tensors per backward.
- LDM:4 actual backwards. Loss range 1.0527881383895874–1.0567623376846313; gradientL2 range 1.1183593273162842–1.2822695970535278;108 finite nonzero-gradient tensors per backward.
- All12 losses and gradient norms are finite and positive.
- vae_parameters_unchanged=true, denoiser_parameters_unchanged=true, frozen_vae_has_no_gradients=true.
- optimizer_constructed=false, optimizer_updates0, optimizer_and_ema_states_actually_allocated=false.
- Sampling settings:batch512,16frames/sample,two8frame segments,5steps,CFG5. The program pass marker covers its finite-output assertions; the public sampling receipt does not expose raw arrays or a separate finiteness field.

Parameter-preservation evidence is the required public aggregate flags. Private weights were not copied or independently rehashed.

## Memory, resources and exclusions

CPU affinity=[0,1,2,3]. Spec declares Torch allocator0.65; the public receipt does not separately expose an observed allocator-fraction field. GPU name NVIDIA RTX A6000; supervisor GPU1 samples increase during this job while GPU0 samples stay unchanged.

Measured Torch peaks:
- VAE allocated3429537280bytes (3.194GiB), reserved3607101440bytes.
- LDM allocated1681969664bytes (1.566GiB), reserved1742733312bytes.

Whole-device samples include other processes:GPU1 initial7823MiB/sampledpeak11983MiB of49140MiB;GPU0 constant12223MiB. Exact sampled losses/timing/memory need not be bitwise identical, and values above are actual observations.

Optimizer/Adam/EMA state was not allocated. Receipt's extra Adam/EMA-byte estimates are estimates only. Measured peaks exclude real optimizer/EMA state, checkpointing and the final accepted corpus. They cannot be promoted to full-training memory. Sampling's batching duration is not single-stream deployment latency.

## Hash checks without source inspection

All documented hashes match the public receipt and independent remote sha256sum. Both spec files also match their exact local copies.

- Source:26cbf30cf75afbdfdee97ed3bdf26de11f91d82b67bf35690b719844b237410d
- Neural spec:2a91a7f6ec63732b90a666b52f615264ee1117b467e41f7a49485b53652896ca
- Base spec:30bcaa928d4e7ed4cf7359d88dfe909334718957fffbd07bc98510467612d421
- Codec:cb97b9a2dd603979ec512dddca07be88ca0c9144a82b1eb935fc28dab6a0486f
- Sampler:cf5c1e3f27665e7fc4d4aa1ede6bcd1931e3563409ad766ff67a576cc0c7b60d
- Fixed candidate view receipt:c4875fa612707507ae8de28c15ca4f1160cff5293467cb91c7801d43184d3b23

The fixed view hash matches the document/spec/public receipt; its private underlying view/index/cache was not read or copied.

## Scope and shutdown

dataset_accepted=false, physical_filter_executed=false, full_generator_training_started=false, model_weights_trained=false;output_writers_closed=true. Candidate source count2296 is recorded in the receipt.

No IsaacSDK is used by this diagnostic. No SDK native-close fallback is asserted, and no teardown stack was requested or collected. Direct process exit0 is the shutdown evidence.

Saved files:documented-command.ps1,stdout.log,local-execution-receipts.jsonl,receipt.json,neural_batch512_neural512_document_v1_exit.json,neural_batch512_neural512_document_v1.log,source_and_specs_remote_sha256.txt,verification.json,evidence-readback-receipts.jsonl and this REPORT.md.
