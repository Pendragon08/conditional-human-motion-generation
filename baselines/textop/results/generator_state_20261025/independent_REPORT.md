# Independent generator optimizer/checkpoint document witness

The bounded state diagnostic passed: actual exit 0, success sentinel and all 75 independent checks true. No observed source/document divergence. The run performed 16 short candidate-data updates and produced diagnostic checkpoints only; it does not establish formal training or model quality.

## One exact run and resource checks

Read only GENERATOR_STATE_ENVIRONMENT.md, configs/generator-state-witness.json and SERVER_NOTES.md. Executed the documented PowerShell SSH invocation once unchanged for generator_state_document_v1; see documented-command.ps1.

Before dispatch, read-only resource queries confirmed GPU1 free 40,726 MiB (above 20 GiB) and data disk free 3,879,411,712 bytes (above 2 GiB). No cleanup or process actions were used. Local session 4736 returned actual exit 0 and TEXTOP_GENERATOR_STATE_SUPERVISED_EXIT0. Supervisor child PID 3635969, child_exit_code=0, timed_out=false, elapsed=58.73029207997024 s; receipt elapsed=57.728795520961285 s. Receipt and original log contain TEXTOP_GENERATOR_STATE_WITNESS_PASSED.

Batch512, four primitives, feature69; spec history2/future8. Receipt CPU affinity [0,1,2,3]. Torch allocator 0.65 is declared in the exact spec; the public receipt does not separately expose an observed-fraction field. GPU1 driver samples rise during this run; GPU0 remains constant.

No code/package/shared-source changes, retries, repairs, view rebuild, deletion, GitHub actions or other-chat messages occurred. Existing GMR/backup/FK jobs were not acted on. Only public diagnostics, read-only resource output and hashes/sizes were collected.

## Actual updates and validation

| Stage | Optimizer | Actual updates / attempts / skips | Training draws | Validation draws | Adam parameter step min/max |
|---|---|---|---:|---:|---|
| VAE | Adam | 8 / 8 / 0 | 2 | 1 | 8 / 8 |
| LDM | AdamW | 8 / 8 / 0 | 2 | 1 | 8 / 8 |

Both stages actually allocate optimizer state tensors and EMA state. Model parameters change, and public before/after parameter hashes differ. VAE weight decay=0; LDM AdamW weight decay=0.01.

Each stage's public trace shows:
- Training before checkpoint: four primitives at steps/actual updates 1–4.
- Restored pending validation: four primitives, each manager step=4, actual updates=4, is_evaluation=true.
- Continued training: four primitives at steps/actual updates 5–8.
- Positive finite losses and true finite-gradient flags throughout.

pending_validation_completed_without_optimizer_updates=true for both. The saved checkpoint is step4; final diagnostic manager step is8. Final8-step weights are not a saved formal trained model. The LDM's frozen diagnostic VAE remains unchanged; the trainable VAE stage correctly changes its own parameters.

## Exact restoration and next batch

For both stages all public fields are true:
model_restored_exactly, optimizer_restored_exactly, ema_restored_exactly, rng_and_generator_restored_exactly, clock_restored_exactly and pending_evaluation_fields_restored.

next_batch_and_rng_probe_identical=true and mismatched_binding_rejected=true in both stages. The procedure identifies Python/NumPy/Torch/CUDA RNG and explicit training/validation batch generators in this test, including the mismatched57D binding rejection.

These conclusions use the canonical-source witness's public exact-restoration assertions plus explicit continuation/validation traces. This independent agent did not copy or inspect private tensor/RNG arrays. The checkpoint boundary is between completed four-primitive batch draws; cached mid-batch/history continuation and later rollout-curriculum phases are outside this test.

EMA is enabled only for diagnostic state stress: ema_diagnostic_fixture=true, release_ema_default=false. Formal recipe defaults are preserved.

## Checkpoint files independently verified on server

| Diagnostic file | Actual bytes | Remote SHA256 |
|---|---:|---|
| vae/state_after4.pth | 684683326 | 5785d063f37ad30f639760a79bad5ffe0e69c30507a24243df90c895f8e5f74b |
| ldm/state_after4.pth | 305124157 | bdf540b976b26a6ca6e47fc3a595314970211a2419386b72c561d8f90e786be3 |

Remote sha256sum and stat outputs independently match the receipt. No checkpoint bytes were copied. Total listed checkpoints=989,807,483 bytes, below run budget 1,100,000,000; each is below the 900,000,000-byte safety budget. Post-run independent disk query reports 2,889,506,816 bytes free; receipt reports 2,889,515,008 at its earlier sample. Both exceed 2 GiB.

## Source/spec identity

All documented source/state/codec/sampler hashes match receipt and independent remote sha256sum, without implementation inspection. Neural-state spec and base spec additionally match exact local file hashes.

- Source: aa511f445a65cad244246c1144b03cc8cdc16b61a242e96bfc6ddbe3224fe072
- State: 46ef2ade1e76ba90815d1afb6c380b049030afb87c6a74e2d0b70cd3db32b94e
- Spec: bd3f7a06e57235c8fcfcc1734b046f48d896f739d755d28e231168ace17efc82
- Base spec: 30bcaa928d4e7ed4cf7359d88dfe909334718957fffbd07bc98510467612d421
- Codec: cb97b9a2dd603979ec512dddca07be88ca0c9144a82b1eb935fc28dab6a0486f
- Sampler: cf5c1e3f27665e7fc4d4aa1ede6bcd1931e3563409ad766ff67a576cc0c7b60d
- Fixed view: c4875fa612707507ae8de28c15ca4f1160cff5293467cb91c7801d43184d3b23

The fixed private view hash matches document/spec/receipt; the underlying private candidate PKLs, arrays, texts and caches were not opened or copied by this agent. Plan hash is retained from the public receipt only.

## Memory and scope

Measured Torch allocated peaks: VAE=4109051904 bytes (3.827 GiB), LDM=1979146752 bytes (1.843 GiB). Reserved peaks are 4357881856 and 2057306112 bytes. These include this diagnostic's actual optimizer/EMA allocation, but do not establish all-phase/full-corpus training capacity or quality.

Whole-device samples include other processes: GPU1 initial7823 MiB, sampled peak12699 MiB of49140 MiB; GPU0 constant12223 MiB. These sampled totals are not exclusive job memory.

dataset_accepted=false, physical_filter_executed=false, full_generator_training_started=false, formal_generator_checkpoint_produced=false; spec formal_training_started=false. model_weights_changed_by_diagnostic=true is expected for this bounded test on unfiltered candidates. The LDM uses a short diagnostic VAE checkpoint, not a fully trained VAE. Formal200k/300k training, final accepted corpus, semantic/physical quality and later curriculum remain untested.

output_writers_closed=true. No IsaacSDK close workaround or SDK fallback is involved or asserted.

Saved command/stdout, public receipts/log, remote checkpoint hash/size proof, source hashes, resource preflight, verification.json and this REPORT.md are all in this directory. No private checkpoint or array files are present.
