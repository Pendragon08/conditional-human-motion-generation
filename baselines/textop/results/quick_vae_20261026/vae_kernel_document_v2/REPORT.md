# Independent Quick VAE document witness v2

PASS: the documented invocation was executed exactly once from the documented ROOT. Supervisor and child exited 0; child 4039476 was reaped normally after 11.424 s. Status: QUICK_VAE_WITNESS_PASSED. No document-versus-reality divergence was observed.

Command: `python3 scripts/supervise_quick_vae.py --mode witness --run vae_kernel_document_v2`

Exactly 4 real diagnostic Adam updates, 0 skipped updates, all 4 gradients finite, 4 validation primitives and 0 validation updates. EMA was disabled and the runtime asserted its EMA dictionary was empty. Exact model/Adam memory roundtrip and next-batch replay passed. All RNG streams are saved/restored by the audited helper (Python, NumPy, Torch CPU, all visible CUDA devices, train/val generators); the memory witness reports restored RNG/next-batch and tests next-batch equality but does not separately assert recursive equality of every RNG state. The payload remained in memory: 513522813 bytes. Diagnostic weights were discarded. There were 0 persistent checkpoints before and after.

The new training fixed-quality-batch block was reviewed independently against the preserved v1 bytes. Its only code change captures all RNG after optional checkpoint restoration, uses a deterministic seed and val generator, draws a fixed validation batch, restores every captured state, and recursively asserts equality before the first training update. Runtime.quality separately captures/restores those same states. This train branch was audited, not dynamically executed by this witness. No extra model experiment or mock was run.

Pre/post checks confirmed all original 19 frozen files, all added bound sources, spec, data acceptance, corpus receipt, and all 8 real corpus artifact SHA256 values were unchanged. Local and remote source bytes matched. Warm environment canonical hash remains 934d8221. The accepted quick corpus remains 42 originals (31 train / 11 val), 174 annotation pairs, 74 CLIP texts; train-only 69D statistics, no mirrors, full-paper acceptance false.

GPU1, CPU0-3, Torch allocator cap 0.65 were used. Before/after GPU1 free memory was 40726 MiB. Measured peak Torch allocation: 3930611712 bytes (3.6607 GiB). The 11.42 s duration yielded three resource samples, too short for a sampled NVML process peak; the peak is the actual Torch counter. RAM and persistent free space stayed above the declared guards. No active personal GPU worker existed before launch; the old PID 3237858 stayed stopped. The common lock was available again afterward.

No training, LDM, SDK, physics, corpus rebuild, dependency install, source/config edit, Git or publication occurred. Train authorization remained false. The remote ROOT has no SERVER_NOTES.md; the local BASE ledger was read.

Receipt SHA256: 4e4c63b3ca1ef6ca5b541b70116339da7d9ea49c3ca4e492c918bc1dc3a18acb

Exit SHA256: 4f3972850f5213befa94c465ab5759173a46a793bb3f74517c2cd5d4def1e1bb

Process-exit SHA256: 53459b62f5972b0cdf0af80662617f7e01d16e879e95f929f68a048844b369a7

Resources SHA256: 2e2f595f383f02d67680dd5fd96ff73bcb61f225e274130bd1cc20e0f6e474bb

Entrypoint SHA256: 660aaec92ca3e7a6b97716739b9af981fc615deb2781ad074ee242bf59ecda72

Spec SHA256: 6f46ee98484b41c3604dc7793b4457e4dc4c4cd606eecb058f2d0849e8f189e8

Corpus-receipt SHA256: d0a44b975c8f390f25dc5e95c414b3b8354dacacc19d5673d7a86e6d21c9c8f1

Data-acceptance SHA256: 5f4f58c2fd9e98f36a23fb44de90d7a24aa0a7ab2fbf63207ea35eacae7cadc6

Private raw logs/receipts and all pre/post snapshots are retained in this dedicated directory, not BASE/results. Local archival issues (Windows default GBK decoding and carriage-return normalization in the progress log) were resolved using explicit UTF-8 and exact byte transfer; they did not trigger a second witness invocation. Their details are retained in independent_v2_diff.json and independent_verification.json.
