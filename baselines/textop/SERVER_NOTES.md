# TextOp environment ledger

### env: textop@92b1d1b1
how: isolated venv on nlp03, shared base read only.
tier: CPUs 4, GPU1 A6000, CUDA allocation cap 65%.
status: spec written BEFORE environment build; validation pending.
source: TeleHuman/TextOp ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22
assets: Yochish/TextOp-Data bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9

### env: textop@c5f7e5fa
Spec corrected: isaac_utils is at upstream/deps/isaac_utils, not under TextOpRobotMDAR. Original failed build log retained. Rebuild pending.

### env: textop@9d7f370d
Build metadata fix: pin setuptools68.2.2/wheel0.42.0 before editable packages. Older base setuptools incorrectly labeled RobotMDAR UNKNOWN. Isolated overlay only; base not modified. Validation pending.

validated: spec9d7f370d seeded official MVAE CUDA forward/backward passed; fresh independent doc witness_v2 passed verbatim SSH invocation, exit0, no divergence. No trained result claimed by environment witness.

validated: spec9d7f370d fresh independent doc witness_v3 passed verbatim SSH invocation after corrected training-observability documentation, exit0, no divergence. Shared environment unchanged; attempt1 preserved, attempt2 uses official skip behavior and separate counters.

### env: textop-data-prep@87f577cc
Warm reuse packages from gmr_amass_20261021 (old dependency spec141547bf), no package/environment mutation. New spec records licensed body-model availability and paper69D target. Source and model hash witness pending. Resources CPU4, no CUDA. MVAE57D is SIGSTOP-held pending data preparation.

Validation update 2026-10-25: author and fresh document-following executions both exited0 with TEXTOP_GMR_BODY_WITNESS_PASSED, SMPL-X num_betas16 and finite GMR qpos[36]. Results/data_preparation_20261025 retains both witnesses and explicit canonical-spec hash87f577cc versus file-byte SHA2566e7f21352c2839d62498a068103cfedcd3949f5416b1ca075477f6bdbef94591. The hash formats represent different serialized inputs and are not contradictory. Six complete-motion processing checks passed; initial errors remain recorded. This certifies CPU preprocessing only, not dynamic tracking quality or full paper dataset readiness.

Data scheduling: server batch500 PID3257584; CPU-only, four threads. Continuation wrapper PID3266979 waits for it to finish successfully, then processes143 additionally matched ACCAD inputs; no overlapping retarget batches, RL launch, or generator resumption. Local archive extraction PID69024 selects1,653 indexed BMLmovi/EyesJapan inputs from already licensed SHA-verified D-drive archives into C-drive private storage with6GiB reserve. No source archive or dataset deletion is performed. Do not put these private data directories in Git.
