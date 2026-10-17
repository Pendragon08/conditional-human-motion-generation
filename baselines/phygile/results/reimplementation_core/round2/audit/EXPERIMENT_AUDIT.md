# Experiment Audit Report — Revised-Scope Re-audit

**Date:** 2026-10-17  
**Auditor:** GPT-5.6-Sol (fresh same-family reviewer, read-only, provisional)  
**Reviewer metadata:** `reviewer_model=gpt-5.6-sol`, `review_independence=same-family`, `acceptance_status=provisional`  
**Overall verdict:** **WARN**  
**Scope:** the specifically listed implementation, core verification, public text-interface verification, and synthetic two-update training-plumbing witness files, plus the same previously inspected PhyGile paper. No source was edited or executed, no remote training was launched, and no GPU was used.

## Immediate blocking defect

The documented public-checkpoint download path and the training entrypoint do not agree on where the provenance manifest lives:

- `reimplementation/fetch_text_encoder.py:6-18` writes the model files under `cache/text/distilbert-base-uncased/` but writes `download.json` under `results/reimplementation_text/`; its completion/failure updates remain at that result path (`reimplementation/fetch_text_encoder.py:29-34`).
- `reimplementation/train.py:117-121` resolves `text_encoder_path` and then unconditionally reads `encoder_path/download.json` from inside the model directory.
- The example points `text_encoder_path` at the cache directory (`reimplementation/train_config.example.json:5`), and the README identifies the authoritative manifest as `results/reimplementation_text/download.json` (`reimplementation/README.md:31-33`).

Therefore, a cache produced only by the supplied fetcher cannot enter `train.py` as documented: training fails before model construction because `cache/text/distilbert-base-uncased/download.json` was never written. The successful two-update witness proves that its particular server cache contained an additional provenance file, but that file is outside this bounded evidence set and was not produced by the shown fetcher. This blocks a claim that the standard fetch→train workflow is closed. Unify the provenance path or have the fetcher atomically copy its completed manifest into the model directory.

## Overall assessment

The revised README now states the core scope accurately. The exact current Python sources are bound to the core, text-interface, and training-witness results by matching SHA-256 values. The core has twelve useful deterministic checks, including a full-size CPU forward. The text evidence supports a real interface run with a recorded public DistilBERT checkpoint revision, but not the author's exact checkpoint and not a trained motion generator. The training witness is a real two-update CPU plumbing test over nine frames from two explicitly artificial arrays using a reduced model; it is not scientific motion training, convergence evidence, or motion-quality evaluation.

The verdict remains WARN because the standard text-provenance handoff to training is broken, several successful-run binaries/arrays are outside the bounded evidence set, the online fetch script resolves moving repository HEAD rather than enforcing the recorded revision, and no research dataset, full-size training, motion evaluation, or physics loop exists.

## A. Ground-truth provenance: PASS

No motion-quality ground truth is used or claimed.

- The core test declares itself a numerical/gradient check rather than a performance test (`reimplementation/test_core.py:1`) and uses seeded random tensors (`reimplementation/test_core.py:11-18`, `reimplementation/test_core.py:30-40`, `reimplementation/test_core.py:84-90`). Core result flags are `trained_model=false`, `dataset_used=false`, and `physics_simulation_tested=false` (`results/reimplementation_core/verification.json:21-24`).
- The text test uses two real English prompts and a public pretrained text encoder only to verify freezing, determinism, padding behavior, token shape, and a finite untrained-denoiser forward (`reimplementation/test_text_encoder.py:15-26`). Its result explicitly says `interface_check_only`, `paper_exact_checkpoint_confirmed=false`, `motion_generator_trained=false`, and no dataset/quality/physics evaluation (`results/reimplementation_text/verification.json:2-6`, `results/reimplementation_text/verification.json:25-35`).
- The training witness manifest labels every array as `synthetic_unit_test`, says rotations are not valid robot rotations, and identifies captions as artificial interface text (`results/reimplementation_training_witness/manifest.json:2-6`, `results/reimplementation_training_witness/manifest.json:8-46`). The witness result records two CPU optimizer updates, no real motion, no full-size generator, and no quality or physics evaluation (`results/reimplementation_training_witness/verification.json:2-15`).
- `train.py` rejects a manifest whose purpose differs from the requested run and rejects `synthetic_unit_test` mapping under `research_training` (`reimplementation/train.py:23-38`). The witness directly exercises this rejection (`reimplementation/test_training_entrypoint.py:22-29`).

The synthetic numerical references are valid unit-test or plumbing witnesses because they are explicitly labeled. They are not independent empirical ground truth and must not be promoted into motion-performance evidence.

## B. Score normalization: PASS

No evaluation score is divided by prediction statistics or rescaled toward one. The only normalization is feature preprocessing fit exclusively on training arrays (`reimplementation/train.py:106-114`; `reimplementation/features.py:29-46`). The witness checks that the saved mean comes from the two training arrays rather than the test array (`reimplementation/test_training_entrypoint.py:51-52`; `results/reimplementation_training_witness/verification.json:10-15`). The two logged losses are raw `x0_mse + 0.01 × load_balance` components (`results/reimplementation_training_witness/run/training_losses.jsonl:1-2`), not normalized performance metrics. Their small two-step decrease has no scientific interpretation.

## C. Result existence, numbers, and hashes: WARN

All named files exist and parse. The following internal checks pass:

- Core exit code is zero (`results/reimplementation_core/exit_code.txt:1`), twelve `PASS` lines occur in the log (`results/reimplementation_core/stdout_stderr.log:1-12`), and the JSON repeats the same test labels, 146,169,702 parameters, and negative experiment flags (`results/reimplementation_core/verification.json:2-24`). All nine embedded Python hashes (`results/reimplementation_core/verification.json:25-34`) equal the current files.
- The text stdout is byte-identical to `results/reimplementation_text/verification.json`; the run receipt records the requested command, exit code zero, and timestamps (`results/reimplementation_text/text_interface_local_transfer/receipt.json:2-10`, `results/reimplementation_text/text_interface_local_transfer/receipt.json:88-92`). The recorded public model is `distilbert/distilbert-base-uncased` revision `12040accade4e8a0f71eabdb258fecc2e7e948be` (`results/reimplementation_text/download.json:2-6`), with file sizes and hashes at `results/reimplementation_text/download.json:7-56`. The model weight hash equals the recorded publisher LFS hash (`results/reimplementation_text/download.json:37-41`). The original remote fetch failure is preserved (`results/reimplementation_text/pipeline.json:2-9`) and the local-transfer recovery is separately labeled (`results/reimplementation_text/pipeline_recovery.json:2-5`).
- Training receipt manifest/config hashes exactly equal the current bounded files (`results/reimplementation_training_witness/run/receipt.json:2-10`). The receipt says two updates over two sequences/nine frames on CPU (`results/reimplementation_training_witness/run/receipt.json:9-19`, `results/reimplementation_training_witness/run/receipt.json:31-34`), matching the verification (`results/reimplementation_training_witness/verification.json:2-15`) and two loss rows (`results/reimplementation_training_witness/run/training_losses.jsonl:1-2`). The wrapper receipt records exit code zero (`results/reimplementation_training_witness_log/receipt.json:2-10`, `results/reimplementation_training_witness_log/receipt.json:88-92`). All nine embedded Python hashes match current files.

The status is WARN because the bounded set omits the actual cached text-model files, the three synthetic `.npy` inputs, `normalization.npz`, and `model_000002.pt`. Their hashes are recorded or referenced, but the reviewer was explicitly confined to the listed files and therefore could not independently rehash those bytes. In particular, the checkpoint hash at `results/reimplementation_training_witness/verification.json:17` is not independently confirmed here. This is an evidence-boundary warning, not evidence that the files are fabricated.

## D. Dead code and exercised paths: WARN

The revision materially improves coverage:

- The reduced model now checks finite, nonzero gradients for every individual expert slice in both weight/bias banks and every gate parameter (`reimplementation/test_core.py:40-51`).
- Per-step boundary reinjection is observed directly rather than inferred from final overwrite (`reimplementation/test_core.py:68-82`).
- ASFO zero-count semantics and cap validation are tested (`reimplementation/test_core.py:92-103`), multi-token chunk equivalence is tested (`reimplementation/test_core.py:105-112`), actual-denoiser CFG/null dropout is tested (`reimplementation/test_core.py:114-125`), and the default 146M-parameter model completes a finite CPU forward (`reimplementation/test_core.py:127-134`).
- The real pretrained text adapter is loaded and exercised (`reimplementation/test_text_encoder.py:15-26`), and the training entrypoint performs two actual optimizer updates and reloads its checkpoint (`reimplementation/test_training_entrypoint.py:43-50`).

Remaining unexercised or incompletely exercised paths are material but honestly scoped:

- The supplied online fetcher failed in the recorded environment (`results/reimplementation_text/pipeline.json:2-9`), and its standard output-path contract is inconsistent with `train.py` as described in the blocker above.
- The fetcher calls `model_info(repository)` without a requested revision and only then records current HEAD (`reimplementation/fetch_text_encoder.py:9-13`). The completed artifact is commit-recorded, but future executions are not pre-pinned to that revision.
- The training witness disables ASFO (`results/reimplementation_training_witness/config.json:33-38`), so the `PairedMotionData` repetition/count-policy branch at `reimplementation/train.py:67-84` and the example's enabled ASFO setting (`reimplementation/train_config.example.json:19`) have no end-to-end witness.
- Only the `unit_test` training route is executed. No `research_training` run, full-size optimizer step, validation/test evaluator, checkpoint resume, generation evaluation, or physics loop is present. The overlap witness changes a declared `source_group` (`reimplementation/test_training_entrypoint.py:30-34`); the separate identical-file-hash rejection branch exists in source (`reimplementation/train.py:39-64`) but is not directly tested.

## E. Scope assessment: WARN

The revised scope language is substantially correct:

- The README denies trained motion-generation weights and full-PhyGile reproduction (`reimplementation/README.md:1-5`), classifies the core tensors as numerical checks (`reimplementation/README.md:7`), documents architecture/null/sampler assumptions (`reimplementation/README.md:9-13`), and lists the missing data, mapping, generator evaluation, GMT, simulation, and PPO work (`reimplementation/README.md:15-17`).
- It distinguishes the real pretrained text-interface test from a motion-quality experiment and says the exact author checkpoint is unknown (`reimplementation/README.md:27-35`).
- It explicitly labels optimizer, schedule, budget, and ASFO settings as unexecuted proposals rather than author settings (`reimplementation/README.md:37-41`) and labels the two-update witness as synthetic/unit-test-only (`reimplementation/README.md:73-75`).

The scope status remains WARN solely because `reimplementation/README.md:65-71` presents a future training invocation after dependencies/data are available, while the supplied fetcher cannot create the provenance file where `train.py` requires it. Until the blocker is fixed, “training entrypoint available” needs the qualifier that an additional, manually placed cache-local `download.json` is required.

## F. Evaluation classification: PASS (classification only)

| Artifact | Classification | Exact claim ceiling |
|---|---|---|
| Core verification | `synthetic_proxy` / `deterministic_unit_check` | Formula, shape, mask, gradient, and finite-forward implementation consistency |
| Text verification | `interface_check_only` using recorded public pretrained weights | Frozen/padded/deterministic DistilBERT interface and finite untrained-denoiser integration |
| Training witness | `synthetic_proxy` / `training_plumbing_unit_test` | Manifest rejection rules, train-only normalization, two reduced-model optimizer updates, and checkpoint serialization/reload |

None is `real_gt`, `simulation_only`, or `human_eval`. The text-interface test is genuine pretrained-model interface evidence, but it does not train or evaluate the motion generator. The two-update witness is genuine execution of training plumbing, but its artificial tensors/text, reduced model, nine training frames, two updates, and absence of motion evaluation prevent any scientific training, convergence, quality, or reproduction claim.

## Core equations and architecture

The paper comparison from the first audit remains applicable; the relevant source hashes for `model.py`, `diffusion.py`, and `text.py` are unchanged.

- **Eq. 7: PASS as a primitive.** `reimplementation/diffusion.py:20-36` implements x0 regression corresponding to the paper at `keyword_literature_update_20261010/sources/phygile.txt:523-536`. The beta schedule remains caller-supplied because the author schedule is unpublished (`reimplementation/diffusion.py:6-15`; `reimplementation/README.md:11`).
- **Eqs. 8–9: PASS with disclosed assumptions.** Parameter mixing and attention masks are at `reimplementation/model.py:38-55`; placement/head averaging is at `reimplementation/model.py:67-75`; paper definitions are at `phygile.txt:537-572`. Multi-token chunking now has direct evidence (`reimplementation/test_core.py:105-112`).
- **Eqs. 10–11: PASS as literal multiplier arithmetic.** Positive-integer cap validation is at `reimplementation/features.py:48-58`; zero-count semantics are now tested (`reimplementation/test_core.py:97-103`) and carefully distinguished from sampler policy (`reimplementation/README.md:13`). Actual oversampling/mirroring remains unimplemented and is not claimed.
- **Eq. 12: WARN as an explicit reconstruction assumption.** The fixed-mask sampler is at `reimplementation/diffusion.py:47-70`; per-step reinjection is tested at `reimplementation/test_core.py:68-82`. It still does not implement the paper's GMT-derived physical prefix, MPJPE selection, simulator loop, or PPO refinement (`phygile.txt:603-631`), and the README now says so (`reimplementation/README.md:5`, `reimplementation/README.md:11`).
- **Eqs. 13–14: PASS algebraically.** `reimplementation/diffusion.py:38-45` and `reimplementation/test_core.py:53-61,114-125` cover TokenProbe and actual-denoiser branches; paper formulas are at `phygile.txt:1167-1187`. Retaining token-valid masks and learned-bias/length effects is now documented (`reimplementation/README.md:11`).
- **Architecture: PASS for major published dimensions; provisional reconstruction overall.** Published dimensions at `phygile.txt:1089-1120` match `reimplementation/model.py:8-20,29-36,57-85`; the default model now completes a finite CPU forward (`reimplementation/test_core.py:127-134`). The paper-unspecified positional encodings, AdaLN/residual ordering, biases, attention reduction/dropout, and sampler choices are disclosed at `reimplementation/README.md:9-13`.

## Public text interface versus motion-generator training

The text artifacts support this narrow statement: a recorded public `distilbert-base-uncased` revision was used to produce deterministic 768D token tensors, stayed frozen/eval under `.train()`, behaved consistently under padding, and fed an untrained full-size denoiser to a finite output (`reimplementation/test_text_encoder.py:15-34`; `results/reimplementation_text/verification.json:2-35`). The result itself says the exact paper checkpoint is unconfirmed (`results/reimplementation_text/verification.json:4-6`).

They do **not** support any of the following: that PhyGile's authors used this exact revision, that a motion generator was trained, that token semantics improved generated motion, or that generated motion has measurable quality or physical validity. The actual checkpoint bytes are also outside the bounded re-audit, so the current-byte claim rests on the recorded runtime hash checks rather than a second independent hash pass.

## Author-published versus proposed training choices

| Choice | Status | Evidence |
|---|---|---|
| 262D representation; 8 layers; width 512; FFN 1024; 4 heads; dropout 0.1; 12 experts; gate shape; mask γ/β | Author-published | `phygile.txt:1089-1120`; `reimplementation/model.py:8-20` |
| x0 objective | Author-published | `phygile.txt:523-536`; `reimplementation/diffusion.py:20-36` |
| Text-condition dropout 0.1 and CFG scale 2.5 | Author-published | `phygile.txt:1167-1187`; `reimplementation/diffusion.py:24-45` |
| Load-balance coefficient 0.01 | Author-published | `phygile.txt:1117-1120`; `reimplementation/diffusion.py:24-36` |
| AdamW, learning rate 1e-4, β=(0.9,0.999), weight decay 0, gradient clip 1 | Independent proposal | `reimplementation/train_config.example.json:13-16`; paper search contains no generation-training specification for these values |
| Linear β schedule 1e-4→0.02 over 1000 steps | Independent proposal | `reimplementation/train_config.example.json:12`; `reimplementation/README.md:39` |
| Batch 4, 100,000 updates, save every 1,000, max 300 frames | Independent proposal | `reimplementation/train_config.example.json:7-10`; `reimplementation/README.md:39` |
| `rho_max=4`, `keep_at_least_one`, mirroring disabled | Independent operational proposal | `reimplementation/train_config.example.json:19`; paper prints ρmax symbol but no audited value at `phygile.txt:579-591` |
| Reduced 2-layer/16-wide/3-expert model, 8 diffusion steps, two CPU updates, ASFO disabled | Synthetic witness only | `results/reimplementation_training_witness/config.json:2-38`; `results/reimplementation_training_witness/verification.json:2-15` |

The proposed values must not be attributed to the authors or treated as reproduced hyperparameters.

## Split-leakage assessment

`load_manifest` hashes every referenced file and collects both `source_group` and file hash per split (`reimplementation/train.py:39-60`), then rejects overlap across any lexically paired split names (`reimplementation/train.py:61-64`). Normalization is fit only after extracting the training entries (`reimplementation/train.py:52-65`, `reimplementation/train.py:106-114`). The witness confirms declared source-group overlap rejection and train-only normalization (`reimplementation/test_training_entrypoint.py:27-34`, `reimplementation/test_training_entrypoint.py:51-57`).

This is a sound declared-group/exact-byte guard, not semantic leakage detection. It cannot detect two differently encoded/cropped derivatives of the same source unless the manifest author gives them the same `source_group`. The code trusts the manifest's provenance fields; no content fingerprint, motion similarity, subject/sequence identity, or annotation-overlap audit is implemented. Claims should say “declared source-group and exact-file-hash overlap rejected,” not “all split leakage eliminated.”

## Claim impact

- **Exact current sources passed twelve deterministic core checks — supported.**
- **A recorded public pretrained DistilBERT checkpoint passed a frozen text-interface test — supported with bounded-artifact and non-author-checkpoint qualifiers.**
- **The public checkpoint is the authors' exact text encoder — unsupported; explicitly marked false.**
- **The untrained full-size denoiser accepts real 768D token outputs and returns finite tensors — supported; this is interface evidence only.**
- **The training entrypoint completed two reduced-model CPU optimizer updates on explicitly synthetic arrays — supported.**
- **The standard supplied fetch→train workflow is ready — unsupported until the provenance-path blocker is fixed.**
- **A full-size motion generator was trained — unsupported and explicitly false.**
- **The two loss rows demonstrate learning, convergence, or motion quality — unsupported.**
- **Declared source-group and identical-file overlap across splits are rejected — supported by source; only the source-group branch has a direct negative witness.**
- **All semantic split leakage is prevented — unsupported.**
- **Example optimizer/schedule/budget/ASFO values reproduce author settings — unsupported; they are correctly labeled proposals.**
- **Any language-motion quality, FID/retrieval, physical executability, simulation, or full-PhyGile result — unsupported.**

## Required corrections

1. **Blocking:** reconcile the text provenance contract. Prefer an explicit `text_encoder_manifest` config path, or write/copy the completed manifest into the cache directory and document it. Add an end-to-end fetch-layout→train-loading test.
2. Make `fetch_text_encoder.py` accept or hard-code the intended commit revision rather than resolving moving HEAD with `model_info(repository)`. Record and enforce repository, revision, status `COMPLETED`, file list, and manifest hash in `train.py` receipts.
3. For audit-complete witnesses, include the synthetic input arrays and checkpoint (or an independently generated inventory containing their sizes and SHA-256 values) in the bounded evidence set. Emit `download_manifest_sha256` and verified checkpoint-file hashes directly in result receipts.
4. Add an end-to-end ASFO-enabled unit test covering both `literal_allow_zero` and `keep_at_least_one`, including the empty-dataset rejection. Keep mirroring disabled until implemented.
5. Add a distinct identical-file-hash cross-split rejection test and phrase leakage claims as declared-group/exact-hash checks.
6. Keep all text and training outputs classified as interface/plumbing checks. Do not interpret two synthetic optimizer updates or decreasing raw losses as a scientific motion experiment.

## Audited input SHA-256 hashes

| Input | SHA-256 |
|---|---|
| `reimplementation/README.md` | `d9ee7de9ed5e1cc6f8880c78ff44f4fdd87a5745d2e522e8464990f8a3f09fa3` |
| `reimplementation/model.py` | `76eb8baa876950a34d4b23b6fa92627515bf38072c9cadf35fbf8789a13e70ab` |
| `reimplementation/features.py` | `dee9e489f91b13217faa95a71bc5bc2f9122a1cc404639a3df9838edb6c7f9ac` |
| `reimplementation/diffusion.py` | `930a2cdb1967119dd51813c6838f58c062fc7d46447239b20cc15f4b638efc4c` |
| `reimplementation/text.py` | `22a04171f7e1218ff9b9ed768d6739ebed24c4072c683b78d60fde595b7fcf70` |
| `reimplementation/test_core.py` | `c12732d27dac9bdee04f6de4353972e94c271a6e08bc7b58608757d68dea402b` |
| `reimplementation/fetch_text_encoder.py` | `a8b2b5bd47493f967690d2807917966888e550f14c4c703cf91165e297c5503d` |
| `reimplementation/test_text_encoder.py` | `80ce12e90c2abafb63cadc5d687ad3a105141a6f6ee500a6ee54f9d1512fd004` |
| `reimplementation/train.py` | `47031d106376a1ddc4d2d3a46d1842101b4f028742aa79f77490bce86cc61baf` |
| `reimplementation/test_training_entrypoint.py` | `bd0dd2e0b5ea127c183825ddc3ba977b619f16b3d9ec3fae91b9b33afb5c55e4` |
| `reimplementation/train_config.example.json` | `74dbc8c4d91e928d1e3d116ffe96bac356c779f3242076943cda565bf3df3d5f` |
| `results/reimplementation_core/verification.json` | `ac8fe6e8a2321c11176cd3e1bfa88e26707113831b4ce402f8d1191ae822efb4` |
| `results/reimplementation_core/stdout_stderr.log` | `a6df1bbbeff79e8a0c1b10a801d31ebfb825c16df7624260f5b2d0dee83d212d` |
| `results/reimplementation_core/exit_code.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `results/reimplementation_text/download.json` | `76c3ed2cf6deef78acdb8c86b27a8133e259a3f752535472d31c548dbd73c136` |
| `results/reimplementation_text/verification.json` | `7a5391c9737833e2576902191227b783fe22d9222609a60124d348901d416246` |
| `results/reimplementation_text/pipeline.json` | `c0e187d62630d7bdfd9cee9f645c502b3637fcd0a545038b10756e77e71cf3d4` |
| `results/reimplementation_text/pipeline_recovery.json` | `6c017f0e13c66c8a546b4541bae1290a21ab3427cd3e654b9ea120165f7f40c9` |
| `results/reimplementation_text/text_interface_local_transfer/receipt.json` | `b5c76a540db9ecc116a06133c1af39a8ee647ea1c64c6fb8545eb0e96834ed15` |
| `results/reimplementation_text/text_interface_local_transfer/stdout_stderr.log` | `7a5391c9737833e2576902191227b783fe22d9222609a60124d348901d416246` |
| `results/reimplementation_training_witness/verification.json` | `9462d4d0a0c84075f75e9b5aa04d88ac4b8634225132b7fad2b1c20d79246fbc` |
| `results/reimplementation_training_witness/manifest.json` | `e13350f4bbd5a3e82f3259fffad4e9a261a6d5ee012029d3674f4ac623aaff9e` |
| `results/reimplementation_training_witness/config.json` | `391a22ea7710281e298fc03ee66d26e9f98d098df458f6e1942db752a2e3f7d2` |
| `results/reimplementation_training_witness/run/receipt.json` | `324cc3e466490a91b4ab7044b688992d4b325ddd18b4f97ab0ebb1744524d551` |
| `results/reimplementation_training_witness/run/training_losses.jsonl` | `4b9e0d9a1561b00c506f8f401bb0868335f488829d9895c5e725d67f42368123` |
| `results/reimplementation_training_witness_log/receipt.json` | `90a4709451eb33f9df5585e651102c776749c9ab400586db6462bb1705d6c7f8` |
| `results/reimplementation_training_witness_log/stdout_stderr.log` | `0bc1274fb6561310116a533e5690a3474ff8a80b1ccf78fb0076d12eb269456c` |
| `keyword_literature_update_20261010/sources/phygile.txt` | `e9bf566306ab41999033d86ec4528278bf19b8f4e7e2a01a36270bdff00c0e63` |
| `keyword_literature_update_20261010/sources/phygile.pdf` | `b66edb729feedc84b507c9e87a65e969cc23f8be0360f624696ea9b3759680a0` |

## Acceptance note

This substantive review is same-family and therefore provisional. Its positive findings are suitable for internal correction and gating, but they are not cross-family acceptance and do not convert interface/plumbing witnesses into experimental reproduction.
