# Experiment Audit Report — Round 3 Bounded Recheck

**Date:** 2026-10-17  
**Auditor:** GPT-5.6-Sol (same-family reviewer, read-only, provisional)  
**Reviewer metadata:** `reviewer_model=gpt-5.6-sol`, `review_independence=same-family`, `acceptance_status=provisional`  
**Overall verdict:** **WARN**  
**Blocking defects:** **none in the bounded source contract**  
**Claim ceiling:** implementation consistency, a genuine public pretrained text-interface check, and a reduced synthetic training-plumbing witness. This is not a full experiment reproduction.

The round-2 blocker is resolved **at source level**. `fetch_text_encoder.py` now queries the fixed commit `12040accade4e8a0f71eabdb258fecc2e7e948be` (`reimplementation/fetch_text_encoder.py:9-11`) and, after a completed download, writes the same serialized manifest to both `results/reimplementation_text/download.json` and `cache/text/distilbert-base-uncased/download.json` (`reimplementation/fetch_text_encoder.py:18,31-39`). This matches the trainer, which requires `encoder_path/download.json` and verifies every listed file hash (`reimplementation/train.py:117-121`). The revised source SHA-256 is `8fbcfff77780bd14af1f29937eef8a514b4edeaead0ece680e6c029eda33b162`.

This correction was not exercised by a fresh network download. The local task cache did not contain a post-fix cache-local `download.json` during review; the recorded online attempt remains the historical failure in `results/reimplementation_text/pipeline.json:2-9`, followed by the separately labeled local-transfer recovery in `results/reimplementation_text/pipeline_recovery.json:2-5`. The fix is statically verified; a new end-to-end fetch-to-train execution remains unwitnessed.

The current core result was recollected after the source change. It records twelve passing numerical checks, 146,169,702 parameters, no trained model, no text encoder, no dataset, and no physics simulation (`results/reimplementation_core/verification.json:2-24`), and embeds the new fetcher hash at `results/reimplementation_core/verification.json:25-34`. Exit code remains zero and the log contains twelve PASS lines (`results/reimplementation_core/exit_code.txt:1`; `results/reimplementation_core/stdout_stderr.log:1-12`). This recollection executes the numerical core; recording the fetcher hash does not execute the fetcher.

## Historical disposition

Round-1 and round-2 materials remain preserved under `results/reimplementation_core/round1/` and `results/reimplementation_core/round2/`. The round-2 audit files are unchanged (`EXPERIMENT_AUDIT.md` SHA-256 `bfca9e0eb9575f9e1b9240068e9fa007a7fc46e1f974a30f35db197fb6210668`; JSON SHA-256 `96b7e16987132d710e990dad3ad00e70b51a26cec021603032c00fc5d57af54e`). Two earlier findings are now historical rather than current blockers:

1. The fetcher/trainer manifest-path mismatch is corrected in source at `reimplementation/fetch_text_encoder.py:35-39`.
2. The moving-HEAD concern is corrected by the explicit revision at `reimplementation/fetch_text_encoder.py:10`.

The earlier bounded-artifact warning is also narrowed: this review independently hashed the seven transferred DistilBERT files, their `download.json`, all three synthetic `.npy` inputs, and `run/model_000002.pt`. Their bytes match the recorded manifests and verification hashes. This does not retroactively turn the old training witness into an execution of the revised fetcher; its receipt and verification correctly retain the previous fetcher SHA at `results/reimplementation_training_witness/run/receipt.json:20-29` and `results/reimplementation_training_witness/verification.json:18-27`.

## A. Ground-truth provenance: PASS

No motion-quality ground truth is used or claimed.

- The core uses seeded numerical tensors and reports `trained_model=false`, `dataset_used=false`, and `physics_simulation_tested=false` (`reimplementation/test_core.py:1-18`; `results/reimplementation_core/verification.json:17-24`).
- The text run uses a real public pretrained DistilBERT only for interface behavior. Its result says `interface_check_only`, `paper_exact_checkpoint_confirmed=false`, `motion_generator_trained=false`, and reports no motion dataset, motion-quality evaluation, or physics loop (`results/reimplementation_text/verification.json:2-6,25-35`).
- The manifest labels its arrays `synthetic_unit_test`, says their rotations are not valid robot rotations, and says they are not motion data (`results/reimplementation_training_witness/manifest.json:2-6`). Its captions are artificial interface strings (`results/reimplementation_training_witness/manifest.json:13-16,25-28,39-42`).
- The training result records two CPU updates and denies real motion data, a full-size trained generator, motion-quality evaluation, and physics (`results/reimplementation_training_witness/verification.json:2-15`).

The synthetic tensors are valid unit/plumbing inputs. They are not empirical motion ground truth.

## B. Score normalization: PASS

There is no evaluation score normalized by prediction statistics or rescaled toward a favorable value. Feature normalization is fit from training arrays only (`reimplementation/train.py:106-114`; `reimplementation/features.py:29-46`). The two logged values are raw training components for two updates (`results/reimplementation_training_witness/run/training_losses.jsonl:1-2`); their small decrease has no scientific interpretation.

## C. Result existence, numbers, and source hashes: PASS with execution-age qualification

All named files exist and parse. Direct hashing confirms:

- The recollected core JSON/log/exit-code hashes are `adf9abc90ea7458a5b125d0b57c9505beca803d80c34b77028ab9b5ade1f0036`, `9fd03de7273d24636e6408a92365e55d730b33e23af204434ab02d4835fbd58f`, and `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`. Its nine embedded source hashes match current files, including the new fetch SHA (`results/reimplementation_core/verification.json:25-34`).
- Task-root `distilbert_download/download.json` is byte-identical to `results/reimplementation_text/download.json` (both SHA-256 `76c3ed2cf6deef78acdb8c86b27a8133e259a3f752535472d31c548dbd73c136`). All seven local files have the exact sizes and hashes recorded at `results/reimplementation_text/download.json:7-56`; `model.safetensors` also matches the recorded publisher LFS digest (`results/reimplementation_text/download.json:37-41`).
- The three synthetic arrays independently hash to the values in the manifest (`results/reimplementation_training_witness/manifest.json:8-37`). The checkpoint independently hashes to `23ab3b2a832d572977a7c4c822244fcf66f675081d71bf2671b948d107c8859e`, matching `results/reimplementation_training_witness/verification.json:17`.
- The run receipt's manifest/config hashes match the actual files (`results/reimplementation_training_witness/run/receipt.json:2-10`). It records two sequences, nine frames, CPU, and two updates (`results/reimplementation_training_witness/run/receipt.json:9-19,31-34`), consistent with two loss rows.

The qualification is temporal: the text and training witnesses predate the fetcher change and retain the prior fetcher source hash. The current core recollection binds the revised source but does not execute text download or training.

## D. Dead code and exercised paths: WARN

The core covers individual expert/gate gradients, mask invariance, CFG, per-step boundary reinjection, 262D normalization, ASFO arithmetic/zero semantics, multi-token chunking, actual-denoiser null conditioning, and a full default-architecture CPU forward (`reimplementation/test_core.py:40-134`; `results/reimplementation_core/stdout_stderr.log:1-12`). The public text adapter was genuinely loaded in the historical interface run (`reimplementation/test_text_encoder.py:15-34`). The reduced training entrypoint genuinely performed two updates and reloaded its checkpoint (`reimplementation/test_training_entrypoint.py:43-50`).

Remaining gaps are material:

- The revised fetcher's completed dual-manifest path has no post-fix execution receipt.
- The witness disables ASFO (`results/reimplementation_training_witness/config.json:33-38`), so ASFO-enabled data weighting has no end-to-end witness.
- Only the synthetic `unit_test` route is executed. There is no real `research_training` run, full-size optimizer step, checkpoint resume test, evaluator, generation-quality evaluation, GMT-derived boundary, simulator loop, or PPO refinement.
- The overlap test exercises declared `source_group` collision (`reimplementation/test_training_entrypoint.py:27-34`). Exact-file-hash collision is implemented in `reimplementation/train.py:39-64` but lacks a distinct negative test.

## E. Scope and claim language: PASS with required ceiling

The README denies trained motion-generation weights and full-PhyGile reproduction, labels core tensors as numerical checks, records paper-unspecified assumptions, lists missing data/evaluation/simulation/PPO work, distinguishes the public text-interface run from motion training, and labels proposed training settings as independent (`reimplementation/README.md:1-17,27-41,73-75`).

The supported claim is bounded implementation/interface consistency. Trained motion generation, convergence, language-motion quality, physical executability, simulation performance, FID/retrieval metrics, human evaluation, and full-paper reproduction remain unsupported.

## F. Evaluation classification: PASS

| Artifact | Classification | Exact claim ceiling |
|---|---|---|
| Recollected core | `synthetic_proxy / deterministic_unit_check` | Formula, mask, gradient, shape, and finite-forward consistency |
| Public DistilBERT run | `interface_check_only / public_pretrained_text_encoder` | Frozen, padded, deterministic token interface and finite untrained-denoiser integration |
| Two-update witness | `synthetic_proxy / training_plumbing_unit_test` | Manifest guards, train-only normalization, two reduced-model CPU updates, checkpoint serialization/reload |

The public pretrained text-interface check is genuine, but it is not a trained motion generator. The optimizer steps are genuine execution, but they use two artificial sequences totaling nine frames, a reduced 2-layer/width-16/3-expert model, eight diffusion steps, ASFO disabled, and no motion metric (`results/reimplementation_training_witness/config.json:2-38`; `results/reimplementation_training_witness/run/receipt.json:7-18,31-34`). They are not a scientific motion experiment.

## Equations 7–14 and architecture

The unchanged round-2 paper comparison remains valid because the core source and paper hashes are unchanged.

- **Eq. 7: PASS as a primitive.** `reimplementation/diffusion.py:20-36` implements x0 regression corresponding to `phygile.txt:523-536`. The author beta schedule is unpublished; the example schedule is proposed.
- **Eqs. 8–9: PASS with assumptions.** Parameter mixing/masks are at `reimplementation/model.py:38-55`, placement/head averaging at `reimplementation/model.py:67-75`, and paper definitions at `phygile.txt:537-572`. Multi-token chunk equivalence is checked at `reimplementation/test_core.py:105-112`.
- **Eqs. 10–11: PASS for literal arithmetic.** Cap validation is at `reimplementation/features.py:48-58`; zero semantics are tested at `reimplementation/test_core.py:92-103`. Actual oversampling/mirroring is absent.
- **Eq. 12: WARN, explicit reconstruction assumption.** The fixed-mask sampler (`reimplementation/diffusion.py:47-70`) and reinjection check (`reimplementation/test_core.py:68-82`) do not implement the paper's GMT physical prefix, MPJPE selection, simulator loop, or PPO refinement (`phygile.txt:603-631`).
- **Eqs. 13–14: PASS algebraically.** `reimplementation/diffusion.py:38-45` and `reimplementation/test_core.py:53-61,114-125` cover standard/negative CFG against `phygile.txt:1167-1187`.
- **Architecture: PASS for major published dimensions; provisional reconstruction overall.** `phygile.txt:1089-1120` matches `reimplementation/model.py:8-20,29-36,57-85`. The default 146,169,702-parameter model completes a finite CPU forward. Paper-unspecified positional encoding, AdaLN/residual ordering, bias, attention reduction/dropout, and sampler choices remain assumptions (`reimplementation/README.md:9-13`).

## Split-leakage assessment

`load_manifest` verifies referenced hashes and gathers `source_group` and exact file hashes per split, then rejects overlap (`reimplementation/train.py:39-64`). Normalization is fit after selecting training entries (`reimplementation/train.py:106-114`). The witness checks source-group rejection and train-only normalization (`reimplementation/test_training_entrypoint.py:27-34,51-57`).

This cannot detect differently encoded, cropped, mirrored, or near-duplicate derivatives unless the manifest assigns a shared `source_group`. Claims must say **declared source-group and exact-byte overlap are rejected**, not that all semantic leakage is eliminated.

## Author-published versus proposed training choices

| Choice | Status | Evidence |
|---|---|---|
| 262D representation; 8 layers; width 512; FFN 1024; 4 heads; dropout 0.1; 12 experts; gate shape; mask gamma/beta | Author-published | `phygile.txt:1089-1120`; `reimplementation/model.py:8-20` |
| x0 objective; condition dropout 0.1; CFG scale 2.5; balance coefficient 0.01 | Author-published | `phygile.txt:523-536,1117-1120,1167-1187` |
| AdamW, LR 1e-4, betas 0.9/0.999, weight decay 0, gradient clip 1 | Independent proposal | `reimplementation/train_config.example.json:13-16` |
| Linear beta 1e-4 to 0.02 over 1000 steps | Independent proposal | `reimplementation/train_config.example.json:12` |
| Batch 4, 100,000 updates, save every 1,000, max 300 frames | Independent proposal | `reimplementation/train_config.example.json:7-10` |
| `rho_max=4`, `keep_at_least_one`, mirroring disabled | Independent operational proposal | `reimplementation/train_config.example.json:19` |
| Reduced model, 8 diffusion steps, 2 CPU updates, ASFO disabled | Synthetic witness only | `results/reimplementation_training_witness/config.json:2-38` |

## Claim impact

- **Supported:** current exact sources passed twelve deterministic numerical checks.
- **Supported with scope qualifier:** a recorded public pretrained DistilBERT revision passed a frozen text-interface test; all seven transferred files independently match its manifest.
- **Unsupported:** that the public checkpoint is the authors' exact checkpoint.
- **Supported as plumbing only:** two reduced-model CPU updates completed on artificial arrays; input and checkpoint hashes independently match their records.
- **Statically supported, not executed post-fix:** revised fetcher and trainer agree on the cache-local provenance path and the fetcher is commit-pinned.
- **Unsupported:** a full-size motion generator was trained, the two losses demonstrate learning/convergence, or any motion quality/physics result exists.
- **Supported with limit:** declared source-group and exact-byte overlap are rejected in source; only source-group has a direct negative witness.
- **Unsupported:** all semantic leakage is eliminated or proposed optimizer/schedule/budget/ASFO values reproduce unpublished author settings.

## Required corrections before stronger claims

1. Add a bounded post-fix fetch-layout-to-trainer provenance test or receipt; local fixture bytes are sufficient.
2. Validate manifest status, repository, revision, complete required file set, and preferably manifest hash in `train.py`, in addition to per-file hashes.
3. Add ASFO-enabled end-to-end tests and a distinct exact-file-hash cross-split rejection test.
4. Keep text and two-update artifacts classified as interface/plumbing checks. Scientific claims require real paired data, full training, held-out evaluation, and the missing physical pipeline.

## Audited input SHA-256 hashes

| Input | Bytes | SHA-256 |
|---|---:|---|
| `reimplementation/README.md` | 8292 | `d9ee7de9ed5e1cc6f8880c78ff44f4fdd87a5745d2e522e8464990f8a3f09fa3` |
| `reimplementation/model.py` | 6105 | `76eb8baa876950a34d4b23b6fa92627515bf38072c9cadf35fbf8789a13e70ab` |
| `reimplementation/features.py` | 2932 | `dee9e489f91b13217faa95a71bc5bc2f9122a1cc404639a3df9838edb6c7f9ac` |
| `reimplementation/diffusion.py` | 4642 | `930a2cdb1967119dd51813c6838f58c062fc7d46447239b20cc15f4b638efc4c` |
| `reimplementation/text.py` | 1100 | `22a04171f7e1218ff9b9ed768d6739ebed24c4072c683b78d60fde595b7fcf70` |
| `reimplementation/test_core.py` | 9050 | `c12732d27dac9bdee04f6de4353972e94c271a6e08bc7b58608757d68dea402b` |
| `reimplementation/fetch_text_encoder.py` | 2286 | `8fbcfff77780bd14af1f29937eef8a514b4edeaead0ece680e6c029eda33b162` |
| `reimplementation/test_text_encoder.py` | 2598 | `80ce12e90c2abafb63cadc5d687ad3a105141a6f6ee500a6ee54f9d1512fd004` |
| `reimplementation/train.py` | 10533 | `47031d106376a1ddc4d2d3a46d1842101b4f028742aa79f77490bce86cc61baf` |
| `reimplementation/test_training_entrypoint.py` | 4230 | `bd0dd2e0b5ea127c183825ddc3ba977b619f16b3d9ec3fae91b9b33afb5c55e4` |
| `reimplementation/train_config.example.json` | 818 | `74dbc8c4d91e928d1e3d116ffe96bac356c779f3242076943cda565bf3df3d5f` |
| `results/reimplementation_core/verification.json` | 2177 | `adf9abc90ea7458a5b125d0b57c9505beca803d80c34b77028ab9b5ade1f0036` |
| `results/reimplementation_core/stdout_stderr.log` | 3176 | `9fd03de7273d24636e6408a92365e55d730b33e23af204434ab02d4835fbd58f` |
| `results/reimplementation_core/exit_code.txt` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `results/reimplementation_text/download.json` | 2844 | `76c3ed2cf6deef78acdb8c86b27a8133e259a3f752535472d31c548dbd73c136` |
| `results/reimplementation_text/verification.json` | 1555 | `7a5391c9737833e2576902191227b783fe22d9222609a60124d348901d416246` |
| `results/reimplementation_text/pipeline.json` | 172 | `c0e187d62630d7bdfd9cee9f645c502b3637fcd0a545038b10756e77e71cf3d4` |
| `results/reimplementation_text/pipeline_recovery.json` | 270 | `6c017f0e13c66c8a546b4541bae1290a21ab3427cd3e654b9ea120165f7f40c9` |
| `results/reimplementation_text/text_interface_local_transfer/receipt.json` | 10706 | `b5c76a540db9ecc116a06133c1af39a8ee647ea1c64c6fb8545eb0e96834ed15` |
| `results/reimplementation_text/text_interface_local_transfer/stdout_stderr.log` | 1555 | `7a5391c9737833e2576902191227b783fe22d9222609a60124d348901d416246` |
| task-root `distilbert_download/download.json` | 2844 | `76c3ed2cf6deef78acdb8c86b27a8133e259a3f752535472d31c548dbd73c136` |
| task-root `distilbert_download/config.json` | 483 | `69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83` |
| task-root `distilbert_download/tokenizer_config.json` | 48 | `a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10` |
| task-root `distilbert_download/tokenizer.json` | 466062 | `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98` |
| task-root `distilbert_download/vocab.txt` | 231508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |
| task-root `distilbert_download/model.safetensors` | 267954768 | `5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063` |
| task-root `distilbert_download/README.md` | 8577 | `b925a22d8efd7a29608a0e10d3f2fb33d5d80df7de4ac6e34093853cd8e0274e` |
| task-root `distilbert_download/LICENSE` | 11356 | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |
| `results/reimplementation_training_witness/verification.json` | 1393 | `9462d4d0a0c84075f75e9b5aa04d88ac4b8634225132b7fad2b1c20d79246fbc` |
| `results/reimplementation_training_witness/manifest.json` | 1561 | `e13350f4bbd5a3e82f3259fffad4e9a261a6d5ee012029d3674f4ac623aaff9e` |
| `results/reimplementation_training_witness/config.json` | 745 | `391a22ea7710281e298fc03ee66d26e9f98d098df458f6e1942db752a2e3f7d2` |
| `results/reimplementation_training_witness/run/receipt.json` | 1533 | `324cc3e466490a91b4ab7044b688992d4b325ddd18b4f97ab0ebb1744524d551` |
| `results/reimplementation_training_witness/run/training_losses.jsonl` | 219 | `4b9e0d9a1561b00c506f8f401bb0868335f488829d9895c5e725d67f42368123` |
| `results/reimplementation_training_witness/synthetic_unit_0.npy` | 4320 | `ce6b7dacee18521f0a6f14029f6b8f9ace7197835be3fd4190d3ce9258e9e276` |
| `results/reimplementation_training_witness/synthetic_unit_1.npy` | 5368 | `f937bcfc2d1b22824bc9143b0a497f94c2c5c58db7295229309659595ba499b9` |
| `results/reimplementation_training_witness/synthetic_unit_2.npy` | 6416 | `1106366418dc2888488cde5c5566ffde048888306fb1993f525d07cb4273b5d3` |
| `results/reimplementation_training_witness/run/model_000002.pt` | 823378 | `23ab3b2a832d572977a7c4c822244fcf66f675081d71bf2671b948d107c8859e` |
| `results/reimplementation_training_witness_log/receipt.json` | 10713 | `90a4709451eb33f9df5585e651102c776749c9ab400586db6462bb1705d6c7f8` |
| `results/reimplementation_training_witness_log/stdout_stderr.log` | 2926 | `0bc1274fb6561310116a533e5690a3474ff8a80b1ccf78fb0076d12eb269456c` |
| `keyword_literature_update_20261010/sources/phygile.txt` | 63706 | `e9bf566306ab41999033d86ec4528278bf19b8f4e7e2a01a36270bdff00c0e63` |
| `keyword_literature_update_20261010/sources/phygile.pdf` | 4981874 | `b66edb729feedc84b507c9e87a65e969cc23f8be0360f624696ea9b3759680a0` |

## Acceptance note

This review is same-family and provisional. The WARN reflects the absence of a scientific motion experiment and of post-fix fetch-to-train execution evidence, not a failure of the bounded numerical checks. No full experiment reproduction is claimed.
