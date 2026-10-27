# Experiment Audit Report

**Date:** 2026-10-17  
**Auditor:** GPT-5.6-Sol (fresh same-family reviewer, read-only, provisional)  
**Reviewer metadata:** `reviewer_model=gpt-5.6-sol`, `review_independence=same-family`, `acceptance_status=provisional`  
**Scope:** only the independently implemented, untrained PhyGile generation core, its deterministic verification artifacts, and the supplied PhyGile paper text/PDF. No source was edited or executed; no model was trained; no GPU, dataset, text weights, simulator, or motion-quality evaluator was used.

## Overall Verdict: WARN

The recorded run is a real, internally consistent set of deterministic numerical/unit checks over the exact hashed Python sources. It is not a motion-generation experiment and supplies no evidence about motion quality, language alignment, physical executability, or reproduction of the full PhyGile system. The core is broadly faithful to the arithmetic of Eqs. 7–11 and 13–14, but the README claim that it implements Eqs. 7–14 item by item is too broad: Eq. 12 is instantiated by a disclosed but paper-unspecified fixed-mask inpainting algorithm, and the paper's physical-prefix/GMT validation semantics are absent. ASFO implements only multiplier arithmetic, not the sampling and mirroring procedure. Several architectural choices remain under-specified by the paper, and one gradient-check label overstates what its assertions prove.

## Audited roots

- Implementation: `work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile/reimplementation/`
- Results: `work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile/results/reimplementation_core/`
- Paper text/PDF: `keyword_literature_update_20261010/sources/phygile.{txt,pdf}`

The PDF was directly opened and text-extracted for PDF pages 5 and 9. Line citations below use the supplied NUL-containing `phygile.txt`, read in binary-tolerant mode, because it provides stable line numbers.

## A. Ground-truth provenance: PASS

There is no dataset ground truth, reference trajectory, official benchmark, or claimed motion-quality score in this run. The checks use seeded random numerical tensors and analytic/self-consistency expectations: `reimplementation/test_core.py:11-28`, `reimplementation/test_core.py:30-46`, `reimplementation/test_core.py:48-79`. The test module identifies itself as non-performance testing at `reimplementation/test_core.py:1`; the README explicitly says the tensors only check formulas, shapes, masks, and gradients and are not generation-quality or physical-executability experiments at `reimplementation/README.md:7`, and repeats that random tokens do not replace a text encoder or training data at `reimplementation/README.md:21`. The result also records `trained_model=false`, `text_encoder_loaded=false`, `dataset_used=false`, and `physics_simulation_tested=false` at `results/reimplementation_core/verification.json:17-20`.

The Eq. 8 expected tensor is derived from the same randomly initialized module parameters (`reimplementation/test_core.py:19-27`). That is valid for a formula unit test because it is explicitly labeled as such; it is not independent ground truth and must never be presented as model performance.

## B. Score normalization: PASS

No performance metric is computed, rescaled, or normalized against the model's own outputs. `BlockNormalizer.fit` computes feature preprocessing statistics over caller-supplied training sequences (`reimplementation/features.py:29-38`), and the associated test checks preservation/round-trip behavior (`reimplementation/test_core.py:63-69`); neither is a reported evaluation score. The maximum in the spatial mask (`reimplementation/model.py:44-46`) implements paper Eq. 9 (`keyword_literature_update_20261010/sources/phygile.txt:554-570`) and is not score normalization. There are no suspicious near-one accuracy/quality claims.

## C. Result existence, values, and source hashes: PASS

All three named result artifacts exist. `results/reimplementation_core/exit_code.txt:1` is `0`; the log contains eight `PASS` records at `results/reimplementation_core/stdout_stderr.log:1-8`; and the same eight labels, status, flags, parameter count, and elapsed value are reproduced in `results/reimplementation_core/verification.json:2-20` and `results/reimplementation_core/stdout_stderr.log:9-36`. An independent static parameter count from the declared modules is 146,169,702, matching `results/reimplementation_core/verification.json:16`.

The five Python hashes embedded at `results/reimplementation_core/verification.json:21-26` exactly match independent SHA-256 hashes of the current files. The JSON and log are mutually consistent, but the elapsed time is only internally reported; these bounded artifacts contain no external run attestation, timestamp, command manifest, hardware manifest, or hash of the paper/README.

## D. Dead code and coverage: WARN

There are no evaluation-metric functions, so no phantom metric output was found. Most numerical primitives are called by `test_core.py`. Important declared paths are nevertheless not exercised:

- The real frozen DistilBERT adapter (`reimplementation/text.py:5-22`) is never imported or called; the result explicitly records `text_encoder_loaded=false` at `results/reimplementation_core/verification.json:18`.
- The default 8-layer/512-wide/12-expert model is only constructed and counted (`reimplementation/test_core.py:76-79`), not forwarded. Forward/backward checks use a much smaller configuration (`reimplementation/test_core.py:16`, `reimplementation/test_core.py:30`).
- The label “finite nonzero gradients into every expert pool and gate” overstates the assertions. `reimplementation/test_core.py:44` checks finiteness only for parameters whose gradients are already non-`None`; `reimplementation/test_core.py:45` checks only aggregate `w1` and the final gate layer's weight in each block. It does not establish nonzero gradients for every individual expert, `w2`, expert biases, all gate layers, or every model parameter.
- `reimplementation/README.md:9` states a mirror-probability behavior, but no mirroring function or probability computation exists in the bounded implementation files. `asfo_multipliers` only returns integer-like sampling multipliers and metadata (`reimplementation/features.py:47-55`).

## E. Scope assessment: WARN

The README is commendably explicit that this is untrained research code and not a full reproduction (`reimplementation/README.md:3`), that the checks are synthetic numerical checks only (`reimplementation/README.md:7`), and that the dataset, 262D codec/mapping, generator training/evaluator, GMT, simulation loop, and PPO adaptation remain missing (`reimplementation/README.md:11-13`). Those limitations match the result flags (`results/reimplementation_core/verification.json:17-20`).

However, `reimplementation/README.md:5` says the implementation corresponds item by item to Eqs. 7–14. That wording exceeds the evidence:

- Paper Eq. 12 conditions generation on a physically executable prefix produced/validated by GMT and is followed by simulation filtering and progressive refinement (`keyword_literature_update_20261010/sources/phygile.txt:603-631`). The code instead chooses fixed-value inpainting with per-step re-noising/reinjection (`reimplementation/diffusion.py:47-70`). The assumption is disclosed at `reimplementation/diffusion.py:50-54` and `reimplementation/README.md:9`, but it is not an implementation of the paper's omitted conditioning/sampler details or physics loop.
- Paper ASFO includes sampling plus rarity-dependent left/right mirroring and tag swapping (`keyword_literature_update_20261010/sources/phygile.txt:575-602`). The code implements only Eqs. 10–11 multiplier arithmetic (`reimplementation/features.py:47-55`).
- The paper specifies contact extraction and canonical-heading preprocessing (`keyword_literature_update_20261010/sources/phygile.txt:1154-1166`); these are not implemented. This is consistent with the README's requirement that callers provide semantically mapped/prepared blocks (`reimplementation/README.md:11-13`), but it limits the claim to an internal 262D tensor layout rather than a paper-complete feature pipeline.

## F. Evaluation classification: PASS (classification only)

**Machine-compatible type:** `synthetic_proxy`  
**Precise subtype:** deterministic formula/shape/mask/gradient unit checks  
**Claim ceiling:** implementation-level consistency only

This is not a `real_gt`, `simulation_only`, or `human_eval` study. It is also not a useful proxy for motion quality: no trained model, dataset, text encoder, physical simulation, or evaluator is present (`results/reimplementation_core/verification.json:17-20`). The `synthetic_proxy` label is used only because the audit schema lacks a `unit_test_only` type; it must not be paraphrased as a synthetic motion-quality evaluation.

## Equation-by-equation and architecture audit

### Eq. 7 — x0 diffusion objective: PASS, with an explicit schedule gap

The paper defines x0 regression at `keyword_literature_update_20261010/sources/phygile.txt:523-536`. Forward noising and clean-motion MSE are implemented at `reimplementation/diffusion.py:20-36`. The beta schedule is not specified by the paper and is therefore required from the caller (`reimplementation/diffusion.py:6-15`), which is correctly disclosed at `reimplementation/README.md:9`. This supports an x0-loss primitive, not reproduction of the author's training setup.

### Eq. 8 — token-level parameter mixing: PASS

The paper mixes expert parameters per token (`keyword_literature_update_20261010/sources/phygile.txt:537-553`). The code computes token gates, mixes both FFN weight matrices and biases before the GELU, and applies the resulting token-specific network (`reimplementation/model.py:38-55`). The one-token formula check is at `reimplementation/test_core.py:15-28`. Multi-token/chunk equivalence and the default 12-expert forward path are not tested.

### Eq. 9 — attention-derived spatial mask: PASS, with under-specified choices

The sigmoid/max mask and token-summed update match the paper formula (`keyword_literature_update_20261010/sources/phygile.txt:554-572`; `reimplementation/model.py:44-55`). The implementation uses head-averaged PyTorch cross-attention weights (`reimplementation/model.py:72-74`) and applies the TP-MoE after the FFN residual, as stated by the paper. Whether the mask should use per-head, pre-dropout, or post-dropout attention is not specified in the paper and is not documented in the README.

### Eqs. 10–11 — ASFO multipliers: WARN

Frequency, median target, round-half-up multiplier, and per-sample maximum follow the printed equations (`keyword_literature_update_20261010/sources/phygile.txt:575-591`; `reimplementation/features.py:47-55`). Two operational problems remain:

1. `reimplementation/README.md:9` explicitly allows a multiplier of zero. If the return value is used as a repetition count, frequent-only samples can be removed, which conflicts with the prose that mirroring is for rare samples while originals are otherwise kept (`keyword_literature_update_20261010/sources/phygile.txt:592-602`). The current test never reaches the zero case (`reimplementation/test_core.py:71-74`).
2. `rho_max` is checked only as `rho_max >= 1` (`reimplementation/features.py:49`), not as an integer. A non-integral cap can enter the metadata and is silently truncated when the per-sample NumPy integer array is created (`reimplementation/features.py:53-54`).

The code also does not implement the actual oversampling dataset/sampler or the paper's mirrored sample construction/tag swap.

### Eq. 12 — physics-prefix conditioning: WARN

The paper gives a conditional distribution and physical-validation loop, not a complete denoising algorithm (`keyword_literature_update_20261010/sources/phygile.txt:603-631`). The code implements one reasonable fixed-mask inpainting reconstruction with a single fixed known-noise tensor across timesteps (`reimplementation/diffusion.py:47-70`). This assumption is disclosed (`reimplementation/diffusion.py:50-54`; `reimplementation/README.md:9`), and the final prefix/suffix equality is tested (`reimplementation/test_core.py:58-61`). It does not reproduce GMT-derived prefixes, concatenation semantics, MPJPE rejection, simulator feedback, or PPO refinement. The test would also pass based on the final overwrite alone; it does not prove that per-step reinjection affected the denoising trajectory.

### Eqs. 13–14 — standard and negative-prompt CFG: PASS algebraically; integration assumption remains

The affine formulas match the paper (`keyword_literature_update_20261010/sources/phygile.txt:1167-1187`; `reimplementation/diffusion.py:38-45`) and the scalar TokenProbe verifies both cases (`reimplementation/test_core.py:48-56`). In the real model, training-time condition dropout zeros token values but retains the original validity mask (`reimplementation/diffusion.py:24-32`); projections, the summary pool, and gates contain learned biases (`reimplementation/model.py:32-34`, `reimplementation/model.py:80-83`, `reimplementation/model.py:94-98`). Thus the null branch is independent of token content but can retain sequence-mask/length and learned-bias effects. The paper does not resolve this choice, the README does not document it, and the TokenProbe test bypasses it.

### Architecture: WARN (substantial match, incomplete specification)

The declared full configuration matches the appendix on the major published dimensions: 8 decoder layers, width 512, FFN 1024, 4 heads, dropout 0.1, 12 two-layer experts, the 768→512→512→12 SiLU gate, γ=24, β=0.25, and load-balance coefficient 0.01 (`keyword_literature_update_20261010/sources/phygile.txt:1089-1120`; `reimplementation/model.py:8-20`, `reimplementation/model.py:29-36`, `reimplementation/model.py:57-85`; `reimplementation/diffusion.py:24-36`). Frozen 768D DistilBERT support exists at `reimplementation/text.py:5-22`, but was not loaded in the result.

The paper does not specify the frame/timestep positional encoding, exact AdaLN parameterization and placement, normalization affine settings, projection/output heads, residual/dropout placement, attention-weight reduction, padding/null-condition behavior, or complete sampler. Some choices are disclosed at `reimplementation/README.md:9`; others are only visible in `reimplementation/model.py:22-27`, `reimplementation/model.py:60-75`, and `reimplementation/model.py:80-105`. The artifact should therefore be described as a plausible paper-grounded reconstruction, not an exact architecture reproduction.

## Claim impact

- **“The exact hashed sources passed eight deterministic numerical checks.” — supported.** Result files, exit code, log/JSON agreement, parameter count, and source hashes are consistent.
- **“The implementation contains paper-grounded primitives for Eqs. 7–11 and 13–14.” — supported with the ASFO and under-specification qualifiers above.**
- **“The implementation corresponds item by item to Eqs. 7–14.” — needs correction.** Eq. 12 is an assumed inpainting realization and omits the physical-prefix/GMT loop; ASFO sampling/mirroring is absent.
- **“The full stated architecture constructs without pretrained weights.” — supported narrowly.** Construction and parameter count are evidenced; a full-size forward pass is not.
- **“Every expert pool and gate receives finite nonzero gradients.” — unsupported as written.** The assertions cover only aggregate `w1` and final gate-weight gradients per reduced block and ignore `None` gradients.
- **“The frozen DistilBERT integration works.” — unverified.** The adapter exists but was not loaded or exercised.
- **Any generation-quality, language-alignment, physics, benchmark, or full-reproduction claim — unsupported.** The README mostly avoids these claims correctly.

## Required corrections before stronger claims

1. Replace `reimplementation/README.md:5` with narrower wording: the code implements selected mathematical primitives corresponding to Eqs. 7–11 and 13–14, plus an explicitly assumed Eq. 12-inspired fixed-mask inpainting sampler.
2. Either implement the ASFO sampler and rarity-dependent mirroring/tag swap, or remove the mirror-probability statement at `reimplementation/README.md:9`. Require integer `rho_max`; clamp multipliers to at least one if they are repetition counts, or document and test intentional zero-count semantics.
3. Rename the gradient test/result label to the exact assertion coverage, or strengthen it to require non-`None`, finite, nonzero gradients for every intended expert tensor, each individual expert, and all gate layers.
4. Add tests that exercise multi-token/chunked TP-MoE, the actual `MotionDenoiser` null/negative conditioning path, and the behavior of the original validity mask under condition dropout. Document the chosen null-conditioning semantics.
5. Document the remaining architecture/sampler assumptions: head-averaged attention for Eq. 9, attention-dropout timing, projection/output biases, exact AdaLN/residual ordering, fixed known-noise reuse during inpainting, and final-only nature of the current boundary-preservation assertion.
6. Keep all result wording at the deterministic-unit-check level until trained weights, paper-compatible data/feature preprocessing, real text encodings, official/validated evaluation, and physics simulation are present.

## Input SHA-256 hashes

| Input | SHA-256 |
|---|---|
| `reimplementation/README.md` | `1146710223aab46dfe4bf453445e64d68ada0dff5f7a998135c81f2cdbd9f420` |
| `reimplementation/model.py` | `76eb8baa876950a34d4b23b6fa92627515bf38072c9cadf35fbf8789a13e70ab` |
| `reimplementation/features.py` | `4a6708949a4754bc38cedd2362b08372f45702875a62116074a6cfee2a79b7c1` |
| `reimplementation/diffusion.py` | `930a2cdb1967119dd51813c6838f58c062fc7d46447239b20cc15f4b638efc4c` |
| `reimplementation/text.py` | `22a04171f7e1218ff9b9ed768d6739ebed24c4072c683b78d60fde595b7fcf70` |
| `reimplementation/test_core.py` | `e0e152279e98371faafea3fe125b79900f2654fb68294f449154b90f67d34458` |
| `results/reimplementation_core/verification.json` | `5ed378287cb8f21df814e945dd741773d9c2c21594b0a2479adfaeaa3515654f` |
| `results/reimplementation_core/stdout_stderr.log` | `0ddd2513565d4384e7b1e864d44493cb43629ca9fbeb44e347fbd8296b194e3e` |
| `results/reimplementation_core/exit_code.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `keyword_literature_update_20261010/sources/phygile.txt` | `e9bf566306ab41999033d86ec4528278bf19b8f4e7e2a01a36270bdff00c0e63` |
| `keyword_literature_update_20261010/sources/phygile.pdf` | `b66edb729feedc84b507c9e87a65e969cc23f8be0360f624696ea9b3759680a0` |

## Acceptance note

This is a same-family review and therefore provisional. Its positive checks may support revision and internal gating, but they are not cross-family acceptance and do not elevate the unit checks into an experimental reproduction.
