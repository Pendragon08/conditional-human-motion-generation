**Independent integrity verdict: PASS with two scoped caveats.** C's files, receipts, counts, classifications, hashes, and E's documented invariants reproduce. The scientific result remains **zero joint-pass gain**, supporting only a stop decision for this bounded parameterization.

- All four batches contain the same 72 ordered cases: 24 sources × 3 offsets, seed 11. Split provenance is 57 train-derived and 15 test-derived inputs; this is internal validation, not untouched testing.
- Recomputed counts:
  - Baseline: condition 5, physical 0, tracking 68, joint 0.
  - Cartesian: 7, 0, 68, 0.
  - Two-link IK: 6, 0, 68, 0.
- Every reported metric has finite denominator **72/72** and missing/nonfinite **0/72**. Analysis means match exactly except one IK rounding difference of `2.7755575615628914e-17`.
- Source-cluster bootstrap over 24 sources and 5,000 resamples gives joint gain `0.0 pp`, CI `[0.0, 0.0]`, for both arms. The degenerate interval does not establish population equivalence.
- Component diagnosis reproduces support/slip counts exactly:
  - Baseline: `4650/8927 = 0.5208916769351406`
  - Cartesian: `4740/8990 = 0.5272525027808677`
  - IK: `4715/8972 = 0.52552385198395`

  Recomputed pooled COM/rotational/material terms are baseline `0.1712094153545382 / 0.15161218024315692 / 0.1917717596801539`, Cartesian `0.1720241956841677 / 0.15094279243204684 / 0.19341074563851002`, and IK `0.16909651869905484 / 0.15228500184682328 / 0.1893225417266508`. Maximum artifact difference is `2.7989e-11 m/s`. These are descriptive components, not replacement scores or thresholds.
- Mean per-case slip and pooled slip legitimately differ because their denominators differ: 72 equally weighted cases versus all support observations.
- Prefix frames 0–60, frame 119, non-leg coordinates, goals, and proposed/original headings are unchanged. IK segment error is `8.36185e-08 m`; Cartesian's `0.0579819 m` segment error is expected from its positional ablation.
- All four initial-state archives are byte-identical, SHA-256 `b25404dd…`; bootstrap tensors equal `first_action`, reset/cache checks are zero. Hidden PhysX-state identity remains outside the evidence.
- Float32 reference invariants pass. Float64 logged norms should not be required to equal norms recomputed from saved float32 vectors: maximum difference is `5.68835e-08`. Independent solver recreation also showed only low-order drift (`1.30771e-10` accepted shift; `2.98023e-08` bone error).
- Observation traces satisfy ordering and zero-first-velocity contracts. NumPy recomputation differs by up to `9.53674e-07 m/s` for the three adjudicated batches and `1.90735e-06` for the source batch; the retained CUDA same-arithmetic audit matches all 622,080 scalars exactly.
- The sole row-level numeric mismatch is descriptive `surface_slip_proxy_mean_m_s`, maximum `6.54324e-10 m/s`; identities, gates, classifications, and reported analysis metrics are unchanged. Current strict metric-equivalence tolerance is therefore tighter than reproducible cross-runtime reduction behavior.
- Real costs: source construction `49.1905804593 s`, cached baseline `50.4387053065 s`, Cartesian `49.7906149086 s`, IK `49.8940196168 s`; candidate charged total `99.6846345253 s`. The baseline ledger snapshot hash reconstructs exactly, and its row is unchanged in the current append-only ledger. Historical full-ledger equality was undocumented.
- Original witness FAILs remain correctly retained: 117 scalar-only checks and one historical-ledger comparison. The adjudicated witness passes with zero failures.
- Filesystem evidence records run SSH exit 0, chunk `5faa0c`, sentinel `SUPPORT_LEG_IK_ADJUDICATED_COMPLETE`, and analyzer exit 0, chunk `028dc7`. There is no synthesized autonomous server `PARENT_EXIT` receipt; `pipeline_finally_receipt.json` does not claim one.

The automated `FAIL_OR_DIVERGENCE` label is retained because deliberately exact descriptive-float and independent-solver checks diverged. Those mismatches do not alter the documented classifications or permitted claims.

