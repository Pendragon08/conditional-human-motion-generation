# V19 independent exact-document witness

Verdict: **PASS for isolated rigid-body velocity reference only**. No articulated humanoid slip calibration or innovation efficacy is claimed.

Read `docs/velocity_reference_fixture_v19_20261012.md`; executed its SSH command exactly once. The tool finished directly in 3.51 seconds, returned exit 0 and no live session ID. Terminal sentinel: `VELOCITY_REFERENCE_DOCUMENT_EXECUTION_COMPLETE PASS CENTER_OF_MASS`. No retry, source edit or extra simulation was performed.

Remote wrapper state is `COMPLETE_DIAGNOSTIC` with calibration `PASS`. Conservative measurement charge is 2.070211956 seconds, exit 0. The fixture has 240 frames, 2 bodies and 13 state components; all values are finite. Independent offline recomputation confirms all preregistered tests:

- Zero COM offset control: maximum body-origin velocity residual 2.428889146e-7 m/s, below 0.003 m/s.
- Nonzero COM setter readback: [0.05999999865889549, 0, 0] m, within setter tolerance of the prescribed 0.06 m.
- Offset body, body-origin hypothesis: maximum residual 0.1800576271 m/s, above the 0.05 m/s rejection boundary.
- Offset body, COM hypothesis: maximum residual 0.00007033406615 m/s, below 0.003 m/s.
- Angular velocity remained exactly 3 rad/s; maximum error 0. Quaternion norm error 1.139997531e-7.

The exported linear velocity refers to the center of mass in this isolated GPU PhysX / CPU tensor fixture. The observed origin-versus-COM difference is consistent with omega times COM displacement (3 rad/s times 0.06 m = 0.18 m/s). This supports checking the humanoid foot velocity calculation's origin convention; it does not independently establish corrected contact-slip measurements for articulated bodies.

All six local source/config/document hashes match the remote versions manifest. Fixture script SHA256: bb6306e801b7225e8e0d68db60eff375963f42a59773a005531255d4939249e5. Synthetic array SHA256: 75b2fad6489f43b372659f7e7f5e794c6ed0ff97cef6ab1c58bf887400b374cd. Independent recomputation matches the summary's origin and COM residuals.

Local evidence includes first_attempt/summary.json, first_attempt/fixture_arrays.npz, versions.json, state.json, budget_ledger.json, first_attempt.log and witness_checks.json. Only synthetic isolated box arrays were copied; no human data, model files or Overleaf content was changed.