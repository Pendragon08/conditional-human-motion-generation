# Fresh paired64 document audit

The documented Python snippet was extracted without modification and executed exactly once on 2026-10-27 at 02:31 UTC. Local invocation exit: **0**. Remote verifier exit: **0**. Status: `PAIRED64_INDEPENDENTLY_RECOMPUTED`; matched sources: **64**; local and remote stderr: empty. No execution-versus-document divergence was observed. The verifier's assertions passed; **18 additional local aggregate/provenance checks** passed. No new SDK/physics/model/training execution or signal was requested.

The remote author summary SHA-256 is `ec1b4c9ae8767351208f97d87a07135918450b262d50a018a5581c8cdba64924`. The local public summary uses CRLF, so its raw file SHA differs; normalizing line endings to LF, and separately serializing its parsed JSON with the author's format, both reproduce the remote SHA exactly. This is transport formatting, not a numerical discrepancy.

## Result and denominators

The fixed 64 motions give 5k **42 complete / 22 failed**, versus 10k **46 complete / 18 failed**. The partition is 41 both complete, 5 new-only complete, 1 old-only complete, and 17 both failed. Completion rises from 65.625% to 71.875% (+6.25 percentage points). These are nominal-duration outcomes under the recorded termination rules.

Paired terminal diagnostics use only the **41 common natural completions**. The separate all-new-completions statistics use **46**, and must not be interpreted as a paired comparison.

| Paired terminal diagnostic | 5k mean | 10k mean | Mean change (10k minus 5k) |
| --- | ---: | ---: | ---: |
| World root position error (m) | 0.161131 | 0.129850 | -0.031281 |
| Root orientation geodesic error (rad) | 0.068205 | 0.072169 | +0.003963 |
| Joint RMSE (rad) | 0.175965 | 0.165905 | -0.010060 |

The trends are mixed. Root/joint mean errors improve; orientation mean worsens. Root-position p95 increases from 0.537126 to 0.554367 m, and joint-RMSE p95 from 0.308721 to 0.315374 rad. Orientation p95 improves from 0.164638 to 0.146616 rad. A negative median paired orientation change does not reverse its positive mean change.

## Source and evidence review

- The invocation pins the comparison source/spec; the comparison pins the endpoint kernel and binds the actual 10k checkpoint/handoff. Reviewed local source/spec/kernel bytes match those pins.
- Selection uses the original private selection, reference SHA, identity, frame length and split; all 64 identities must match the prior baseline audit. Process/resource/outer exits and closed writers are checked.
- Strict log, journal and trace index are SHA-bound. Journal trials must agree with the log. Every selected motion has one trial and one unique trace. Binary and NPZ fields must be exactly equal, with finite endpoint inputs and the declared sample count.
- Failure takes precedence over simultaneous natural completion. External timeout/unknown end is rejected as incomplete evidence. Failed trials receive no extrapolated terminal error.
- Natural completion requires T-1 control transitions. Recorded pose at (T-1)/50 seconds is compared with independent reference T-1; the controller last consumed T-2. Joint names are reordered explicitly; quaternion geodesic error handles sign symmetry. Root position uses world coordinates after scene-origin subtraction, without per-frame yaw/root realignment or initial-offset correction.
- `--verify-output` compares the recomputed private paired bytes and public summary with the existing author output and returns before the write branch. Original run artifacts are not intentionally rewritten.

The 5k endpoint audit is reused under its fixed SHA, not recomputed from 5k raw traces in this pass. The 10k recomputation uses the pinned metric implementation, not a second independent algorithm. These are boundaries of this verification, not new validation failures.

The inspected SERVER_NOTES.md documents warm tracker `b7a3bf0c` but had no comparison-specific ledger entry at audit time. This is a documentation bookkeeping gap; no ledger/source/spec/output was edited here.

This fixed64 diagnostic cannot accept the full library, semantic quality, a paper score, or contact-conditioned slip. The other 8214 original sources remain unattempted here. Teacher failure does not prove intrinsic motion infeasibility. No private source identities or arrays were exported. Proof and aggregate results are in this directory only.
