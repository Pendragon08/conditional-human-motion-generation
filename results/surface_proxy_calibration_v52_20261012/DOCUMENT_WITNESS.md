# V52 independent document witness

**Status: PASS**. The documented CPU SSH invocation was executed verbatim exactly once, exit 0. Sentinel: `SURFACE_PROXY_ANALYTICAL_CONTROLS PASS_ANALYTICAL_CONTROLS 24`. No retry, repair, installation, generator, simulator or GPU invocation.

PASS only for 24 prescribed instantaneous rigid-body synthetic controls using the four same-family native foot collision boxes and frozen V21 COM-corrected surface proxy. Provisional same-family formula/interface calibration; not efficacy, independent contact ground truth, a dynamics-integrated rolling trajectory, human no-slip evidence, or evidence that the current high-angular-speed association is a false positive. Forces are prescribed, angles are parameter samples; both rocking branches use +3 rad/s at distinct signed pose samples. No prior success rates or thresholds may be changed from this witness.

## Evidence

- V52 source SHA256: `4ff5b27d3fa5addcc3eaf07cb6b95234305ff3ad98538a693b0f2591c83560f5`
- V50 parent source SHA256: `e31006c9ffdda4db16b1da2fadac47558c674a5f34f4c65687437f90c5a3e826`
- Summary SHA256: `333793689b5bdb7912d6966d3a3eb3c356024408500ef1dc512de34961dcce60`
- Synthetic NPZ SHA256: `e61e8ddca9f22ee391fe895f8bb84ed954f43321c3a4ebab4e7bf209e8a84e4f`
- Remote NumPy: 1.23.5. Read-only remote hashes match the summary and exact local audit bytes.
- Remote raw metadata SHA256 equals the existing local public metadata bytes; only the synthetic NPZ and summary were retrieved.
- All 120 arrays are finite float64 with 60 samples per control and exact dimensions. Four eight-vertex boxes have zero radii and measured native local COM.
- Independent audit used explicit y-axis rotation matrices and checked velocities at every tied lowest vertex. It did not invoke the producer or its V21 metric.
- Supported controls: 60 observations each; static and both rocking poses yield slip fraction 0; 0.2 m/s translation and rocking plus translation yield fraction 1 at frozen 0.05 m/s threshold. Airborne controls have zero support, null slip, and height 1 m.
- Support thresholds remain vertical force >10 N and lowest height <0.02 m. Support force 49.05 N is prescribed, not measured.
- V52 wrapper changes output/path bindings and parent-hash telemetry only. V50 failure and first output remain separate.
- Signed rocking angles do not imply a negative angular velocity: both branches prescribe +3 rad/s at separate pose samples. These samples do not certify time integration.

## Per-case independent audit

| Foot | Control | Support observations | Expected speed m/s | Slip fraction | Pass |
|---|---|---:|---:|---:|---|
| L_Ankle | static | 60 | 0 | 0.0 | yes |
| L_Ankle | flat_sliding | 60 | 0.2 | 1.0 | yes |
| L_Ankle | positive_rocking | 60 | 0 | 0.0 | yes |
| L_Ankle | negative_rocking | 60 | 0 | 0.0 | yes |
| L_Ankle | rocking_plus_sliding | 60 | 0.2 | 1.0 | yes |
| L_Ankle | airborne | 0 | 0 | None | yes |
| L_Toe | static | 60 | 0 | 0.0 | yes |
| L_Toe | flat_sliding | 60 | 0.2 | 1.0 | yes |
| L_Toe | positive_rocking | 60 | 0 | 0.0 | yes |
| L_Toe | negative_rocking | 60 | 0 | 0.0 | yes |
| L_Toe | rocking_plus_sliding | 60 | 0.2 | 1.0 | yes |
| L_Toe | airborne | 0 | 0 | None | yes |
| R_Ankle | static | 60 | 0 | 0.0 | yes |
| R_Ankle | flat_sliding | 60 | 0.2 | 1.0 | yes |
| R_Ankle | positive_rocking | 60 | 0 | 0.0 | yes |
| R_Ankle | negative_rocking | 60 | 0 | 0.0 | yes |
| R_Ankle | rocking_plus_sliding | 60 | 0.2 | 1.0 | yes |
| R_Ankle | airborne | 0 | 0 | None | yes |
| R_Toe | static | 60 | 0 | 0.0 | yes |
| R_Toe | flat_sliding | 60 | 0.2 | 1.0 | yes |
| R_Toe | positive_rocking | 60 | 0 | 0.0 | yes |
| R_Toe | negative_rocking | 60 | 0 | 0.0 | yes |
| R_Toe | rocking_plus_sliding | 60 | 0.2 | 1.0 | yes |
| R_Toe | airborne | 0 | 0 | None | yes |

No document/reality divergence found for this narrowly stated invocation and analytic control claim. This witness does not update old scores, relax thresholds, certify simulator contact dynamics, or write Overleaf.
