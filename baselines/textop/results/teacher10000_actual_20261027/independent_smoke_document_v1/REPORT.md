# Fresh document-following 10k teacher smoke

Completed the documented `teacher10000_smoke_document_v1` invocation exactly once. Its actual SDK child and outer supervisor both exited **0**, the child was reaped, writers closed, and no timeout or supervisor cleanup signal occurred. The subsequent CPU-only read-only audit passed **47 verification groups**, including the documented `verify_smoke_witnesses` on an in-memory spec copy with the actual author/fresh receipt and exit hashes. The on-disk spec and activation fields were unchanged.

## Actual execution

- Durable driver **4146848**, supervisor **4146849**, wrapper/process group **4146856**, SDK worker **4146864**.
- Actual checkpoint **10,000 iterations / 200,000 optimizer updates**, SHA `6f949e4edc50d9a763a3634d716162cb5ddecb079bd1142d3720f907f5efab45`. This evaluation performed **0 optimizer updates**.
- **8 environments; 8/8 natural whole-motion completions; 0 failed trials; 63 vectorized control calls / 252 physics steps**. Across the eight trials: **450 control steps and 458 recorded poses**, source lengths **52–64 frames**. Supervisor duration **117.298749 seconds**.
- Receipt SHA `40569b36a083be653d5a587f2512133c07b3ae98a6e03433da2e257cc027726f`; exit SHA `7bb47228e2dc9881dd04882d57e14844947100b35bcc6d9cbceda60c1b9868eb`. Complete artifact hashes are in SUMMARY.json.
- The same original eight short motions selected for the 5k smoke were used. This is neither eight newly sampled motions nor the 64-motion evaluation.

## Source, physics and evidence checks

All seven bound pilot/runtime-helper sources, the warm environment hash `b7a3bf0c`, original robot URDF, **20 frozen training runtime bindings**, and **3 handoff-control source bindings** matched. The physical reset/control/recording core is byte-identical to the frozen 5k executor except the running receipt label; **26 numerical/resource settings** agree. Seed 20261026, strict pelvis 0.25 m/end-effector 0.5 m thresholds, orientation termination, 50 Hz control/4 substeps, 8-env smoke and resource caps remain unchanged. The frozen 8/64 selection metadata was verified without dispatching a sample stage.

Both author and fresh runs use the validated static solid floor, matching the teacher profile and identical actual generated collision-asset bytes: 4000 × 4000 × 0.2 m, top z=0, friction 1/1, restitution 0, self-collisions enabled. This retains the declared solid-floor representation; the author's exact original collision geometry is not claimed available.

The existing protocol verified private journal/strict/index/binary/NPZ hashes and one trial per selected source. The independent reader additionally compared **128 numeric field arrays across 16 author/fresh traces (916 poses)**, verified finite values, binary↔NPZ equality, 29-joint order against reference-export metadata, source joint targets, quaternion norms, and timestamps. Export metadata contains 30 bodies; the recorder's selected body set has 14. Each trace is initial frame 0 followed by consumed targets 0..steps−1, captured before autoreset; every natural completion has steps=T−1. The final reference frame T−1 is not an additional consumed control target.

Fresh current-target errors independently recomputed from private arrays: control-step-weighted 14-body mean **0.068624923 m**, maximum **0.375589222 m**, 29-joint RMSE **0.173582406 rad**. These are errors against consumed current control targets, not terminal-reference endpoint metrics or paper benchmark scores.

## Resource and shutdown evidence

Preflight found no existing document run, author child/outer exits 0 and closed writers, no personal SDK overlap, and the common lock free. Shared GPU1 free memory was **40,726 MiB**, available RAM **138,282,487,808 B**, and persistent free disk **5,915,152,384 B**. All ten fresh resource samples retained CPU affinity 0–3; minimum RAM **133420204032 B**, minimum free disk **5885915136 B**. Observed owned GPU peaks were **GPU1 11,062 MiB** and auxiliary **GPU0 454 MiB**; Torch peak **8,822,339,584 B**, with the bound Torch allocation fraction 0.65. These are sampled process peaks, not complete device-allocation maxima. The final sample has no owned processes or GPU allocations, the common lock is released, and old 57D PID 3237858 remains stopped.

**Native close did not return; the documented 20-second fallback was used after completed numerical receipts and closed writers.** Both stacks block inside SimulationContext.stop → callback → render. The author stack ends at `is_gpu_dynamics_enabled`; the fresh stack ends at `torch.cuda.set_device`. Neither proves that native `app.close` was reached. Both logs retain one `DriverShaderCacheManager::init() called without a shutdown()` error marker. No additional document/runtime divergence was found.

## Preserved audit errors and scope

The first CPU-only auditor exited 1 because it incorrectly equated 30 exported bodies with the 14 recorded bodies. The second exited 1 because it incorrectly required the author and fresh shutdown stacks to have the same leaf frame. Those verifier assumptions were corrected only in the read-only audit; both failures are retained in SUMMARY.json. A local read-only lookup also used a nonexistent exporter filename before locating the manifest-order definition in the existing comparison source. No sources, numerical settings, thresholds, environment packages, production processes or runtime receipts were changed, and no SDK invocation was repeated.

The original production handoff remains last-published **11,400**, recoverable **10,000 / 200,000**; original process/supervisor exits **0/0** and outer exit **1** are retained. No training signal was sent by this pass. Parent reports both closed smoke runs privately archived with SHA verification; this pass wrote only this report and its aggregate summary. No raw arrays, logs or per-source identities are published here.

All eight smoke trials completed, so this pass **does not exercise the physical-failure branch**. The other 8,270 originals were not physically evaluated here. Sample activation was still false at audit completion. No 64-trial comparison, whole-library acceptance, semantic-quality acceptance, or full-paper reproduction claim is made. Separate terminal-reference endpoint analysis remains unperformed in this pass.
