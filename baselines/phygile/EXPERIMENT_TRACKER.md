# PhyGile experiment tracker

|Item|Status|Evidence / boundary|
|---|---|---|
|Official release inspection|DONE|Pinned tracking commit d486216; project link is Code (Tracking); no published release/checkpoint found|
|Seeded CUDA kernel|DONE|results/kernel_witness.json|
|Fresh-agent simulation smoke|DONE|results/document_witness/report.md; 1 PPO update; checkpoint persisted|
|20-clip, seed-42 initial-expert training|DONE|results/pilot_train20_nostate_seed42/receipt.json; 500 updates; exit 0; 12,288,000 transitions|
|Initial-checkpoint evaluation entrypoint|DONE|results/eval_initial_check_metrics/metrics.json; not a quality claim|
|Predeclared 16-combination evaluation|DONE|results/evaluation_suite.json; all 16 exit 0; completion is not a performance claim|
|Raw metric recomputation|DONE|results/summary.json; results/evaluation_summary.csv; raw states and per-step CSVs|
|Checkpoint finite-state / expert change verification|DONE|results/checkpoint_verification.json; finite parameters; only expert 0 changed|
|Reliable tracking baseline|NOT_ACHIEVED|Final test success 2/128 (1.5625%); 4 clips from one held-out source group; one training seed|
|Full two-stage GMT|NOT_RUN|Matched curriculum data/labels and training budget not available for this run|
|Robot-native text diffusion generation|NOT_RUN|Full official code/weights not found in inspected release|
|Physics-prefix closed-loop adaptation|NOT_RUN|Full official implementation/inputs not available|
|G1 hardware deployment|OUT_OF_SCOPE|User requested simulation only|

No NOT_RUN or IN_PROGRESS item may be described as completed reproduction.
