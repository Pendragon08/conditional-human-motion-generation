# PhyGile experiment tracker

|Item|Status|Evidence / boundary|
|---|---|---|
|Official release inspection|DONE|Pinned tracking commit d486216; project link is Code (Tracking); no published release/checkpoint found|
|Seeded CUDA kernel|DONE|results/kernel_witness.json|
|Fresh-agent simulation smoke|DONE|results/document_witness/report.md; 1 PPO iteration; checkpoint persisted|
|20-clip, seed-42 initial-expert training|DONE|results/pilot_train20_nostate_seed42/receipt.json; 500 PPO iterations; exit 0; 12,288,000 transitions|
|Initial-checkpoint evaluation entrypoint|DONE|results/eval_initial_check_metrics/metrics.json; not a quality claim|
|Predeclared 16-combination evaluation|DONE|results/evaluation_suite.json; all 16 exit 0; completion is not a performance claim|
|Raw metric recomputation|DONE|results/summary.json; results/evaluation_summary.csv; raw states and per-step CSVs|
|Checkpoint finite-state / expert change verification|DONE|results/checkpoint_verification.json; finite parameters; only expert 0 changed|
|Reliable general tracking baseline|NOT_ESTABLISHED|2000-iteration test success 26/128 (20.3125%); one source group and one training seed|
|Full two-stage GMT|NOT_RUN|Matched curriculum data/labels and training budget not available for this run|
|Robot-native text diffusion generator training|NOT_RUN_ON_REAL_MOTION_DATA|Independent core and training entrypoint exist; paired data and codec/mapping remain unavailable|
|Physics-prefix closed-loop adaptation|NOT_RUN|Full official implementation/inputs not available|
|G1 hardware deployment|OUT_OF_SCOPE|User requested simulation only|
|500-to-2000 continuation|DONE|results/continuation_2000/pipeline.json; 1500 extra PPO training iterations; fixed final checkpoint|
|Fixed 2000-iteration evaluation|DONE|results/continuation_2000/evaluation_suite.json; 8 combinations; same first-episode protocol|
|Continuation state accounting|DONE|results/continuation_2000/state_comparison.json; parent/final hashes and expert/optimizer changes|
|Paper-grounded generator core numerical checks|DONE_UNIT_CHECK_ONLY|results/reimplementation_core/verification.json; 12 checks; no motion-quality claim|
|Actual pretrained text interface|DONE_INTERFACE_CHECK_ONLY|results/reimplementation_text/verification.json; pinned DistilBERT; generator remains untrained on real motion data|
|Independent training entrypoint|DONE_PLUMBING_CHECK_ONLY|results/reimplementation_training_witness/verification.json; 2 CPU updates on explicitly synthetic tensors and a reduced model|
|Curriculum integrity repairs|DONE_REGRESSION_ONLY|results/prefix_integrity/document_run/regression/verification.json; 12 software checks; unsafe evaluators disabled; full curriculum not validated|
|Repaired-source simulation training check|DONE_RUNTIME_ONLY|results/prefix_integrity/document_run/one_ppo/receipt.json; separate 32-env × 1-PPO diagnostic, 768 transitions; curriculum off|
|Transactional physics-prefix orchestration|DONE_SYNTHETIC_INTERFACE_ONLY|reimplementation/physics_prefix.py; two-window synthetic rollback/commit tests; no real simulator snapshot adapter yet|
|Fresh pre-reset feedback capture and acceptance gate|DONE_SIMULATION_DIAGNOSTIC_ONLY|results/prefix_integrity/document_run/feedback_gate/verification.json; 1 training clip, 32 first episodes, 0/32 accepted 50-frame windows at 0.1 m; no generated candidate, no closed-loop PPO|

No NOT_RUN or IN_PROGRESS item may be described as completed reproduction.
