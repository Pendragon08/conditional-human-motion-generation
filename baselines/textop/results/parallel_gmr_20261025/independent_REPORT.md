# Independent latest-revision parallel GMR document witness

PASS: documented command executed once, actual exit 0. All 48 recorded independent checks passed. Eight complete CMU sample motions match serial and previous stored CMU outputs exactly. This run measured 2.36x speedup; no production gate or worker scheduling was changed by this witness.

## Exact run and actual exits

Read only PARALLEL_GMR_ENVIRONMENT.md, configs/gmr-parallel-spec.json and SERVER_NOTES.md. Executed the exact documented PowerShell SSH command once unchanged with --run document_v1. Local session 34745 returned actual exit_code=0; stdout and saved receipt report TEXTOP_PARALLEL_GMR_WITNESS_PASSED.

Serial child exit0; parallel controller child exit0. Both processed8 sources with0 failures. Parallel worker exit codes=[0,0,0,0], PIDs3651963/3651964/3651965/3651966. Both run statuses are PREPROCESSING_FINISHED_FILTER_PENDING.

No retry, code/package edit, repair, gate change, production GMR launch, upload/backup/FK action, worker stop, private array/manifest/checkpoint copy or publication occurred. All earlier author_v1 failure and v2/v3 lower-speed evidence were left untouched. This uses the latest source/spec hashes below.

## Numerical and metadata comparison

- diagnostic_source_count=8, canonical_top_level_outputs=8, total_output_pose_frames=5040.
- array_comparisons=112, metadata_field_comparisons=144.
- parallel_vs_serial maximum errors are0 for qpos_29, contacts, features_69, mirrored qpos/contact/features and times_s.
- new_serial_vs_previous_kernel maximum errors are0 for the same fields.
- contacts_exact=true, timestamps_exact=true, annotations_exact=true, disjoint_sources_verified=true.
- Tolerances remain atol1e-5/rtol1e-6; no tolerance was relaxed.
- Both logs report8 completed originals/8 mirrors,29DOF,69D,50fps, zero failures, train5/val3 and178 paired segments.

The public aggregate flags/errors are backed by the canonical witness source. Private array values and source identifiers were not inspected or copied. No implementation/AST inspection was performed by this agent.

## Four shared CPUs and declared thread behavior

Each of the four worker PIDs was observed in the receipt's resource samples with exact affinity [0,1,2,3]. All sampled job processes remain within CPU0–3; shared_four_core_worker_affinity_verified=true. There are94 serial and40 parallel resource samples. The command's taskset also binds the outer driver to the same CPU set.

The current spec declares four workers sharing these four cores, Torch/MKL/OMP4 per worker, OpenBLAS1, OMP_WAIT_POLICY=PASSIVE and KMP_BLOCKTIME=0. Parallel log reports threads_per_worker4. The outer launcher environment remains exactly as documented; it was not changed to expand resources. Lower-level per-worker BLAS/OpenMP environment snapshots are not separate public receipt fields; identity of the latest source/spec supplies that configuration evidence.

CUDA_VISIBLE_DEVICES is empty in the documented invocation/spec and both final logs report cpu_only=true. This does not authorize16 CPU cores or a GPU workload.

## Actual timing

| Measurement | Seconds |
|---|---:|
| Serial child wall time | 95.36418103054166 |
| Parallel child wall time | 40.40935499034822 |
| Parallel dispatcher internal time | 40.09039897657931 |

Measured speedup=2.3599530616937443x, computed from the two comparable child wall times. The internal dispatcher time excludes a small part of launch/supervision overhead and is not used as the denominator.

The document reports author_v4 approximately2.41x; this independent sample reports2.36x. Timing varies and the procedure explicitly allows reporting actual observed performance. No numerical/resource/hash divergence was found. These eight complete CMU motions do not guarantee4x speedup, full-cohort duration or throughput under different contention.

## Source/spec identity

Each hash matches the document, public receipt and independent remote sha256sum, without reading implementation content:
- prepare_paper_data.py: 06539b1a7593ef45beda815d07252b4850f2c47237f2d0000c77bdcd60f101a5
- parallel_prepare_data.py: 79d532acd48f9ff3db2ea9096a2b6f211666c9cb3b1581b1ea3d37f49d88ec47
- gmr_parallel_witness.py: a0ae6fc5e4dd6570219ed723e21c25f02cfa8e5d451a262775171229c4ca16a0
- gmr-parallel-spec.json: b797d14305c7459a926573033eedd355e506a9722e1db0dec6d6fa063198a657
- data-prep-env-spec.json: 6e7f21352c2839d62498a068103cfedcd3949f5416b1ca075477f6bdbef94591

Both exact local spec hashes also match. The spec references preserved original baseline90db2afdb9baf3521356c7995c95d2cd48bdc56836ffc4539073c4c8e4b819dc; comparison against its previously stored CMU outputs passes with maximum difference0. No old failed-profile timing is presented as current validation.

## Scope and saved evidence

physical_filter_executed=false, dataset_accepted=false, full_generator_training_started=false, outputs_private=true and output_writers_closed=true. Both logs keep physical_filter_passed=false and ready_for_generator_training=false.

These8 private diagnostic outputs are separate from full-corpus production counts. The witness did not activate WAIT_FOR_VALIDATION, restart production, or add these diagnostics to production totals. Parent scheduling decisions remain separate.

Saved files: documented-command.ps1,stdout.log,local-execution-receipts.jsonl,receipt.json,serial.log,parallel.log,source_and_specs_remote_sha256.txt,verification.json,evidence-readback-receipts.jsonl and REPORT.md.
